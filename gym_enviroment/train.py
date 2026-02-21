import os

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_enviroment.config.config import config
from gym_enviroment.env import NegotiationEnv

# ------------------------------------------------------------------
# Plug in your agent and reward function here
# ------------------------------------------------------------------
from gym_enviroment.examples.mimic_agent import MimicAgent as MyRLAgent, MimicReward as MyRewardFunction


CONFIG = {
    "algorithm": "PPO",
    "domains": config.environment["domains"],
    "opponents": config.environment["opponents"],
    "deadline_round": int(config.environment["deadline_round"]),
    "n_envs": int(config.environment.get("n_envs", 1)),
    "seed": int(config.core.get("seed", 0)),
    "learning_rate": config.training["learning_rate"],
    "n_steps": int(config.training["n_steps"]),
    "batch_size": int(config.training["batch_size"]),
    "n_epochs": int(config.training["n_epochs"]),
    "gamma": config.training["gamma"],
    "clip_range": config.training["clip_range"],
    "ent_coef": config.training["ent_coef"],
    "target_kl": config.training["target_kl"],
    "total_timesteps": int(config.training.get("total_timesteps", 200_000)),
    "device": config.core["device"],
}


class ArtifactCheckpointCallback(BaseCallback):
    def __init__(self, run, checkpoint_freq: int, save_path: str):
        super().__init__()
        self.run = run
        self.checkpoint_freq = int(checkpoint_freq)
        self.save_path = save_path
        self._last_checkpoint = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.checkpoint_freq <= 0:
            return True
        if self.num_timesteps - self._last_checkpoint < self.checkpoint_freq:
            return True

        self._last_checkpoint = self.num_timesteps
        checkpoint_base = os.path.join(self.save_path, f"model_{self.num_timesteps}")
        self.model.save(checkpoint_base)

        artifact = wandb.Artifact(
            name=f"checkpoint-{self.run.id}-{self.num_timesteps}",
            type="model",
            metadata={"timestep": self.num_timesteps},
        )
        artifact.add_file(f"{checkpoint_base}.zip")
        self.run.log_artifact(artifact)
        return True


def make_env_fn(rank: int):
    def _thunk():
        env = NegotiationEnv(
            our_agent_class=MyRLAgent,
            domains=CONFIG["domains"],
            deadline_round=CONFIG["deadline_round"],
            opponent_names=CONFIG["opponents"],
            reward_fn=MyRewardFunction(),
        )
        env.reset(seed=CONFIG["seed"] + rank)
        return Monitor(env)

    return _thunk


def build_vec_env():
    env_fns = [make_env_fn(rank) for rank in range(CONFIG["n_envs"])]
    return DummyVecEnv(env_fns)


def main():
    print(f"Total timesteps: {CONFIG['total_timesteps']}", flush=True)

    run = wandb.init(
        project=config.logging["wandb"]["project"],
        entity=config.logging["wandb"].get("entity"),
        config=CONFIG,
        sync_tensorboard=True,
        save_code=True,
    )

    probe_env = NegotiationEnv(
        our_agent_class=MyRLAgent,
        domains=CONFIG["domains"],
        deadline_round=CONFIG["deadline_round"],
        opponent_names=CONFIG["opponents"],
        reward_fn=MyRewardFunction(),
    )
    check_env(probe_env)
    probe_env.close()

    vec_env = build_vec_env()

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        gamma=CONFIG["gamma"],
        clip_range=CONFIG["clip_range"],
        ent_coef=CONFIG["ent_coef"],
        target_kl=CONFIG["target_kl"],
        verbose=1,
        seed=CONFIG["seed"],
        device=CONFIG["device"],
        tensorboard_log=f"{run.dir}/tb",
    )

    checkpoint_freq = int(config.logging.get("checkpoint_freq", 0))
    callbacks = []
    if checkpoint_freq > 0:
        callbacks.append(ArtifactCheckpointCallback(
            run=run,
            checkpoint_freq=checkpoint_freq,
            save_path=f"{run.dir}/checkpoints",
        ))

    callbacks.append(WandbCallback(
        gradient_save_freq=0,
        model_save_freq=0,
        verbose=0,
    ))

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=CallbackList(callbacks),
    )
    final_model_path = f"{run.dir}/final_model"
    model.save(final_model_path)

    artifact = wandb.Artifact(
        name=f"model-{run.id}",
        type="model",
        metadata={"total_timesteps": CONFIG["total_timesteps"]},
    )
    artifact.add_file(f"{final_model_path}.zip")
    run.log_artifact(artifact)

    vec_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
