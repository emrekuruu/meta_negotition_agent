import warnings

import wandb
from wandb.integration.sb3 import WandbCallback
from sklearn.exceptions import ConvergenceWarning

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_enviroment.callbacks import ArtifactCheckpointCallback, RolloutNormalizedDenseCallback
from gym_enviroment.config.config import config
from gym_enviroment.env import NegotiationEnv

# ------------------------------------------------------------------
# Plug in your agent and reward function here
# ------------------------------------------------------------------
from gym_enviroment.examples.mimic_agent import MimicAgent as MyRLAgent, MimicReward as MyRewardFunction

# Silence sklearn GP bound-convergence spam from some built-in opponent agents.
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"sklearn\.gaussian_process\.kernels",
)


CONFIG = {
    "algorithm": "RecurrentPPO",
    "domains": config.environment["domains"],
    "opponents": config.environment["opponents"],
    "deadline_round": int(config.environment["deadline_round"]),
    "n_envs": int(config.environment.get("n_envs", 1)),
    "seed": int(config.core.get("seed", 0)),
    "learning_rate": config.training["learning_rate"],
    "rollout_buffer_size": int(
        config.training.get(
            "rollout_buffer_size",
            int(config.training.get("n_steps", 256)) * int(config.environment.get("n_envs", 1)),
        )
    ),
    "batch_size": int(config.training["batch_size"]),
    "n_epochs": int(config.training["n_epochs"]),
    "gamma": config.training["gamma"],
    "clip_range": config.training["clip_range"],
    "ent_coef": config.training["ent_coef"],
    "target_kl": config.training["target_kl"],
    "total_timesteps": int(config.training.get("total_timesteps", 200_000)),
    "device": config.core["device"],
    "policy_hidden_sizes": config.training.get("policy_hidden_sizes", [128, 128, 64]),
    "lstm_hidden_size": int(config.training.get("lstm_hidden_size", 128)),
    "n_lstm_layers": int(config.training.get("n_lstm_layers", 1)),
    "shared_lstm": bool(config.training.get("shared_lstm", False)),
    "enable_critic_lstm": bool(config.training.get("enable_critic_lstm", True)),
}


def _use_subproc_vec_env() -> bool:
    return CONFIG["n_envs"] > 1


def _validate_rollout_config() -> None:
    rollout_buffer_size = CONFIG["rollout_buffer_size"]
    n_envs = CONFIG["n_envs"]
    batch_size = CONFIG["batch_size"]

    if rollout_buffer_size <= 0:
        raise ValueError("training.rollout_buffer_size must be > 0")
    if n_envs <= 0:
        raise ValueError("environment.n_envs must be > 0")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")

    if rollout_buffer_size % n_envs != 0:
        raise ValueError(
            "training.rollout_buffer_size must be divisible by environment.n_envs. "
            f"Got rollout_buffer_size={rollout_buffer_size}, n_envs={n_envs}."
        )

    n_steps = rollout_buffer_size // n_envs
    CONFIG["n_steps"] = n_steps

    if batch_size > rollout_buffer_size:
        raise ValueError(
            f"training.batch_size ({batch_size}) cannot exceed rollout buffer size "
            f"({rollout_buffer_size})."
        )
    if rollout_buffer_size % batch_size != 0:
        raise ValueError(
            "training.rollout_buffer_size must be divisible by training.batch_size for clean PPO minibatches. "
            f"Got rollout_buffer_size={rollout_buffer_size}, batch_size={batch_size}."
        )


def make_env_fn(rank: int):
    def _thunk():
        env = NegotiationEnv(
            our_agent_class=MyRLAgent,
            domains=CONFIG["domains"],
            deadline_round=CONFIG["deadline_round"],
            opponent_names=CONFIG["opponents"],
            reward_fn=MyRewardFunction(),
        )
        return Monitor(env, info_keywords=("normalized_total_dense_reward",))

    return _thunk


def build_vec_env():
    env_fns = [make_env_fn(rank) for rank in range(CONFIG["n_envs"])]
    if _use_subproc_vec_env():
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env.seed(CONFIG["seed"])
    return vec_env


def main():
    if RecurrentPPO is None:
        raise ImportError(
            "RecurrentPPO requires sb3-contrib. Install it with: pip install sb3-contrib"
        )

    _validate_rollout_config()
    print(f"Total timesteps: {CONFIG['total_timesteps']}", flush=True)
    print(
        f"Vec env: {'subproc' if _use_subproc_vec_env() else 'dummy'} (n_envs={CONFIG['n_envs']}), "
        f"rollout buffer={CONFIG['rollout_buffer_size']}, n_steps={CONFIG['n_steps']}, "
        f"lstm_hidden_size={CONFIG['lstm_hidden_size']}, n_lstm_layers={CONFIG['n_lstm_layers']}",
        flush=True,
    )

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
    policy_hidden_sizes = [int(size) for size in CONFIG["policy_hidden_sizes"]]
    policy_kwargs = {
        "net_arch": {
            "pi": policy_hidden_sizes,
            "vf": policy_hidden_sizes,
        },
        "lstm_hidden_size": CONFIG["lstm_hidden_size"],
        "n_lstm_layers": CONFIG["n_lstm_layers"],
        "shared_lstm": CONFIG["shared_lstm"],
        "enable_critic_lstm": CONFIG["enable_critic_lstm"],
    }

    model = RecurrentPPO(
        "MlpLstmPolicy",
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
        policy_kwargs=policy_kwargs,
    )

    checkpoint_freq = int(config.logging.get("checkpoint_freq", 0))
    callbacks = []
    if checkpoint_freq > 0:
        callbacks.append(ArtifactCheckpointCallback(
            run=run,
            checkpoint_freq=checkpoint_freq,
            save_path=f"{run.dir}/checkpoints",
        ))

    callbacks.append(RolloutNormalizedDenseCallback())

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
