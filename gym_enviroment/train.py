import os
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from gym_enviroment.env import NegotiationEnv
from gym_enviroment.config.config import config

# ------------------------------------------------------------------
# Plug in your agent and reward function here
# ------------------------------------------------------------------
from gym_enviroment.my_agent import MyRLAgent, MyRewardFunction


CONFIG = {
    "algorithm": "PPO",
    "domains": config.environment["domains"],
    "opponents": config.environment["opponents"],
    "deadline_round": config.environment["deadline_round"],
    "learning_rate": config.training["learning_rate"],
    "n_steps": config.training["n_steps"],
    "batch_size": config.training["batch_size"],
    "n_epochs": config.training["n_epochs"],
    "gamma": config.training["gamma"],
    "clip_range": config.training["clip_range"],
    "ent_coef": config.training["ent_coef"],
    "target_kl": config.training["target_kl"],
    "n_envs": int(config.environment["n_envs"]),
    "seed": int(config.core.get("seed", 0)),
}


def make_env_fn(rank, domains, opponents, deadline_round, base_seed):
    def _thunk():
        reward_fn = MyRewardFunction()
        env = NegotiationEnv(
            our_agent_class=MyRLAgent,
            domains=domains,
            deadline_round=deadline_round,
            opponent_names=[opponents[rank % len(opponents)]],
            reward_fn=reward_fn,
        )
        env.reset(seed=base_seed + rank)
        return Monitor(env)
    return _thunk


def main():
    TOTAL_TIMESTEPS = (
        len(CONFIG["domains"])
        * len(CONFIG["opponents"])
        * CONFIG["deadline_round"]
        * 60
    )
    CONFIG["total_timesteps"] = TOTAL_TIMESTEPS
    print(f"Total timesteps: {TOTAL_TIMESTEPS}", flush=True)

    wandb.init(
        project=config.logging["wandb"]["project"],
        config=CONFIG,
        sync_tensorboard=True,
    )

    run_log_dir = f"episode_logs/{wandb.run.name}"
    os.makedirs(run_log_dir, exist_ok=True)

    # Validate the environment against the Gymnasium API once before training
    _probe_env = NegotiationEnv(
        our_agent_class=MyRLAgent,
        domains=CONFIG["domains"],
        deadline_round=CONFIG["deadline_round"],
        opponent_names=CONFIG["opponents"],
        reward_fn=MyRewardFunction(),
    )
    check_env(_probe_env)
    _probe_env.close()

    venv = SubprocVecEnv(
        [make_env_fn(i, CONFIG["domains"], CONFIG["opponents"], CONFIG["deadline_round"], CONFIG["seed"])
         for i in range(CONFIG["n_envs"])],
        start_method="spawn",
    )
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        model = PPO(
            "MlpPolicy",            # swap for your custom policy class if needed
            venv,
            learning_rate=CONFIG["learning_rate"],
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            gamma=CONFIG["gamma"],
            clip_range=CONFIG["clip_range"],
            ent_coef=CONFIG["ent_coef"],
            target_kl=CONFIG["target_kl"],
            verbose=1,
            tensorboard_log=f"{tmp}/tb_logs",
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)


if __name__ == "__main__":
    main()
