import wandb
import tempfile
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from gym_enviroment.env import NegotiationEnv
from gym_enviroment.agent import MainStrategy
from gym_enviroment.callbacks import (
    ArtifactOnlyCheckpoint,
    RewardTrackingCallback,
    CoefficientTrackingCallback,
    StrategyTrackingCallback,
    UtilityTrackingCallback,
)
from gym_enviroment.custom_policy import MultiInputPolicy
from gym_enviroment.config.config import config

# 1. Configuration
CONFIG = {
    "algorithm": "PPO",
    "domains": config.environment["domains"],
    "opponents": config.environment["opponents"],
    "deadline_round": config.environment["deadline_round"],
    "checkpoint_freq": config.logging["checkpoint_freq"],
    "learning_rate": config.training["learning_rate"],
    "n_steps": config.training["n_steps"],         
    "batch_size": config.training["batch_size"],
    "n_epochs": config.training["n_epochs"],
    "gamma": config.training["gamma"],
    "clip_range": config.training["clip_range"],
    "ent_coef": config.training["ent_coef"],
    "target_kl": config.training["target_kl"],
    "device": config.core["device"],
    "n_envs": int(config.environment["n_envs"]),          
    "start_method": "spawn",
}

def make_negotiation_env(domains, opponents, deadline_round, seed=0):
    env = NegotiationEnv(
        our_agent_class=MainStrategy,
        domains=domains,
        deadline_round=deadline_round,
        opponent_names=opponents,
    )
    env.reset(seed=seed)
    return Monitor(env)

def make_env_fn(rank, domains, opponents, deadline_round, base_seed):
    def _thunk():
        assigned_opponent = [opponents[rank % len(opponents)]]
        return make_negotiation_env(domains, assigned_opponent, deadline_round, seed=base_seed + rank)
    return _thunk

def main():

    TOTAL_TIMESTEPS = (
        len(CONFIG["domains"])
        * len(CONFIG["opponents"])
        * CONFIG["deadline_round"]
        * 60
    )

    print(f"Total timesteps: {TOTAL_TIMESTEPS}", flush=True)

    CONFIG["TOTAL_TIMESTEPS"] = TOTAL_TIMESTEPS

    # 3. W&B
    wandb.init(
        project="negotiation-rl",
        config=CONFIG,
        sync_tensorboard=True,
    )
    
    # Get the W&B run name and create run-specific log directory
    run_name = wandb.run.name
    run_log_dir = f"episode_logs/{run_name}"
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Created log directory for run: {run_log_dir}")

    # 4. Check a single raw env once
    print("Creating environment for check_env...")
    _probe_env = NegotiationEnv(
        our_agent_class=MainStrategy,
        domains=CONFIG["domains"],
        deadline_round=CONFIG["deadline_round"],
        opponent_names=CONFIG["opponents"],
    )
    print("Checking environment...")
    check_env(_probe_env)
    _probe_env.close()

    base_seed = int(config.core.get("seed", 0))

    # Vectorized envs
    print(f"Creating {CONFIG['n_envs']} parallel environments...")
    
    # Log opponent assignment for each environment
    for i in range(CONFIG["n_envs"]):
        assigned_opponent = CONFIG["opponents"][i % len(CONFIG["opponents"])]
        print(f"Environment {i}: Training against {assigned_opponent}")
    
    venv = SubprocVecEnv(
        [make_env_fn(i, CONFIG["domains"], CONFIG["opponents"], CONFIG["deadline_round"], base_seed)
         for i in range(CONFIG["n_envs"])],
        start_method=CONFIG["start_method"],
    )

    # Observation and reward normalization
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_obs=10.0)

    with tempfile.TemporaryDirectory() as temp_dir:
        vecnorm_path = f"{temp_dir}/vecnorm.pkl"

        episode_freq = 10

        model = PPO(
            MultiInputPolicy,
            venv,
            learning_rate=CONFIG["learning_rate"],
            n_steps=CONFIG["n_steps"],         # steps per env
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            gamma=CONFIG["gamma"],
            clip_range=CONFIG["clip_range"],
            ent_coef=CONFIG["ent_coef"],
            verbose=1,
            device=CONFIG["device"],
            tensorboard_log=f"{temp_dir}/tb_logs",
            target_kl=CONFIG["target_kl"],
        )

        checkpoint_cb = ArtifactOnlyCheckpoint(
            save_freq=CONFIG["checkpoint_freq"],
            artifact_name=f"ppo_negotiator_{wandb.run.name}",
        )
        
        reward_tracking_cb = RewardTrackingCallback(
            max_round=CONFIG["deadline_round"],
            episode_freq=episode_freq,   # Async I/O allows more frequent logging
            log_dir=f"{run_log_dir}/reward_tracking", 
            verbose=0
        )

        strategy_tracking_cb = StrategyTrackingCallback(
            episode_freq=episode_freq,   # Async I/O - no training slowdown
            log_dir=f"{run_log_dir}/strategy_tracking",
            verbose=0
        )   

        utility_tracking_cb = UtilityTrackingCallback(
            episode_freq=episode_freq,   # Async I/O - background processing
            log_dir=f"{run_log_dir}/utility_tracking",
            verbose=0
        )

        coefficient_tracking_cb = CoefficientTrackingCallback(
            episode_freq=episode_freq,  # Most data-heavy, slightly less frequent
            log_dir=f"{run_log_dir}/coefficient_tracking",
            verbose=0
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, reward_tracking_cb, strategy_tracking_cb, utility_tracking_cb, coefficient_tracking_cb],
        )

if __name__ == "__main__":
    main()
