"""
Training callbacks used by gym_enviroment.train.

This module keeps training orchestration logic in one place so the training entrypoint
can stay focused on environment/model setup.
"""

import os
from typing import Dict

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


def _artifact_safe_name(value: str) -> str:
    """
    Convert an arbitrary run name into a W&B-artifact-safe identifier.

    W&B artifact names allow alphanumeric characters and a small set of symbols.
    This helper normalizes everything else to '-' and avoids empty names.
    """
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in value.strip())
    safe = safe.strip("-_.")
    return safe or "run"


class RolloutNormalizedDenseCallback(BaseCallback):
    """
    Logs rollout-level metrics and coverage visualizations.

    Responsibilities:
    1. Track completed episodes by opponent/domain for coverage monitoring.
    2. Record rollout mean of `normalized_total_dense_reward` from SB3 episode info.
    3. Emit W&B bar charts showing episode distribution across opponents/domains.
    """

    def __init__(self):
        super().__init__()
        self._episode_count_by_opponent: Dict[str, int] = {}
        self._episode_count_by_domain: Dict[str, int] = {}

    def _on_step(self) -> bool:
        """
        Update coverage counters whenever an environment in the vectorized batch
        finishes an episode.
        """
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if not done:
                continue

            opponent = str(info.get("opponent", "unknown"))
            domain = str(info.get("domain", "unknown"))

            self._episode_count_by_opponent[opponent] = self._episode_count_by_opponent.get(opponent, 0) + 1
            self._episode_count_by_domain[domain] = self._episode_count_by_domain.get(domain, 0) + 1

        return True

    def _log_coverage_bars(self) -> None:
        """
        Log coverage as W&B bar charts.

        W&B bar charts are table-backed, so we create small summary tables from the
        cumulative episode counters and log the generated bar plots.
        """
        if self._episode_count_by_opponent:
            opponent_table = wandb.Table(columns=["opponent", "episodes"])
            for opponent, count in sorted(self._episode_count_by_opponent.items(), key=lambda item: item[0]):
                opponent_table.add_data(opponent, count)
            wandb.log(
                {
                    "coverage/episodes_by_opponent": wandb.plot.bar(
                        opponent_table,
                        "opponent",
                        "episodes",
                        title="Episode Coverage by Opponent",
                    )
                },
                step=self.num_timesteps,
            )

        if self._episode_count_by_domain:
            domain_table = wandb.Table(columns=["domain", "episodes"])
            for domain, count in sorted(self._episode_count_by_domain.items(), key=lambda item: item[0]):
                domain_table.add_data(domain, count)
            wandb.log(
                {
                    "coverage/episodes_by_domain": wandb.plot.bar(
                        domain_table,
                        "domain",
                        "episodes",
                        title="Episode Coverage by Domain",
                    )
                },
                step=self.num_timesteps,
            )

    def _on_rollout_end(self) -> None:
        """
        Compute rollout mean of normalized dense reward and log coverage plots.

        SB3 keeps recent episode summaries in `ep_info_buffer` when Monitor wrapper
        writes episode info. We aggregate from that buffer to publish a rollout mean.
        """
        ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
        if not ep_info_buffer:
            return

        values = [
            ep_info["normalized_total_dense_reward"]
            for ep_info in ep_info_buffer
            if "normalized_total_dense_reward" in ep_info
        ]
        if values:
            self.logger.record("rollout/normalized_total_dense_reward_mean", safe_mean(values))

        self._log_coverage_bars()


class ArtifactCheckpointCallback(BaseCallback):
    """
    Saves model checkpoints to disk and logs them to W&B as artifact versions.

    Unlike one-artifact-per-step naming, this callback uses one run-scoped artifact
    name and logs each checkpoint as a new version to reduce W&B artifact clutter.
    """

    def __init__(self, run, checkpoint_freq: int, save_path: str):
        super().__init__()
        self.run = run
        self.checkpoint_freq = int(checkpoint_freq)
        self.save_path = save_path
        self.artifact_name = f"checkpoint-{_artifact_safe_name(run.name or run.id)}"
        self._last_checkpoint = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Save/log checkpoint when checkpoint frequency is reached.
        """
        if self.checkpoint_freq <= 0:
            return True
        if self.num_timesteps - self._last_checkpoint < self.checkpoint_freq:
            return True

        self._last_checkpoint = self.num_timesteps
        checkpoint_base = os.path.join(self.save_path, f"model_{self.num_timesteps}")
        self.model.save(checkpoint_base)

        artifact = wandb.Artifact(
            name=self.artifact_name,
            type="model",
            metadata={"timestep": self.num_timesteps},
        )
        artifact.add_file(f"{checkpoint_base}.zip")
        self.run.log_artifact(artifact, aliases=["latest", f"step-{self.num_timesteps}"])
        return True
