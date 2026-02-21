import os
import tempfile
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class ArtifactOnlyCheckpoint(BaseCallback):
    """
    Every `save_freq` env steps:
      1) Save model to a temp zip
      2) Log it as a Weights & Biases artifact (alias with the step count)
      3) Delete the temp file
    """

    def __init__(self, save_freq: int, artifact_name: str = "ppo_negotiator", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.artifact_name = artifact_name

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            # 1) Dump model to a temporary zip
            fd, path = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            self.model.save(path)
            if self.verbose:
                print(f"[{self.num_timesteps}] checkpoint saved to {path}")

            # 2) Push as W&B artifact with a step alias
            art = wandb.Artifact(self.artifact_name, type="model")
            art.add_file(path)
            logged = wandb.log_artifact(art, aliases=[f"step_{self.num_timesteps}"])
            logged.wait()
            if self.verbose:
                print(f"[{self.num_timesteps}] logged artifact as version alias 'step_{self.num_timesteps}'")

            # 3) Remove the temporary file
            os.remove(path)
            if self.verbose:
                print(f"[{self.num_timesteps}] removed temp file")

        return True
