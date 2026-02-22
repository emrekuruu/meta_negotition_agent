import glob as _glob
from typing import Optional, ClassVar

import numpy as np

from nenv import Offer
from nenv.Action import Action
from nenv.Agent import AbstractAgent
from nenv.Bid import Bid


class TrainedMimicAgent(AbstractAgent):
    """
    Tournament agent that runs a PPO policy trained by MimicAgent.

    Set artifact_path to your W&B artifact before running a tournament:

        TrainedMimicAgent.artifact_path = "entity/project/model-run-name:latest"

    The model is loaded once on first initiate() and shared across all sessions.
    """

    artifact_path: str = "emre-kuru-zye-in-niversitesi/negotiation-rl/checkpoint-9iwphhmq-100000:v0"
    _model: ClassVar = None  # loaded once, shared across all instances
    BID_WINDOW: ClassVar[int] = 5

    @property
    def name(self) -> str:
        return "TrainedMimicAgent"

    def initiate(self, opponent_name: Optional[str]) -> None:
        if TrainedMimicAgent._model is None:
            import wandb
            from stable_baselines3 import PPO

            api = wandb.Api()
            artifact = api.artifact(self.artifact_path)
            model_dir = artifact.download()
            model_files = sorted(_glob.glob(f"{model_dir}/*.zip"))
            TrainedMimicAgent._model = PPO.load(model_files[-1])

        # Mirror training-time observation semantics.
        self._last_our_offer_utility = 1.0

    def receive_offer(self, bid: Bid, t: float) -> None:
        # No-op: this policy currently uses only time and bid-utility history.
        pass

    def act(self, t: float) -> Action:
        opp_utils = [bid.utility for bid in self.last_received_bids]
        recent = opp_utils[-self.BID_WINDOW:]
        padded_recent = [0.0] * (self.BID_WINDOW - len(recent)) + recent
        obs = np.array([t, self._last_our_offer_utility] + padded_recent, dtype=np.float32)

        action, _ = self._model.predict(obs, deterministic=True)
        raw = float(np.clip(action[0], -1.0, 1.0))
        target = float(np.clip(0.5 * (raw + 1.0), self.preference.reservation_value, 1.0))
        bid = self.preference.get_bid_at(target)
        if self.can_accept() and bid <= self.last_received_bids[-1]:
            return self.accept_action
        self._last_our_offer_utility = bid.utility
        return Offer(bid)
