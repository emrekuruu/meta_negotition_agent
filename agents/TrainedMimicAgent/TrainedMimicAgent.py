import glob as _glob
from typing import Optional, ClassVar

import numpy as np

from nenv import Offer
from nenv.Action import Action
from nenv.Agent import AbstractAgent
from nenv.Bid import Bid
from nenv.OpponentModel import BayesianOpponentModel
from gym_enviroment.config.config import config


def _normalize_opponent_label(label: str) -> str:
    normalized = "".join(ch for ch in label.lower() if ch.isalnum())
    if normalized.endswith("agent"):
        normalized = normalized[:-5]
    return normalized


class TrainedMimicAgent(AbstractAgent):
    """
    Tournament agent that runs a PPO policy trained by MimicAgent.

    Set artifact_path to your W&B artifact before running a tournament:

        TrainedMimicAgent.artifact_path = "entity/project/model-run-name:latest"

    The model is loaded once on first initiate() and shared across all sessions.
    """

    artifact_path: str = "emre-kuru-zye-in-niversitesi/negotiation-rl/checkpoint-b5szwzuy-300000:v0"
    _model: ClassVar = None  # loaded once, shared across all instances
    BID_WINDOW: ClassVar[int] = 5
    OPPONENT_LABELS: ClassVar[list[str]] = [
        _normalize_opponent_label(name) for name in config.environment.get("opponents", [])
    ]
    OPPONENT_TYPE_DIM: ClassVar[int] = len(OPPONENT_LABELS) + 1

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
        self._opponent_type_onehot = [0.0] * self.OPPONENT_TYPE_DIM
        if opponent_name is not None:
            normalized = _normalize_opponent_label(opponent_name)
            try:
                idx = self.OPPONENT_LABELS.index(normalized)
            except ValueError:
                idx = self.OPPONENT_TYPE_DIM - 1
            self._opponent_type_onehot[idx] = 1.0
        else:
            self._opponent_type_onehot[self.OPPONENT_TYPE_DIM - 1] = 1.0
        self._opp_estimated_utils = []
        self._opponent_model = BayesianOpponentModel(self.preference)

    def receive_offer(self, bid: Bid, t: float) -> None:
        self._opponent_model.update(bid, t)
        self._opp_estimated_utils.append(self._opponent_model.preference.get_utility(bid))

    def act(self, t: float) -> Action:
        opp_utils = [bid.utility for bid in self.last_received_bids]
        recent_our = opp_utils[-self.BID_WINDOW:]
        padded_recent_our = [0.0] * (self.BID_WINDOW - len(recent_our)) + recent_our

        recent_opp = self._opp_estimated_utils[-self.BID_WINDOW:]
        padded_recent_opp = [0.0] * (self.BID_WINDOW - len(recent_opp)) + recent_opp

        obs = np.array(
            [t, self._last_our_offer_utility] + padded_recent_our + padded_recent_opp + self._opponent_type_onehot,
            dtype=np.float32,
        )

        action, _ = self._model.predict(obs, deterministic=True)
        raw = float(np.clip(action[0], -1.0, 1.0))
        target = float(np.clip(0.5 * (raw + 1.0), self.preference.reservation_value, 1.0))
        bid = self.preference.get_bid_at(target)
        if self.can_accept() and bid <= self.last_received_bids[-1]:
            return self.accept_action
        self._last_our_offer_utility = bid.utility
        return Offer(bid)
