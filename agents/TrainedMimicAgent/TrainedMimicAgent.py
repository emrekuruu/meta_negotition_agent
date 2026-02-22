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
    Tournament agent that runs a recurrent PPO policy trained by MimicAgent.

    Set artifact_path to your W&B artifact before running a tournament:

        TrainedMimicAgent.artifact_path = "entity/project/checkpoint-run-name:latest"

    The model is loaded once on first initiate() and shared across all instances.
    """

    artifact_path: str = "emre-kuru-zye-in-niversitesi/negotiation-rl/checkpoint-deep-night-4:latest"
    _model: ClassVar = None
    MAX_HISTORY: ClassVar[int] = int(config.environment.get("deadline_round", 100))
    OPPONENT_LABELS: ClassVar[list[str]] = [
        _normalize_opponent_label(name) for name in config.environment.get("opponents", [])
    ]
    OPPONENT_TYPE_DIM: ClassVar[int] = len(OPPONENT_LABELS) + 1

    @property
    def name(self) -> str:
        return "TrainedMimicAgent"

    @classmethod
    def _flatten_history(cls, history: list[tuple[float, float, float]]) -> list[float]:
        padded = [[0.0, 0.0, 0.0] for _ in range(cls.MAX_HISTORY)]
        keep = history[: cls.MAX_HISTORY]
        for idx, (our_u, opp_u, t_offer) in enumerate(keep):
            padded[idx] = [
                float(np.clip(our_u, 0.0, 1.0)),
                float(np.clip(opp_u, 0.0, 1.0)),
                float(np.clip(t_offer, 0.0, 1.0)),
            ]
        return [x for row in padded for x in row]

    def initiate(self, opponent_name: Optional[str]) -> None:
        if TrainedMimicAgent._model is None:
            import wandb
            from sb3_contrib import RecurrentPPO

            api = wandb.Api()
            artifact = api.artifact(self.artifact_path)
            model_dir = artifact.download()
            model_files = sorted(_glob.glob(f"{model_dir}/*.zip"))

            if not model_files:
                raise FileNotFoundError(f"No model zip found in artifact directory: {model_dir}")

            TrainedMimicAgent._model = RecurrentPPO.load(model_files[-1])

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

        self._lstm_state = None
        self._episode_start = np.ones((1,), dtype=bool)

        self._current_t = 0.0
        self._opponent_offer_history: list[tuple[float, float, float]] = []
        self._our_offer_history: list[tuple[float, float, float]] = []

        self._opponent_model = BayesianOpponentModel(self.preference)

    def receive_offer(self, bid: Bid, t: float) -> None:
        self._opponent_model.update(bid, t)
        self._current_t = float(np.clip(t, 0.0, 1.0))

        our_u = float(self.preference.get_utility(bid))
        opp_u = float(self._opponent_model.preference.get_utility(bid))
        self._opponent_offer_history.append((our_u, opp_u, self._current_t))

    def act(self, t: float) -> Action:
        self._current_t = float(np.clip(t, 0.0, 1.0))

        opponent_hist = self._flatten_history(self._opponent_offer_history)
        our_hist = self._flatten_history(self._our_offer_history)
        obs = np.array(
            [self._current_t] + self._opponent_type_onehot + opponent_hist + our_hist,
            dtype=np.float32,
        )

        action, self._lstm_state = self._model.predict(
            obs,
            state=self._lstm_state,
            episode_start=self._episode_start,
            deterministic=True,
        )
        self._episode_start[0] = False

        raw = float(np.clip(action[0], -1.0, 1.0))
        target = float(np.clip(0.5 * (raw + 1.0), self.preference.reservation_value, 1.0))
        bid = self.preference.get_bid_at(target)
        if self.can_accept() and bid <= self.last_received_bids[-1]:
            return self.accept_action

        predicted_opp_u = float(self._opponent_model.preference.get_utility(bid))
        self._our_offer_history.append((float(bid.utility), predicted_opp_u, self._current_t))
        return Offer(bid)
