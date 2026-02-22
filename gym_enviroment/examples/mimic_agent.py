"""
MimicAgent — example RL agent that learns to mimic a reference negotiation strategy.
"""

from typing import Optional, Type

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from nenv import Offer, Accept
from nenv.Action import Action
from nenv.Agent import AbstractAgent
from nenv.Bid import Bid
from nenv.OpponentModel import BayesianOpponentModel

from agents.HybridAgent.HybridAgent import HybridAgent

from gym_enviroment.config.config import config
from gym_enviroment.rl_agent import AbstractRLAgent
from gym_enviroment.reward import AbstractRewardFunction


def _normalize_opponent_label(label: str) -> str:
    normalized = "".join(ch for ch in label.lower() if ch.isalnum())
    if normalized.endswith("agent"):
        normalized = normalized[:-5]
    return normalized


class MimicAgent(AbstractRLAgent):
    """
    RL agent that learns to reproduce the bidding curve of a reference strategy.

    Observation space:
        [current_t,
         opponent_type_one_hot...,
         opponent_offer_history...,  # MAX_HISTORY x (our_u, opp_u_est, t_offer)
         our_offer_history...]       # MAX_HISTORY x (our_u, opp_u_est, t_offer)

    Histories are chronological and zero-padded to MAX_HISTORY, where
    MAX_HISTORY == deadline_round.
    """

    mimic_class: Type[AbstractAgent] = HybridAgent
    MAX_HISTORY: int = int(config.environment.get("deadline_round", 100))
    OPPONENT_LABELS = [
        _normalize_opponent_label(name) for name in config.environment.get("opponents", [])
    ]
    OPPONENT_TYPE_DIM: int = len(OPPONENT_LABELS) + 1
    HISTORY_FEATURES_PER_ENTRY: int = 3  # our_u, predicted_opp_u, t

    @classmethod
    def get_observation_space(cls) -> gym.Space:
        history_size = cls.MAX_HISTORY * cls.HISTORY_FEATURES_PER_ENTRY
        obs_dim = 1 + cls.OPPONENT_TYPE_DIM + history_size + history_size
        return spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    @classmethod
    def get_action_space(cls) -> gym.Space:
        # Symmetric range is better behaved for PPO exploration.
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    @property
    def name(self) -> str:
        if self.mimic_class is not None:
            return f"Mimic({self.mimic_class.__name__})"
        return "MimicAgent"

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
        self._action = None
        self._target_utility = 1.0
        self._mimic_target = 1.0
        self._current_t = 0.0

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

        self._opponent_offer_history: list[tuple[float, float, float]] = []
        self._our_offer_history: list[tuple[float, float, float]] = []

        self._opponent_model = BayesianOpponentModel(self.preference)
        self._shadow_accepted = False
        self._shadow_accept_round = None

        # Shadow instance — evolves in lockstep with the real negotiation.
        self._shadow = self.mimic_class(self.preference, self.session_time, [])
        self._shadow.initiate(opponent_name)

    def receive_offer(self, bid: Bid, t: float) -> None:
        # Keep the shadow agent in sync so opponent-model-based strategies work correctly.
        self._shadow.receive_bid(bid, t)
        self._opponent_model.update(bid, t)
        self._current_t = float(np.clip(t, 0.0, 1.0))

        our_u = float(self.preference.get_utility(bid))
        opp_u = float(self._opponent_model.preference.get_utility(bid))
        self._opponent_offer_history.append((our_u, opp_u, self._current_t))

    def set_action(self, action: np.ndarray) -> None:
        self._action = action
        raw = float(np.clip(action[0], -1.0, 1.0))
        self._target_utility = 0.5 * (raw + 1.0)

    def build_observation(self) -> np.ndarray:
        opponent_hist = self._flatten_history(self._opponent_offer_history)
        our_hist = self._flatten_history(self._our_offer_history)
        obs = [self._current_t, *self._opponent_type_onehot, *opponent_hist, *our_hist]
        return np.array(obs, dtype=np.float32)

    def act(self, t: float) -> Action:
        self._current_t = float(np.clip(t, 0.0, 1.0))

        # Ask the shadow what it would bid — this also advances its internal state.
        shadow_action = self._shadow.act(t)
        if isinstance(shadow_action, Accept):
            self._shadow_accepted = True
            if self._shadow_accept_round is None:
                self._shadow_accept_round = round(t * self.session_time)
        else:
            bid = shadow_action.bid
            our_u = float(self.preference.get_utility(bid))
            opp_u = float(self._opponent_model.preference.get_utility(bid))
            self._our_offer_history.append((our_u, opp_u, self._current_t))
            self._mimic_target = our_u

        self._target_utility = self.preference.get_bid_at(self._target_utility).utility
        return shadow_action


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def make_mimic_agent(strategy_class: Type[AbstractAgent]) -> Type[MimicAgent]:
    """
    Return a MimicAgent subclass configured to mimic *strategy_class*.

    Example
    -------
        from agents.boulware.Boulware import BoulwareAgent
        MimicBoulware = make_mimic_agent(BoulwareAgent)
    """

    class _MimicAgent(MimicAgent):
        mimic_class = strategy_class

        @property
        def name(self) -> str:
            return f"Mimic({strategy_class.__name__})"

    _MimicAgent.__name__ = f"Mimic{strategy_class.__name__}"
    _MimicAgent.__qualname__ = f"Mimic{strategy_class.__name__}"
    return _MimicAgent


# ------------------------------------------------------------------
# Reward
# ------------------------------------------------------------------

class MimicReward(AbstractRewardFunction):
    """
    Reward function paired with MimicAgent.
    """

    def on_reset(self, env) -> None:
        super().on_reset(env)
        self._episode_dense_total = 0.0
        self._episode_dense_rewards = []
        self._normalized_total_dense_reward = 0.0

    def dense_reward(self, env) -> float:
        if env.last_our_bid is None:
            return 0.0

        t = env.current_round / env.deadline_round
        our_utility = float(env.our_agent._target_utility)

        if our_utility > env.our_agent._mimic_target:
            reward = (env.our_agent._mimic_target / (our_utility + 0.00001)) * t
        else:
            reward = (our_utility / (env.our_agent._mimic_target + 0.00001)) * t

        self._episode_dense_total += reward
        self._episode_dense_rewards.append(reward)
        return reward

    def terminal_reward(self, env) -> float:
        acceptance_round = env.current_round if env.agreement_reached else env.deadline_round
        weight_sum = sum(round_idx / env.deadline_round for round_idx in range(1, acceptance_round + 1))
        if weight_sum > 0.0:
            self._normalized_total_dense_reward = sum(self._episode_dense_rewards) / weight_sum
        else:
            self._normalized_total_dense_reward = 0.0
        return 0.0

    def get_log_info(self) -> dict:
        info = super().get_log_info()
        info["dense_reward_total_episode"] = self._episode_dense_total
        info["normalized_total_dense_reward"] = self._normalized_total_dense_reward
        return info
