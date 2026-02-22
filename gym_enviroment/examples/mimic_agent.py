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

    Observation space (2 + 2 * BID_WINDOW + OPPONENT_TYPE_DIM,):
        [0] t                      -- normalised negotiation time in [0, 1]
        [1] last_our_offer_u       -- utility of our previous offer (own utility)
        [2:2+BID_WINDOW]           -- last BID_WINDOW opponent bid utilities
                                      in own-utility scale, zero-padded on the left
        [2+BID_WINDOW:2+2*BID_WINDOW]
                                   -- last BID_WINDOW estimated opponent-self utilities
                                      from BayesianOpponentModel, zero-padded on the left
        [2+2*BID_WINDOW:]          -- one-hot opponent type (+1 unknown bucket)

    Action space (1,):
        [0] normalized_target      -- normalized target in [-1, 1], mapped to
                                      utility target in [0, 1] for reward shaping

    The shadow agent (the one being mimicked) is kept perfectly in sync: it receives
    every opponent bid and its act() is called each round so its internal state
    (opponent models, time curves, etc.) evolves exactly as it would in a real session.
    """

    mimic_class: Type[AbstractAgent] = HybridAgent
    BID_WINDOW: int = 5
    OPPONENT_LABELS = [
        _normalize_opponent_label(name) for name in config.environment.get("opponents", [])
    ]
    OPPONENT_TYPE_DIM: int = len(OPPONENT_LABELS) + 1

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    @classmethod
    def get_observation_space(cls) -> gym.Space:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 + 2 * cls.BID_WINDOW + cls.OPPONENT_TYPE_DIM,),
            dtype=np.float32,
        )

    @classmethod
    def get_action_space(cls) -> gym.Space:
        # Symmetric range is better behaved for PPO exploration.
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------------
    # AbstractAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        if self.mimic_class is not None:
            return f"Mimic({self.mimic_class.__name__})"
        return "MimicAgent"

    def initiate(self, opponent_name: Optional[str]) -> None:
        self._action = None
        self._target_utility = 1.0
        self._mimic_target = 1.0
        self._last_our_offer_utility = 1.0
        self._last_t = 0.0
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
        self._shadow_accepted = False
        self._shadow_accept_round = None
        # Shadow instance — evolves in lockstep with the real negotiation
        self._shadow = self.mimic_class(self.preference, self.session_time, [])
        self._shadow.initiate(opponent_name)

    def receive_offer(self, bid: Bid, t: float) -> None:
        # Keep the shadow agent in sync so opponent-model-based strategies work correctly
        self._shadow.receive_bid(bid, t)
        self._opponent_model.update(bid, t)
        self._opp_estimated_utils.append(self._opponent_model.preference.get_utility(bid))

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def set_action(self, action: np.ndarray) -> None:
        self._action = action
        raw = float(np.clip(action[0], -1.0, 1.0))
        self._target_utility = 0.5 * (raw + 1.0)

    def build_observation(self) -> np.ndarray:
        opp_utils = [self.preference.get_utility(bid) for bid in self.last_received_bids]
        recent_our = opp_utils[-self.BID_WINDOW:]
        padded_recent_our = [0.0] * (self.BID_WINDOW - len(recent_our)) + recent_our

        recent_opp = self._opp_estimated_utils[-self.BID_WINDOW:]
        padded_recent_opp = [0.0] * (self.BID_WINDOW - len(recent_opp)) + recent_opp

        obs = [
            self._last_t,
            self._last_our_offer_utility,
            *padded_recent_our,
            *padded_recent_opp,
            *self._opponent_type_onehot,
        ]
        return np.array(obs, dtype=np.float32)

    def act(self, t: float) -> Action:
        self._last_t = t

        # Ask the shadow what it would bid — this also advances its internal state
        shadow_action = self._shadow.act(t)
        if isinstance(shadow_action, Accept):
            self._shadow_accepted = True
            if self._shadow_accept_round is None:
                self._shadow_accept_round = round(t * self.session_time)
        else:
            self._mimic_target = self.preference.get_utility(shadow_action.bid)
            self._last_our_offer_utility = self._mimic_target

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
        self._dense_reward_per_episode_length = 0.0

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
        return reward
        

    def terminal_reward(self, env) -> float:
        acceptance_round = env.current_round if env.agreement_reached else env.deadline_round
        self._dense_reward_per_episode_length = self._episode_dense_total / max(acceptance_round, 1)
        return 0.0

    def get_log_info(self) -> dict:
        info = super().get_log_info()
        info["dense_reward_total_episode"] = self._episode_dense_total
        info["dense_reward_per_episode_length"] = self._dense_reward_per_episode_length
        return info
