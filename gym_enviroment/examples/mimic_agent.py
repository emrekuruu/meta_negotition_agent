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

from agents.HybridAgent.HybridAgent import HybridAgent

from gym_enviroment.rl_agent import AbstractRLAgent
from gym_enviroment.reward import AbstractRewardFunction


class MimicAgent(AbstractRLAgent):
    """
    RL agent that learns to reproduce the bidding curve of a reference strategy.

    Observation space (1,):
        [0] t                      -- normalised negotiation time in [0, 1]

    Action space (1,):
        [0] normalized_target      -- normalized target in [-1, 1], mapped to
                                      utility target in [0, 1] for reward shaping

    The shadow agent (the one being mimicked) is kept perfectly in sync: it receives
    every opponent bid and its act() is called each round so its internal state
    (opponent models, time curves, etc.) evolves exactly as it would in a real session.
    """

    mimic_class: Type[AbstractAgent] = HybridAgent

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    @classmethod
    def get_observation_space(cls) -> gym.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

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
        self._last_t = 0.0
        self._shadow_accepted = False
        self._shadow_accept_round = None
        # Shadow instance — evolves in lockstep with the real negotiation
        self._shadow = self.mimic_class(self.preference, self.session_time, [])
        self._shadow.initiate(opponent_name)

    def receive_offer(self, bid: Bid, t: float) -> None:
        # Keep the shadow agent in sync so opponent-model-based strategies work correctly
        self._shadow.receive_bid(bid, t)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def set_action(self, action: np.ndarray) -> None:
        self._action = action
        raw = float(np.clip(action[0], -1.0, 1.0))
        self._target_utility = 0.5 * (raw + 1.0)

    def build_observation(self) -> np.ndarray:
        return np.array([self._last_t], dtype=np.float32)

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

    def dense_reward(self, env) -> float:

        if env.last_our_bid is None:
            return 0.0

        t = env.current_round / env.deadline_round
        our_utility = float(env.our_agent._target_utility)

        if our_utility > env.our_agent._mimic_target:
            return (env.our_agent._mimic_target / (our_utility + 0.00001))  * t
        else:
            return (our_utility / (env.our_agent._mimic_target + 0.00001)) 
        

    def terminal_reward(self, env) -> float:
        return 0.0
