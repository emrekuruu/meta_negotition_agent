"""
Your RL negotiation agent.

Fill in each method to define:
  - What the policy observes  (get_observation_space + build_observation)
  - What the policy outputs   (get_action_space + set_action)
  - How the agent negotiates  (act + receive_offer + initiate)
  - How the agent is rewarded (MyRewardFunction)
  - What gets logged to W&B   (get_extra_info)
"""

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from nenv import Offer
from nenv.Action import Action
from nenv.Bid import Bid
from nenv.OpponentModel import AbstractOpponentModel
from nenv.Preference import Preference

from gym_enviroment.rl_agent import AbstractRLAgent
from gym_enviroment.reward import AbstractRewardFunction


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

class MyRLAgent(AbstractRLAgent):

    @property
    def name(self) -> str:
        return "MyRLAgent"

    # ------------------------------------------------------------------
    # Spaces — define what the policy sees and outputs
    # ------------------------------------------------------------------

    @classmethod
    def get_observation_space(cls) -> gym.Space:
        raise NotImplementedError(
            "Define the observation space. Example:\n"
            "  return spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)"
        )

    @classmethod
    def get_action_space(cls) -> gym.Space:
        raise NotImplementedError(
            "Define the action space. Example:\n"
            "  return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)"
        )

    # ------------------------------------------------------------------
    # RL hooks
    # ------------------------------------------------------------------

    def build_observation(self) -> np.ndarray:
        """Convert current negotiation state to an observation array."""
        raise NotImplementedError(
            "Build an observation from self.preference, self.last_received_bids, etc."
        )

    def set_action(self, action: np.ndarray) -> None:
        """Store the action chosen by the RL policy for use in act()."""
        self._action = action

    # ------------------------------------------------------------------
    # Negotiation logic
    # ------------------------------------------------------------------

    def initiate(self, opponent_name: Optional[str]) -> None:
        self._action = None
        # Add any per-episode initialisation here

    def receive_offer(self, bid: Bid, t: float) -> None:
        # Called when the opponent makes an offer.
        # Update any internal state (e.g. opponent model) here.
        pass

    def act(self, t: float) -> Action:
        """Decide what to do this round using self._action."""
        raise NotImplementedError(
            "Use self._action and self.preference to return an Offer or Accept.\n\n"
            "Example (bid at a target utility from the action):\n"
            "  target = float(np.clip(self._action[0], 0.0, 1.0))\n"
            "  bid = self.preference.get_bid_at(target)\n"
            "  if self.can_accept() and bid <= self.last_received_bids[-1]:\n"
            "      return self.accept_action\n"
            "  return Offer(bid)"
        )


# ----------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------

class MyRewardFunction(AbstractRewardFunction):

    def on_reset(self, env) -> None:
        # Reset any per-episode state here (baselines, counters, etc.)
        pass

    def dense_reward(self, env) -> float:
        """Reward at every non-terminal step."""
        raise NotImplementedError(
            "Return a float reward for each negotiation step.\n\n"
            "Available: env.our_preference, env.last_our_bid, env.last_opponent_bid,\n"
            "           env.current_round, env.deadline_round, env.our_agent"
        )

    def terminal_reward(self, env) -> float:
        """Reward at the end of an episode (agreement or deadline)."""
        raise NotImplementedError(
            "Return a float terminal reward.\n\n"
            "Available: env.agreement_reached, env.final_utility, env.last_agreed_bid,\n"
            "           env.our_preference, env.opponent_preference"
        )

