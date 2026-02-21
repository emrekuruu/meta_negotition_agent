from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gym_enviroment.env import NegotiationEnv


class AbstractRewardFunction(ABC):
    """
    Base class for RL negotiation reward functions.

    Extend this class to define your own reward signal. You must implement:
        - dense_reward: reward given at every non-terminal step
        - terminal_reward: reward given at the final step of an episode

    Optionally override:
        - on_reset: called at the start of each episode for per-episode state setup

    The env object provides:
        env.our_preference       -- own utility function (Preference)
        env.opponent_preference  -- opponent's utility function (Preference)
        env.last_our_bid         -- our most recent Bid (None before first offer)
        env.last_opponent_bid    -- opponent's most recent Bid (None before first offer)
        env.current_round        -- current negotiation round (int)
        env.deadline_round       -- total rounds in this episode (int)
        env.agreement_reached    -- True if deal was made (bool)
        env.final_utility        -- our utility at agreement or reservation value (float)
        env.last_agreed_bid      -- the agreed Bid (None if no agreement)
        env.our_agent            -- the AbstractRLAgent instance (for agent-specific state)
    """

    def on_reset(self, env: 'NegotiationEnv') -> None:
        """
        Called at the start of each episode after agents are initialized.

        Override to reset per-episode state (e.g. baselines, running statistics).
        """
        pass

    def compute(self, env: 'NegotiationEnv') -> float:
        """Dispatches to dense_reward or terminal_reward based on env.done."""
        if env.done:
            return self.terminal_reward(env)
        return self.dense_reward(env)

    @abstractmethod
    def dense_reward(self, env: 'NegotiationEnv') -> float:
        """
        Reward signal for every non-terminal negotiation step.

        Return a float. Positive values encourage the behaviour that produced
        this step; negative values discourage it.
        """
        raise NotImplementedError

    @abstractmethod
    def terminal_reward(self, env: 'NegotiationEnv') -> float:
        """
        Reward signal for the final step of an episode.

        Called when env.done is True — either because agreement was reached
        (env.agreement_reached == True) or the deadline expired.
        """
        raise NotImplementedError
