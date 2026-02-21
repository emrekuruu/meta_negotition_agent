import itertools
from typing import List, Type

import numpy as np
import gymnasium as gym

from nenv import domain_loader
from nenv.utils.DynamicImport import load_agent_class
from nenv import Accept

from gym_enviroment.rl_agent import AbstractRLAgent
from gym_enviroment.reward import AbstractRewardFunction


class NegotiationEnv(gym.Env):
    """
    Gymnasium environment for training an RL negotiation agent.

    The RL agent is defined by your AbstractRLAgent subclass, which controls
    what the policy sees (observation space) and what it outputs (action space).
    The reward signal is defined by your AbstractRewardFunction subclass.

    Episode flow:
        reset() -- picks a domain/opponent pair, initialises agents
        step(action) -- our agent receives the action, acts, opponent responds
        done -- True when agreement is reached or deadline expires

    State available to reward functions and agents each step:
        our_preference       -- own utility function
        opponent_preference  -- opponent's utility function
        last_our_bid         -- our most recent Bid (None before first offer)
        last_opponent_bid    -- opponent's most recent Bid
        current_round        -- current round index
        deadline_round       -- total rounds allowed
        agreement_reached    -- True if a deal was made
        final_utility        -- our utility at end of episode
        last_agreed_bid      -- the agreed Bid (None if no agreement)
    """

    def __init__(
        self,
        our_agent_class: Type[AbstractRLAgent],
        domains: List[str],
        deadline_round: int,
        opponent_names: List[str],
        reward_fn: AbstractRewardFunction,
    ):
        super().__init__()

        self.our_agent_class = our_agent_class
        self.domains = domains
        self.deadline_round = deadline_round
        self.opponent_names = opponent_names
        self.reward_fn = reward_fn

        # Load opponent classes and build the cycling combination list
        self.opponents = [load_agent_class(path) for path in opponent_names]
        self.combinations = list(itertools.product(self.domains, self.opponents))
        self.combination_index = 0

        # Spaces are declared by the agent class — no hardcoding here
        self.observation_space = our_agent_class.get_observation_space()
        self.action_space = our_agent_class.get_action_space()

        # Episode state — set properly in reset()
        self.our_agent = None
        self.opponent_agent = None
        self.our_preference = None
        self.opponent_preference = None
        self.done = False
        self.final_utility = 0.0
        self.current_round = 0
        self.last_opponent_bid = None
        self.last_our_bid = None
        self.last_agreed_bid = None
        self.agreement_reached = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        return self.our_agent.build_observation()

    def _get_info(self):
        domain_name, opponent_class = self.combinations[self.combination_index - 1]
        return {
            "round": self.current_round,
            "t": self.current_round / self.deadline_round,
            "domain": domain_name,
            "opponent": opponent_class.__name__,
            "agreement_reached": self.agreement_reached,
            "final_utility": self.final_utility,
        }

    def _get_reward(self):
        return self.reward_fn.compute(self)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Cycle through domain × opponent combinations
        domain_name, opponent_class = self.combinations[self.combination_index]
        self.combination_index = (self.combination_index + 1) % len(self.combinations)

        # Load preferences for this domain
        self.our_preference, self.opponent_preference = domain_loader(domain_name)

        # Instantiate agents
        self.our_agent = self.our_agent_class(self.our_preference, self.deadline_round, [])
        self.opponent_agent = opponent_class(self.opponent_preference, self.deadline_round, [])

        self.our_agent.initiate(self.opponent_agent.name)
        self.opponent_agent.initiate(self.our_agent.name)

        # Reset episode state
        self.done = False
        self.final_utility = 0.0
        self.current_round = 0
        self.last_opponent_bid = None
        self.last_our_bid = None
        self.last_agreed_bid = None
        self.agreement_reached = False

        # Give the reward function a chance to reset per-episode state
        self.reward_fn.on_reset(self)

        return self._get_obs(), self._get_info()

    def step(self, action):
        t = self.current_round / self.deadline_round

        # Forward opponent's last bid to our agent (if any)
        if self.last_opponent_bid is not None:
            self.our_agent.receive_bid(self.last_opponent_bid, t)

        # Give the RL action to our agent, then let it act
        self.our_agent.set_action(action)
        our_action = self.our_agent.act(t)

        if isinstance(our_action, Accept):
            self.done = True
            self.agreement_reached = True
            self.last_agreed_bid = our_action.bid
            self.final_utility = self.our_preference.get_utility(our_action.bid)
        else:
            self.last_our_bid = our_action.bid
            self.opponent_agent.receive_bid(our_action.bid, t)
            opponent_action = self.opponent_agent.act(t)

            if isinstance(opponent_action, Accept):
                self.done = True
                self.agreement_reached = True
                self.last_agreed_bid = opponent_action.bid
                self.final_utility = self.our_preference.get_utility(opponent_action.bid)
            else:
                self.last_opponent_bid = opponent_action.bid

        self.current_round += 1

        if self.current_round >= self.deadline_round:
            self.done = True
            self.final_utility = self.our_preference.reservation_value

        reward = self._get_reward()

        return self._get_obs(), reward, self.done, False, self._get_info()

    def render(self):
        pass
