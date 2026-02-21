import wandb
import torch
import itertools
import numpy as np
from typing import List, Type

import gymnasium as gym
from gymnasium.spaces import Box
from nenv import domain_loader
from nenv.utils.DynamicImport import load_agent_class
from nenv import AbstractAgent, Accept
from gym_enviroment.config.config import config
from gym_enviroment.custom_policy import get_default_observation_space
from gym_enviroment.rewards import RewardCalculator

class NegotiationEnv(gym.Env):
    """
    Gymnasium environment for negotiation where RL agent controls 
    strategy selection and Nash-based multivariate bidding equations.
    
    Observation Space:
        Dictionary with two components:
        - 'negotiation_history': (96, 12) - Sequential negotiation features
        - 'candidate_forecasts': (5, 336) - Future utility forecasts per candidate
    
    Action Space:
        Box(98) - [strategy_index, w1, w2, ..., w97]
        - strategy_index: Which predefined strategy to use (0-4)
        - w1...w96: Weights for Nash utilities from 96 history points  
        - w97: Weight for current time feature
    
    Reward:
        Final utility achieved by our agent at episode end
    """

    def __init__(self, our_agent_class: Type[AbstractAgent], domains: List[str], deadline_round: int, opponent_names: List[str], mode: str = "oracle"):
        super().__init__()

        # Store device from config
        self.device = torch.device(config.core['device'])
        
        self.our_agent_class = our_agent_class
        self.domains = domains
        self.deadline_round = deadline_round
        self.opponent_names = opponent_names
        self.mode = mode

        # Load opponent classes
        self.opponents = [load_agent_class(path) for path in opponent_names]
        self.combinations = list(itertools.product(self.domains, self.opponents))
        self.combination_index = 0

        # Action space setup for dual utility targeting multivariate equations
        max_history_points = int(config.core['max_history_points'])  # 96 history points
        
        # New dual utility targeting: separate our_utility and opponent_utility features
        # Each equation has: our_utility_features (96) + opponent_utility_features (96) + time (1) = 193
        features_per_equation = 2 * max_history_points + 1  # 193
        total_features = 2 * features_per_equation  # 386 for both equations
        
        # Action vector: [strategy_index, our_weights(193), opp_weights(193)]
        # - strategy_index: continuous in [0, max_candidates-1], rounded at use
        # - our_weights(193): weights for our utility equation [our_utils(96), opp_utils(96), time(1)]
        # - opp_weights(193): weights for opponent utility equation [our_utils(96), opp_utils(96), time(1)]
        # Total action space: 1 + 193 + 193 = 387 dimensions
        coeff_bound = 50.0  # Allow reasonable range for learning
        low = np.array([0.0] + [-coeff_bound] * total_features, dtype=np.float32)
        high = np.array([float(config.core['max_candidates']) - 1.0] + [coeff_bound] * total_features, dtype=np.float32)
        
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        
        # Observation space: multi-input dictionary structure
        self.observation_space = get_default_observation_space()

        # Episode state
        self.our_agent = None
        self.opponent_agent = None
        self.our_preference = None
        self.opponent_preference = None
        self.done = False
        self.final_utility = 0.0
        self.current_round = 0
        self.last_opponent_bid = None
        self.last_our_bid = None
        self.episode_count = 0  # Track episodes for logging
        self.agreement_reached = False  # Direct tracking of agreement status
        
        # Initialize reward calculator
        self.reward_calculator = None  # Will be set after agent initialization
        
    def _get_obs(self):
        """Get current observation in dictionary format."""
        obs = self.our_agent.build_observation()
        
        # Ensure observations are in correct format (numpy arrays)
        # The custom policy will handle device conversion
        for key, value in obs.items():
            if not isinstance(value, np.ndarray):
                obs[key] = np.array(value, dtype=np.float32)
            else:
                obs[key] = value.astype(np.float32)
        
        return obs

    def _get_info(self):
        """Get environment info."""
        current_opponent = "unknown"
        if hasattr(self, 'opponent_agent') and self.opponent_agent:
            current_opponent = self.opponent_agent.__class__.__name__
        
        info = {
            "domain": self.combinations[self.combination_index][0] if hasattr(self, 'combinations') else "unknown",
            "opponent": current_opponent,
            "round": self.current_round,
            "time_left": 1.0 - (self.current_round / self.deadline_round),
            "nash_dense": getattr(self, 'last_nash_reward', 0.0),
            "strategy_fit": getattr(self, 'last_strategy_fit_reward', 0.0),
            "terminal_reward": getattr(self, 'last_terminal_reward', 0.0),
            "agreement_reached": getattr(self, 'agreement_reached', False)
        }
        
        # Add utility information for callbacks if bids are available
        if hasattr(self, 'our_preference') and hasattr(self, 'opponent_preference'):
            # Add our bid info if available
            if hasattr(self, 'last_our_bid') and self.last_our_bid is not None:
                info["our_bid"] = self.last_our_bid
                info["our_utility_our_bid"] = self.our_preference.get_utility(self.last_our_bid)
                info["opp_utility_our_bid"] = self.opponent_preference.get_utility(self.last_our_bid)
            
            # Add opponent bid info if available
            if hasattr(self, 'last_opponent_bid') and self.last_opponent_bid is not None:
                info["opp_bid"] = self.last_opponent_bid
                info["opp_utility_opp_bid"] = self.opponent_preference.get_utility(self.last_opponent_bid)
                info["our_utility_opp_bid"] = self.our_preference.get_utility(self.last_opponent_bid)
        
        return info

    def _get_reward(self):            
        total_reward = self.reward_calculator.get_total_reward()
        self.last_total_reward = total_reward
        return total_reward
        
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Cycle through domain-opponent combinations
        domain_name, opponent_class = self.combinations[self.combination_index]
        self.combination_index = (self.combination_index + 1) % len(self.combinations)

        # Load domain preferences
        self.our_preference, self.opponent_preference = domain_loader(domain_name)
        
        # Initialize agents
        if self.mode == "oracle":
            self.our_agent = self.our_agent_class(self.our_preference, self.opponent_preference, self.deadline_round, self.mode)
        else:
            self.our_agent = self.our_agent_class(self.our_preference, self.deadline_round, [])
       
        self.opponent_agent = opponent_class(self.opponent_preference, self.deadline_round, [])
        
        # Initialize agents
        self.our_agent.initiate(self.opponent_agent.name)
        self.opponent_agent.initiate(self.our_agent.name)
        
        # Initialize reward calculator after agent setup
        self.reward_calculator = RewardCalculator(self)
        
        # Track episode count for logging
        self.episode_count += 1
        
        # Reset episode state
        self.done = False
        self.final_utility = 0.0
        self.current_round = 0
        self.last_opponent_bid = None
        self.last_our_bid = None
        self.last_agreed_bid = None  # Track the bid that was agreed upon
        self.agreement_reached = False  # Reset agreement status
        
        # Initialize reward components for callback tracking
        self.last_nash_reward = 0.0
        self.last_strategy_fit_reward = 0.0
        self.last_terminal_reward = 0.0
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one step in the environment."""

        # Get current time for agents
        t = self.current_round / self.deadline_round
        
        # Our agent receives opponent's last bid (if any)
        if self.last_opponent_bid is not None:
            self.our_agent.receive_bid(self.last_opponent_bid, t)
        
        self.our_agent.set_rl_action(action)
        
        # Our agent acts
        our_action = self.our_agent.act(t)
        
        # Check if our agent accepted
        if isinstance(our_action, Accept):
            self.done = True
            self.agreement_reached = True
            self.final_utility = self.our_preference.get_utility(our_action.bid)
            self.last_agreed_bid = our_action.bid
        else:
            self.last_our_bid = our_action.bid
            self.opponent_agent.receive_bid(our_action.bid, t)
            opponent_action = self.opponent_agent.act(t)

            # Check if opponent accepted
            if isinstance(opponent_action, Accept):
                self.done = True
                self.agreement_reached = True
                self.final_utility = self.our_preference.get_utility(opponent_action.bid)
                self.last_agreed_bid = opponent_action.bid
            else:
                # Store opponent's bid for next round
                self.last_opponent_bid = opponent_action.bid    
                
        # Update round
        self.current_round += 1
        
        # Check if deadline reached
        if self.current_round >= self.deadline_round:
            self.done = True
            self.final_utility = self.our_preference.reservation_value
        
        # Calculate reward to ensure terminal reward is set if episode is done
        reward = self._get_reward()
        
        return self._get_obs(), reward, self.done, False, self._get_info()

    def render(self):
        """Render method removed for performance."""
        pass