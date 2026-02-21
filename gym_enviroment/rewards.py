"""
Reward calculation module for negotiation environment.

Provides modular reward components that can be enabled/disabled via configuration.
"""

import numpy as np
from typing import TYPE_CHECKING, Optional
from gym_enviroment.config.config import config

if TYPE_CHECKING:
    from gym_enviroment.env import NegotiationEnv


class RewardCalculator:
    """Handles all reward calculations for the negotiation environment."""
    
    def __init__(self, env: 'NegotiationEnv'):
        self.env = env
        
    def get_total_reward(self) -> float:
        """Compute total reward using configuration-based component selection."""
        
        if not self.env.our_agent._is_forecasting_ready():
            return 0.0

        if self.env.done:
            return self._get_terminal_reward()
        else:
            return self._get_dense_reward()
    
    def _get_terminal_reward(self) -> float:
        """Compute terminal reward when episode ends."""
        
        # Check for early episode end punishment
        if config.rewards.get('use_early_end_punishment', True):
            early_threshold = config.rewards.get('early_end_threshold', 100)
            if self.env.current_round < early_threshold:
                # Episode ended too early - apply punishment instead of normal terminal reward
                early_penalty = config.rewards.get('early_end_penalty', -0.3)
                self.env.last_terminal_reward = early_penalty
                return early_penalty
        
        # Check if terminal rewards are enabled
        if not config.rewards.get('use_terminal_reward', True):
            return 0.0

        if not self.env.agreement_reached:
            terminal_reward = config.rewards.get('no_agreement_penalty', -0.5)
            self.env.last_terminal_reward = terminal_reward
            return terminal_reward

        # Get utilities for the agreed bid
        our_utility = self.env.our_preference.get_utility(self.env.last_agreed_bid)
        opponent_utility = self.env.opponent_preference.get_utility(self.env.last_agreed_bid)
        
        # New terminal reward logic to prevent early termination incentives:
        if our_utility < opponent_utility:
            # Minimize the utility difference when we're losing
            # This gives a continuous reward signal while keeping all rewards ≤ 0
            terminal_reward = -abs(opponent_utility - our_utility)
        else:
            # We didn't lose, so no punishment 
            terminal_reward = 0.0
            
        self.env.last_terminal_reward = terminal_reward
        return terminal_reward
    
    def _get_dense_reward(self) -> float:
        """Compute dense reward for each step using enabled components."""
        
        # Initialize component rewards
        nash_reward = 0.0
        strategy_fit_reward = 0.0
        
        # Calculate enabled reward components
        if config.rewards.get('use_nash_reward', True):
            nash_reward = self._get_nash_improvement_reward() * config.rewards.get('nash_weight', 1.0)
            
        if config.rewards.get('use_strategy_fit_reward', True):
            strategy_fit_reward = self._get_strategy_fit_reward() * config.rewards.get('strategy_fit_weight', 1.0)
            
        # Store detailed reward components for callback access  
        self.env.last_nash_reward = nash_reward 
        self.env.last_strategy_fit_reward = strategy_fit_reward
        
        # Sum enabled components
        total_dense_reward = nash_reward + strategy_fit_reward        
        return float(total_dense_reward)
    
    def _get_nash_improvement_reward(self) -> float:
        """Compute Nash utility improvement reward using forecasts."""
        baseline, updated = self.env.our_agent.nash_baseline_forecast, self.env.our_agent.updated_forecast
        if baseline is None or updated is None:
            return 0.0

        L = int(config.core['forecast_length'])
        N = int(config.core['simulation_bids_count'])

        # Align sequences on the common window [t+N, t+L]
        if L <= N:
            return 0.0
        
        baseline_slice = baseline[N:L]          # length L-N
        updated_slice = updated[0:L-N]          # length L-N

        if baseline_slice.shape[0] != updated_slice.shape[0] or baseline_slice.shape[0] == 0:
            return 0.0

        delta_avg = float(np.mean(updated_slice) - np.mean(baseline_slice))
        delta_avg = float(np.clip(delta_avg, -1.0, 1.0))
        return float(delta_avg) * ( self.env.current_round /  self.env.deadline_round) ** 2
    
    def _get_strategy_fit_reward(self) -> float:
        """Compute strategy fit reward based on utility difference between our bid and selected strategy's bid."""
        
        # Get the first strategy offer (what it would offer at current time)
        simulated_points = self.env.our_agent.selected_strategy_simulated_points
        strategy_offers = [point for point in simulated_points if point.who == 1]
        
        strategy_bid = strategy_offers[0].bid
        strategy_utility = self.env.our_preference.get_utility(strategy_bid)
        
        # Get our most recent bid (the one we just made) - much cleaner access!
        our_last_bid = self.env.last_our_bid
        our_utility = self.env.our_preference.get_utility(our_last_bid)

        # Calculate utility difference
        utility_difference = abs(our_utility - strategy_utility)
        
        # Apply exponential penalty: small differences (~0.01) barely penalized, large differences (~0.5) heavily penalized
        penalty = -1 * (utility_difference ** 2.5)
            
        # Scale by time factor to keep rewards reasonable
        return float(penalty * (1 / self.env.deadline_round))

