import os
import nenv
import numpy as np
from typing import Union, Dict
from nenv import Action, Bid

from .pareto import ParetoWalker

opponent_model_map = {
    "ClassicFrequencyOpponentModel": nenv.OpponentModel.ClassicFrequencyOpponentModel,
    "CUHKOpponentModel": nenv.OpponentModel.CUHKOpponentModel,
    "BayesianOpponentModel": nenv.OpponentModel.BayesianOpponentModel,
    "WindowedFrequencyOpponentModel": nenv.OpponentModel.WindowedFrequencyOpponentModel,
    "ConflictBasedOpponentModel": nenv.OpponentModel.ConflictBasedOpponentModel,
    "RegressionCOMBOpponentModel": nenv.OpponentModel.RegressionCOMBOpponentModel,
    "StepwiseCOMBOpponentModel": nenv.OpponentModel.StepwiseCOMBOpponentModel,
    "ExpectationCOMBOpponentModel": nenv.OpponentModel.ExpectationCOMBOpponentModel,
}

class ParetoWalkerAgent(nenv.AbstractAgent):

    def __init__(self, preference: nenv.Preference, session_time: int, estimators=None, mode: str = "oracle", estimated_preference: nenv.Preference = None, **kwargs):
        super().__init__(preference, session_time, estimators or [])

        # Base Configuration
        self.mode = mode
        self.deadline = session_time
        self.estimated_preference = estimated_preference

        if self.mode != "oracle":

            opponent_model_class = os.getenv("OPPONENT_MODEL")
            
            self.opponent_model = opponent_model_map[opponent_model_class](
                self.preference,
            )

            print(type(self.opponent_model))

        # Pareto state
        self.pareto_state = None
        self.pareto_update_frequency = 99999 if self.mode == "oracle" else self.deadline // 80
        self.bid_space = None

        # Store initial desired point (can be overridden)
        self.set_bid_space()

        if self.mode != "oracle":
            self.initial_desired_point = {
                'our_utility': 0.9,
                'opponent_utility': 0.6
            }
        else:
            self.initial_desired_point = {
                'our_utility': self.get_nash_bid().utility_a,
                'opponent_utility': self.get_nash_bid().utility_b
            }

    @property
    def name(self) -> str:
        return "ParetoWalkerAgent"

    def initiate(self, opponent_name: Union[None, str]):
        self.pareto_state = self._initialize_pareto_state()

    def receive_offer(self, bid: Bid, t: float):
        """Process received offer."""
        # Update opponent model
        if self.mode != "oracle":
            self.opponent_model.update(bid, t)

    # ============================================================================
    # Low-level Helpers
    # ============================================================================

    def _initialize_pareto_state(self):
        """Initialize or reset Pareto state."""
        return {
            'pareto_front': None,
            'last_pareto_update': 0,
            'pareto_index': -1,
            'current_pareto_point': None,
            'main_strategy_starting_t': -1,
            'target_time': 1.0,
            'update_frequency': self.pareto_update_frequency,
            'ball_index': 0,
            'last_ball_point': None,
            'desired_bid': None,
            'desired_utility_point': self.initial_desired_point.copy(),
            'desired_index': None,
        }

    def _get_opponent_preference(self) -> nenv.Preference:
        """Get opponent preference."""
        if self.mode == "oracle":
            return self.estimated_preference
        else:
            return self.opponent_model.preference

    def set_bid_space(self):
        opponent_pref = self._get_opponent_preference()
        bid_space = nenv.BidSpace(self.preference, opponent_pref)
        self.bid_space = bid_space
    
    def get_nash_bid(self):
        return self.bid_space.nash_point

    def _sigmoid_weight(self, t: float, steepness: float = 10.0, midpoint: float = 0.5) -> float:
        """
        Calculate sigmoid weight for transitioning from initial to Nash.

        Args:
            t: Current normalized time (0 to 1)
            steepness: How steep the transition is (higher = steeper)
            midpoint: When the transition happens (0.5 = halfway through negotiation)

        Returns:
            Weight from 0 to 1 (0 = use initial desired, 1 = use Nash)
        """
        return 1 / (1 + np.exp(-steepness * (t - midpoint)))

    def set_desired_utility_point(self, our_utility: float, opponent_utility: float):
        """
        Set the initial desired utility point.

        Args:
            our_utility: Initial desired utility for us
            opponent_utility: Initial desired utility for opponent
        """
        self.initial_desired_point = {
            'our_utility': our_utility,
            'opponent_utility': opponent_utility
        }
        # Update pareto state if already initialized
        if self.pareto_state:
            self.pareto_state['desired_utility_point'] = self.initial_desired_point.copy()

    def _update_adaptive_desired_point(self, t: float) -> Dict[str, float]:
        """
        Adaptively update desired point based on time and Nash equilibrium.
        Transitions from initial desired point to Nash as negotiation progresses.

        Args:
            t: Current normalized time (0 to 1)

        Returns:
            Updated desired utility point
        """
        # Get current Pareto front
        pareto_front = self.pareto_state['pareto_front']
        if not pareto_front:
            return self.pareto_state['desired_utility_point']

        # Find Nash point
        if (t * self.deadline) % self.pareto_update_frequency == 0:
            self.set_bid_space()

        nash_bid = self.get_nash_bid()
        if not nash_bid:
            return self.pareto_state['desired_utility_point']

        # Calculate sigmoid weight (0 = use initial, 1 = use Nash)
        weight = self._sigmoid_weight(t, steepness=10.0, midpoint=0.5)

        # Weighted average between initial desired and Nash
        adaptive_desired = {
            
            'our_utility': (1 - weight) * self.initial_desired_point['our_utility'] +  weight * nash_bid.utility_a,

            'opponent_utility': (1 - weight) * self.initial_desired_point['opponent_utility'] + weight * nash_bid.utility_b
        
        }

        return adaptive_desired

    # ============================================================================
    # High-level: Strategy Execution
    # ============================================================================

    def _execute_pareto_walker_strategy(self, t: float) -> Bid:
        """
        Execute Pareto Walker strategy with adaptive desired point and Nash-sorted ball selection.
        """

        opponent_preference = self._get_opponent_preference()

        # Update adaptive desired point based on time (transitions to Nash)
        adaptive_desired_point = self._update_adaptive_desired_point(t)
        self.pareto_state['desired_utility_point'] = adaptive_desired_point

        # Update Pareto state (front and walking)
        self.pareto_state = ParetoWalker.update_pareto_state(
            self.preference, t, opponent_preference, self.pareto_state,
            adaptive_desired_point
        )

        # Get next bid from Nash-sorted ball with state management
        selected_bid, self.pareto_state = ParetoWalker.get_bid_from_sorted_ball(
            self.preference,
            opponent_preference,
            self.pareto_state
        )

        return selected_bid

    def act(self, t: float) -> Action:
        """Main action: Use Pareto Walker strategy."""
        selected_offer = self._execute_pareto_walker_strategy(t)

        # AC Next
        if self.can_accept():
                if self.preference.get_utility(self.last_received_bids[-1]) > self.preference.get_utility(selected_offer):
                    return self.accept_action

        return nenv.Offer(selected_offer)

