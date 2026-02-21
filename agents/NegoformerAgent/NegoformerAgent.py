import os
import nenv
import numpy as np
from typing import Union, List, Tuple, Dict
from nenv import Action, Bid

from .pareto import ParetoWalker
from .forecasting.forecaster import Forecaster
from .velocity_control import ACITracker
from .tracking import NegoformerTrackerMixin

class NegoformerAgent(NegoformerTrackerMixin, nenv.AbstractAgent):

    def __init__(self, 
                 preference: nenv.Preference, session_time: int, estimators=None, 
                 mode: str = "oracle",
                 estimated_preference: nenv.Preference = None, **kwargs):
        super().__init__(preference, session_time, estimators or [])

        self._setup_strategy_constants()
        self._setup_base_configuration(mode, session_time, estimated_preference)
        self._setup_pareto_components()
        self._setup_forecasting_components()
        self._setup_tracking_and_plotting()
    
    @property
    def name(self) -> str:
        return "NegoformerAgent"

    def initiate(self, opponent_name: Union[None, str]):
        self.pareto_state = self._initialize_pareto_state()
        self.opponent_name = opponent_name 
        self.plot_dir = self._determine_plot_dir(self.opponent_name)
        print(self.opponent_name)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.plot_prediction_schedule()

    # ============================================================================
    # Setup Helpers
    # ============================================================================

    def _setup_strategy_constants(self):
        """
        Central place for strategy constants.

        These are kept explicit so readers can quickly understand timing/shape
        assumptions without scanning formulas.
        """
        # Start forecasting after this fraction of the session has elapsed.
        self.forecast_ready_fraction = 0.10

        # Refresh Pareto/Nash estimate every deadline/80 rounds in non-oracle mode.
        # Oracle mode disables refresh explicitly (see `enable_pareto_refresh`).
        self.pareto_update_divisor = 80

        # Initial desired-point bias around Nash.
        self.initial_utility_bias = 0.1

        # Forecast geometry defaults derived from deadline.
        self.default_max_context_divisor = 2
        self.aci_buffer_divisor = 200

    def _setup_base_configuration(
        self,
        mode: str,
        session_time: int,
        estimated_preference: nenv.Preference,
    ):
        self.mode = mode
        self.deadline = session_time
        self.estimated_preference = estimated_preference

        if self.mode != "oracle":
            self.opponent_model = nenv.OpponentModel.BayesianOpponentModel(
                self.preference,
            )

        # Forecasting starts after this many received offers.
        self.forecasting_ready_threshold = int(self.deadline * self.forecast_ready_fraction)

    def _setup_pareto_components(self):
        self.pareto_state = None
        self.enable_pareto_refresh = self.mode != "oracle"
        self.pareto_update_frequency = self.deadline // self.pareto_update_divisor
        self.bid_space = None

        self.set_bid_space()

        # Bias: +0.1 for us, -0.1 for opponent (will diminish over time via sigmoid)
        nash_bid = self.get_nash_bid()
        self.initial_desired_point = {
            'our_utility': nash_bid.utility_a + self.initial_utility_bias,
            'opponent_utility': nash_bid.utility_b - self.initial_utility_bias
        }

    def _setup_forecasting_components(self):
        # Forecasting Tracking
        self.forecast_data: List[Tuple[float, float]] = []
        self.historical_bids: List[Tuple[Bid, float]] = []

        # Forecasting Hyperparameters
        self.horizon = int(os.getenv("HORIZON"))
        self.max_predictions = max(1, int(os.getenv("MAX_PREDICTIONS", 100)))
        self.aci_buffer = max(1, self.deadline // self.aci_buffer_divisor)
        self.forecasting_rounds = self._generate_forecasting_rounds(base_frequency=self.aci_buffer)
        self.forecast_frequency = self._configure_forecast_schedule(base_frequency=self.aci_buffer)
        self.forecaster = Forecaster(
            max_context=self.deadline // self.default_max_context_divisor,
            prediction_length=self.horizon + self.aci_buffer,
            deadline=self.deadline
        )

        # Forecasting State
        self.projected_to_cross = False
        self._forecast_history: List[Tuple[np.ndarray, int]] = []

        # ACI tracker
        self.aci_tracker = ACITracker(lookback=self.horizon)

    def _setup_tracking_and_plotting(self):
        self.plot_dir = None  # Set in initiate() once opponent name is known

        self.tracking = {
            'time': [],
            'target_time': [],
            'opponent_utility': [],
            'nash_utility': [],
            'our_utility': [],
        }

    # ============================================================================
    # Low-level Helpers
    # ============================================================================

    def _get_forecast_schedule_bounds(self, base_frequency: int) -> Tuple[int, int, int]:
        """Return schedule bounds and target prediction count.

        Target count is bounded by:
        - available active rounds,
        - `max_predictions`,
        - minimum spacing implied by `base_frequency`.
        """
        start = max(0, self.forecasting_ready_threshold)
        end = max(start, self.deadline)
        active_rounds = max(1, end - start + 1)
        max_by_base_frequency = 1 + ((active_rounds - 1) // max(1, base_frequency))
        target_predictions = max(1, min(self.max_predictions, active_rounds, max_by_base_frequency))

        return start, end, target_predictions

    def _configure_forecast_schedule(self, base_frequency: int) -> int:
        """
        Configure representative forecast frequency for reporting/debugging.

        Scheduling itself is handled by `_generate_forecasting_rounds`.
        """
        start, end, target_predictions = self._get_forecast_schedule_bounds(base_frequency)
        if target_predictions <= 1:
            return max(1, end - start + 1)

        span = end - start
        return max(1, int(np.ceil(span / (target_predictions - 1))))

    def _generate_forecasting_rounds(self, base_frequency: int) -> List[int]:
        """Generate list of rounds where forecasting should occur.

        Generates evenly spread integer rounds with exact target count whenever
        enough active rounds are available.
        """
        start, end, target_predictions = self._get_forecast_schedule_bounds(base_frequency)
        if target_predictions <= 1:
            return [start]

        span = end - start
        return [
            start + ((i * span) // (target_predictions - 1))
            for i in range(target_predictions)
        ]

    def _round_from_time(self, t: float) -> int:
        """Convert normalized time into a stable integer round index."""
        return max(0, min(self.deadline, int(round(t * self.deadline))))

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
            'allow_pareto_refresh': self.enable_pareto_refresh,
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
        return self.opponent_model.preference

    def set_bid_space(self):
        opponent_pref = self._get_opponent_preference()
        self.bid_space = nenv.BidSpace(self.preference, opponent_pref)
    
    def get_nash_bid(self):
        return self.bid_space.nash_point

    def _sigmoid_weight(self, t: float) -> float:
        """
        Calculate sigmoid weight for transitioning from initial to Nash.

        Args:
            t: Current normalized time (0 to 1)

        Returns:
            Weight from 0 to 1 (0 = use initial desired, 1 = use Nash)
        """
        return 1 / (1 + np.exp(-10 * (t - 0.5)))

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

        Uses predicted Nash with a diminishing bias:
        - At t=0: our_utility = Nash + 0.1, opponent_utility = Nash - 0.1
        - At t=1: our_utility = Nash, opponent_utility = Nash (pure Nash)

        The bias diminishes according to a sigmoid function.

        Args:
            t: Current normalized time (0 to 1)

        Returns:
            Updated desired utility point
        """
        # First iteration: pareto_front doesn't exist yet, use initial_desired_point
        # (This is initialization, not a fallback - initial_desired_point is correct for t≈0)
        pareto_front = self.pareto_state['pareto_front']
        if not pareto_front:
            return self.initial_desired_point.copy()

        # In oracle mode, keep desired point fixed (no time-adaptive updates).
        if self.mode == "oracle":
            return self.initial_desired_point.copy()

        # Update bid space periodically for better Nash estimate
        if self._round_from_time(t) % self.pareto_update_frequency == 0:
            self.set_bid_space()

        nash_bid = self.get_nash_bid()
        if not nash_bid:
            raise ValueError("Nash bid is None - cannot compute adaptive desired point")

        # Calculate sigmoid weight (0 at t=0, 1 at t=1)
        weight = self._sigmoid_weight(t)

        # Diminishing bias: full (0.1) at start, zero at end
        bias = self.initial_utility_bias * (1 - weight)

        adaptive_desired = {
            'our_utility': nash_bid.utility_a + bias,
            'opponent_utility': nash_bid.utility_b - bias
        }
        return adaptive_desired

    # ============================================================================
    # Mid-level: Data & Forecasting
    # ============================================================================

    def _get_forecast_data(self) -> List[Tuple[float, float]]:
        """Recalculate forecast data with current opponent model."""
        opponent_preference = self._get_opponent_preference()
        return [(opponent_preference.get_utility(bid), t) for bid, t in self.historical_bids]

    def get_opponent_forecast(self) -> np.ndarray:
        """
        Get forecasted opponent utility median trajectory.

        Returns:
            1D numpy array, shape: (prediction_length,)
        """
        self.forecast_data = self._get_forecast_data()
        return self.forecaster(self.forecast_data)

    def _refresh_forecast_if_due(self, t: float):
        """Generate/refresh forecast only on configured rounds."""
        current_round = self._round_from_time(t)
        should_refresh_forecast = current_round in self.forecasting_rounds
        if not should_refresh_forecast:
            return

        forecast = self.get_opponent_forecast()
        median = forecast
        if len(median) == 0:
            return

        self._forecast_history.append((forecast, current_round))
        # Prune forecasts whose buffer zone has fully expired
        self._forecast_history = [
            (f, r) for f, r in self._forecast_history
            if current_round < r + self.horizon + self.aci_buffer
        ]

    # ============================================================================
    # High-level: Strategy Execution
    # ============================================================================

    def receive_offer(self, bid: Bid, t: float):
        """Process received offer."""
        # Update opponent model
        if self.mode != "oracle":
            self.opponent_model.update(bid, t)

        # ACI step: evaluate stored forecasts within a buffer zone centered on the horizon.
        # For each forecast, buffer zone is indices [horizon - aci_buffer, horizon + aci_buffer).
        # Forecasts are made on an evenly distributed schedule
        # (approximately every `forecast_frequency` rounds).
        current_opp_utility = self._get_opponent_preference().get_utility(bid)
        current_round = self._round_from_time(t)

        for forecast, forecast_round in self._forecast_history:
            median = forecast
            idx = current_round - forecast_round
            if self.horizon - self.aci_buffer <= idx < self.horizon + self.aci_buffer:
                did_issue = self.aci_tracker.issue(
                    float(median[idx]),
                    current_round,
                )
                if did_issue:
                    self.aci_tracker.observe(current_opp_utility, current_round)
                break

        # Track metrics for visualization
        self._track_metrics(t, bid, current_opp_utility)

        # Store bid for recalculation
        self.historical_bids.append((bid, t))

    def _execute_pareto_walker_strategy(self, t: float) -> Bid:
        """
        Execute full Negoformer strategy (forecasting + ACI + Pareto selection).
        """

        self._refresh_forecast_if_due(t)

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
            last_received_utility = self.preference.get_utility(self.last_received_bids[-1])
            selected_offer_utility = self.preference.get_utility(selected_offer)
            if last_received_utility > selected_offer_utility:
                return self.accept_action

        return nenv.Offer(selected_offer)


    # ============================================================================
    # Visualization & Debugging
    # ============================================================================

    def terminate(self, is_accept: bool, opponent_name: str, t: float):
        """Called when negotiation ends. Generate summary plots and debug outputs."""
        self.plot_prediction_schedule()
        self.plot_negotiation_summary(opponent_name)
        self.plot_observation_intervals(opponent_name)
        self.plot_interval_alphas(opponent_name)
        self._save_aci_csv(opponent_name)
