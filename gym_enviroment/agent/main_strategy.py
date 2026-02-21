from typing import List, Dict, Union, Tuple
import numpy as np
import nenv
from nenv import Action, Bid
from gym_enviroment.config.config import config
from typing import Any

# Import all modules
from .forecasting import Forecaster
from .observation import ObservationBuilder
from .time import TimeEstimator
from .common.offer_point import OfferPoint

# Import predefined strategies for strategy selection
from agents.boulware.Boulware import BoulwareAgent
from agents.conceder.Conceder import ConcederAgent
from agents.SAGA.SAGAAgent import SAGAAgent
from agents.HybridAgent.HybridAgent import HybridAgent
from agents.MICRO.MICRO import MICROAgent
from agents.Kawaii.Kawaii import Kawaii


class MainStrategy(nenv.AbstractAgent):
    """
    RL-guided multivariate Nash-based bidding strategy.
    
    Creates bidding curves using a multivariate linear equation with tanh:
    utility = tanh(w1*nash1 + w2*nash2 + ... + w96*nash96 + w97*time)
    
    Features:
    - Nash utilities: our_utility * opponent_utility from 96 history points
    - Time feature: current negotiation progress (0.0-1.0)
    - RL-learned weights: supports positive and negative coefficients
    - tanh activation: maps linear combination to [-1,1] range
    """
    
    def __init__(self, preference: nenv.Preference, estimated_preference: nenv.Preference, session_time: int, mode = "oracle",  params: Dict = None):
        super().__init__(preference, session_time, [])
        
        # Configuration
        params = params or {}
        self.mode = mode
        
        # Initialize modules
        self.forecaster = Forecaster()
        self.time_estimator = TimeEstimator()
        
        # State management
        self.estimated_preference = estimated_preference if mode == "oracle" else None
        self.opponent_model = estimated_preference if mode == "oracle" else None
        self.offer_history = []  # List of OfferPoint objects for forecasting
        
        # Strategy selection setup
        self.available_strategies = self._initialize_strategies()
        self.initial_strategy = self.available_strategies['micro']
        
        # Current negotiation state
        self.current_forecasts = []
        self.current_candidate_list = []
        self.candidate_names = []
        self.current_candidates = {}
        
        # RL integration - multivariate linear equation
        # action vector: [strategy_index(float), w1, w2, ..., wN, wt]
        self.rl_action: Union[int, np.ndarray] = 0
        
        # Multivariate equation parameters
        self.max_history_points = int(config.core['max_history_points'])  # 96
        
        # New dual utility targeting: separate our_utility and opponent_utility features
        # Each equation has: our_utility_features (96) + opponent_utility_features (96) + time (1) = 193
        self.features_per_equation = 2 * self.max_history_points + 1  # 193
        self.total_features = 2 * self.features_per_equation  # 386 for both equations
        # Total action space: 1 (strategy) + 386 (dual equations) = 387

        # Forecasts for reward shaping and RL features
        self.baseline_forecast: Union[None, np.ndarray] = None  # Opponent utilities for simulation
        self.nash_baseline_forecast: Union[None, np.ndarray] = None  # Nash utilities for RL features
        self.updated_forecast: Union[None, np.ndarray] = None
            
    def _initialize_strategies(self) -> Dict[str, nenv.AbstractAgent]:
        """Initialize available predefined strategies for selection."""
        strategies = {}
        
        # Create strategy instances
        strategies['saga'] = SAGAAgent(self.preference, self.session_time, [])
        strategies['hybrid'] = HybridAgent(self.preference, self.session_time, [])
        strategies['conceder'] = ConcederAgent(self.preference, self.session_time, [])
        strategies['boulware'] = BoulwareAgent(self.preference, self.session_time, [])
        strategies['micro'] = MICROAgent(self.preference, self.session_time, [])
        strategies['kawaii'] = Kawaii(self.preference, self.session_time, [])
        
        return strategies
    
    @property
    def name(self) -> str:
        return "MainStrategy"
    
    def initiate(self, opponent_name: Union[None, str]):
        """Initialize for a new negotiation."""
        # Initialize opponent model
        if self.mode != "oracle":
            self.opponent_model = nenv.OpponentModel.ClassicFrequencyOpponentModel(self.preference)
        
        # Initialize initial strategy
        self.initial_strategy.initiate(opponent_name)
        
        # Initialize all available strategies
        for strategy in self.available_strategies.values():
            strategy.initiate(opponent_name)
        
        # Reset state
        self.offer_history = []
        self.current_forecasts = []
        self.current_candidate_list = []
        self.candidate_names = []
        self.current_candidates = {}
        self.rl_action = 0
        self.last_candidate = {}
    
    def receive_offer(self, bid: Bid, t: float):
        """Process received offer."""
        # Update opponent model
        
        if self.mode != "oracle":
            self.opponent_model.update(bid, t)

        self.estimated_preference = self.opponent_model if self.mode == "oracle" else self.opponent_model.preference
        
        # Record in offer history for forecasting
        offer_point = OfferPoint(0, bid, t)  # 0 = opponent offer
        self.offer_history.append(offer_point)
        
        # Update time estimator
        self.time_estimator.update(t)
        
        # Update initial strategy
        self.initial_strategy.receive_offer(bid, t)
        
        # Update all available strategies
        for strategy in self.available_strategies.values():
            strategy.receive_offer(bid, t)
    
    # ---------- Forecasting ----------
    
    def _convert_opponent_forecast_to_nash(self, opponent_forecast: np.ndarray) -> np.ndarray:
        """Convert forecasted opponent utilities to Nash utilities.
        
        For each forecasted opponent utility:
        1. Convert to bid using opponent's preference model
        2. Evaluate that bid for both sides
        3. Compute Nash utility = our_utility * opponent_utility
        
        Args:
            opponent_forecast: Array of forecasted opponent utility values
            
        Returns:
            Array of Nash utility values
        """
        nash_utilities = []
        
        for opp_util in opponent_forecast:
            opp_bid = self.estimated_preference.get_bid_at(opp_util)
            opp_util = self.estimated_preference.get_utility(opp_bid)
            our_utility = self.preference.get_utility(opp_bid)
            nash_utility = our_utility * opp_util
            nash_utilities.append(nash_utility)
        return np.array(nash_utilities)
    

    def _generate_opponent_forecast(self, t: float) -> np.ndarray:
        """Generate forecast for opponent's future offers based on current state."""

        # Get target times for prediction
        target_times = self.time_estimator.populate(t, config.core['forecast_length'])
        
        # Use the forecaster with current offer history (no modifications)
        predictions = self.forecaster.predict_from_history(
            offer_points=self.offer_history,
            estimated_preference=self.estimated_preference,
            target_times=target_times
        )
        
        return predictions
    
    def _save_strategy_state(self, strategy: nenv.AbstractAgent) -> dict:
        """Save strategy state that gets modified during simulation."""
        saved_state = {}
        
        # Common state attributes that get modified
        state_attrs = [
            'lastOffer', 'lastReceivedBid', 'last_received_bids',
            'my_last_bid', 'my_last_bids', 'rounds', 'isFirst',
            'received_bids', 'oppUtility', 'allBids', 'opponents',
            'oppBidHistory', 'firstBidFromOpponent', 'slotHistory',
            'ActionOfOpponent', 'receivedBid', 'totalHistory',
            'lastOpponentBid', 'opponentLastBid', 'bidHistory',
            'opponentHistory', 'opponent_model'
        ]
        
        for attr in state_attrs:
            if hasattr(strategy, attr):
                original_value = getattr(strategy, attr)
                if isinstance(original_value, list):
                    saved_state[attr] = original_value.copy()
                elif isinstance(original_value, dict):
                    saved_state[attr] = original_value.copy()
                elif hasattr(original_value, 'copy'):
                    saved_state[attr] = original_value.copy()
                else:
                    saved_state[attr] = original_value
                    
        return saved_state
    
    def _restore_strategy_state(self, strategy: nenv.AbstractAgent, saved_state: dict) -> None:
        """Restore strategy state after simulation."""
        for attr, value in saved_state.items():
            if hasattr(strategy, attr):
                setattr(strategy, attr, value)

    def _simulate_patch_for_strategy(self, strategy: nenv.AbstractAgent, t: float, opponent_forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate N-step patch for a given strategy without deepcopy using state save/restore."""
        N = config.core['simulation_bids_count']
        time_increment = (1 / config.environment["deadline_round"]) 
        
        # Save strategy state before simulation
        saved_state = self._save_strategy_state(strategy)
        current_time = t

        try:
            simulated_points: List[OfferPoint] = []
            for i in range(N):
                opp_util = opponent_forecast[i] 
                opp_bid = self.estimated_preference.get_bid_at(opp_util)
                strategy.receive_bid(opp_bid, current_time)
                simulated_points.append(OfferPoint(0, opp_bid, current_time))
                act = strategy.act(current_time)
                if hasattr(act, 'bid') and act.bid:
                    simulated_points.append(OfferPoint(1, act.bid, current_time))
                current_time += time_increment
            return simulated_points
        finally:
            # Always restore original state
            self._restore_strategy_state(strategy, saved_state)
    
    
    def _generate_forecast_for_strategy(self, strategy: nenv.AbstractAgent, t: float, opponent_forecast: np.ndarray) -> np.ndarray:
        """Generate forecast by simulating N bids with strategy and opponent forecast."""

        # Get simulation parameters
        N = config.core['simulation_bids_count']
        time_increment = (1 / config.environment["deadline_round"])
        
        # Simulate N rounds using state save/restore instead of deepcopy
        simulated_points = self._simulate_patch_for_strategy(strategy, t, opponent_forecast)

        # Step 4: Generate forecast using last (96-2*N-1) real bids + simulated bids
        required_real_bids = config.core['sequence_length'] - 2 * N - 1
        
        # Take the required number of real bids from the end
        real_bids_subset = self.offer_history[-required_real_bids:]
        
        # Combine real history + simulated offers
        combined_history = real_bids_subset + simulated_points
        
        # Get target times for final prediction
        target_times = self.time_estimator.populate(t + N * time_increment, config.core['forecast_length'])
        
        # Use the forecaster with combined history
        predictions = self.forecaster.predict_from_history(
            offer_points=combined_history,
            estimated_preference=self.estimated_preference,
            target_times=target_times
        )
        
        return predictions
    
    # ---------- Observation State Building ----------

    def build_observation(self) -> Dict[str, np.ndarray]:
        """Build observation for RL agent from OfferPoint history.

        Prepares forecasts (opponent forecast, per-strategy candidate forecasts, baseline)
        so that act() does not recompute them.
        """

        if not self._is_forecasting_ready():
            self.current_forecasts = [np.zeros(config.core['forecast_length']) for _ in range(len(self.available_strategies))]
            self.baseline_forecast = np.zeros(config.core['forecast_length'])
            self.nash_baseline_forecast = np.zeros(config.core['forecast_length'])

        else:
            t_anchor = self.offer_history[-1].t if self.offer_history else 0.0
            opp_fc = self._generate_opponent_forecast(t_anchor)
            self.baseline_forecast = opp_fc  # Keep original for simulation
            
            # Generate Nash forecasts for RL equation features
            self.nash_baseline_forecast = self._convert_opponent_forecast_to_nash(opp_fc)

            # Generate forecasts for each strategy
            forecasts = []
            for name, strategy in self.available_strategies.items():
                forecasts.append(self._generate_forecast_for_strategy(strategy, t_anchor, opp_fc))
            
            self.current_forecasts = forecasts

        return ObservationBuilder.build_observation(
            candidate_forecasts=self.current_forecasts,
            offer_points=self.offer_history,
            estimated_preference=self.estimated_preference
        )
    
    # ---------- RL Integration ----------

    def set_rl_action(self, action: np.ndarray):
        """Set the RL action vector strictly as array-like [strategy_index, c0..cD]."""
        self.rl_action = np.array(action, dtype=float)

    def _parse_action(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Parse RL action vector into (strategy_index, our_utility_weights, opponent_utility_weights)."""
        if not isinstance(self.rl_action, (list, np.ndarray)):
            raise ValueError("rl_action must be array-like: [strategy_index, our_weights..., opp_weights...]")
        action_array = np.array(self.rl_action, dtype=float)

        # Strategy index
        strategy_index_float = float(action_array[0]) if action_array.size > 0 else 0.0
        strategy_index = int(np.clip(np.rint(strategy_index_float), 0, len(self.available_strategies) - 1))
        self.selected_strategy = int(strategy_index)
        self.updated_forecast = self.current_forecasts[strategy_index]

        # Parse dual utility equation weights
        our_weights = np.zeros(self.features_per_equation, dtype=float)
        opp_weights = np.zeros(self.features_per_equation, dtype=float)
        
        if action_array.size >= 1 + self.total_features:
            # Full action vector: [strategy, our_weights(193), opp_weights(193)]
            our_weights = action_array[1:1 + self.features_per_equation]
            opp_weights = action_array[1 + self.features_per_equation:1 + self.total_features]
            
        elif action_array.size > 1:
            # Partial action vector - fill what we can
            remaining = action_array[1:]
            
            # Fill our_weights first
            our_fill_size = min(remaining.size, self.features_per_equation)
            our_weights[:our_fill_size] = remaining[:our_fill_size]
            
            # Fill opponent_weights with remaining
            if remaining.size > self.features_per_equation:
                opp_remaining = remaining[self.features_per_equation:]
                opp_fill_size = min(opp_remaining.size, self.features_per_equation)
                opp_weights[:opp_fill_size] = opp_remaining[:opp_fill_size]

        return strategy_index, our_weights, opp_weights



    # ---------- Action Selection ----------
    
    def act(self, t: float) -> Action:
        """Main action decision logic.

        Requires that build_observation() has been called earlier in the step,
        which primes forecasts and caches opponent forecast.
        """
        
        selected_offer = None

        # Initial strategy (not enough history for forecasting)
        if not self._is_forecasting_ready():
            action = self.initial_strategy.act(t)
            selected_offer = action.bid

        # Main strategy with forecasting (forecasts already prepared in build_observation)
        else:   
            selected_offer = self._execute_main_strategy(t).bid

        # AC_Next Acceptance Strategy
        if self.can_accept() and False:
            return self.accept_action
        
        if (self.preference.get_utility(selected_offer) < (1-t)) and not bool(config.core['training']):
            selected_offer = self.preference.get_bid_at(1-t)

        offer_point = OfferPoint(1, selected_offer, t)  
        self.offer_history.append(offer_point)
        
        self._notify_strategies_of_sent_bid(selected_offer)
        
        return nenv.Offer(selected_offer)
    
    def _is_forecasting_ready(self) -> bool:
        """Check if we have enough history for forecasting."""
        return len(self.offer_history) >= config.core['sequence_length'] 
    
    def _execute_main_strategy(self, t: float) -> Action:
        """Execute the main forecasting-based strategy."""
        bid = self._generate_bid_with_equation_fitting(t)
        self._notify_strategies_of_sent_bid(bid)
        return nenv.Offer(bid)
    
    def _extract_dual_utility_features(self, history_points: List[OfferPoint], t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract separate our_utility and opponent_utility features from negotiation history.
        
        Args:
            history_points: List of OfferPoint objects from history + patches
            t: Current negotiation time
            
        Returns:
            Tuple of (our_features, opponent_features) where each contains:
            [our_util_1, our_util_2, ..., our_util_96, opp_util_1, opp_util_2, ..., opp_util_96, time]
        """
        our_utilities = []
        opponent_utilities = []
        
        # Take the most recent max_history_points from the history
        recent_points = history_points[-self.max_history_points:] if len(history_points) > self.max_history_points else history_points
        
        # Pad with zeros if we don't have enough points
        while len(recent_points) < self.max_history_points:
            our_utilities.append(0.0)  # Zero padding for missing points
            opponent_utilities.append(0.0)
        
        # Extract separate utilities for each recent point
        for point in recent_points:
            our_utility = point.bid.utility
            opponent_utility = self.estimated_preference.get_utility(point.bid)
            our_utilities.append(our_utility)
            opponent_utilities.append(opponent_utility)
        
        # Build feature vectors for each equation: [our_utils(96), opp_utils(96), time(1)]
        our_features = np.array(our_utilities + opponent_utilities + [t], dtype=float)
        opp_features = np.array(our_utilities + opponent_utilities + [t], dtype=float)
        
        return our_features, opp_features
    
    def _create_dual_utility_strategy(self, our_weights: np.ndarray, opp_weights: np.ndarray, history_points: List[OfferPoint]) -> callable:
        """Create a dual utility targeting strategy from RL-learned feature weights.
        
        Args:
            our_weights: Feature weights for our utility equation
            opp_weights: Feature weights for opponent utility equation  
            history_points: Negotiation history + patches for feature extraction
            
        Returns:
            A callable function(t) -> (target_our_utility, target_opponent_utility)
            where t is in negotiation time scale [0,1] (0=start, 1=deadline)
        """
        def dual_utility_strategy(t: float) -> Tuple[float, float]:
            """Dual utility strategy function: takes negotiation time, returns target utility pair.
            
            Args:
                t: Negotiation time in [0,1] scale (0=start, 1=deadline)
                
            Returns:
                Tuple of (target_our_utility, target_opponent_utility)
            """
            # Ensure t is in valid range [0,1]
            t_clipped = float(np.clip(t, 0.0, 1.0))
            
            # Extract dual utility features from history + current time
            our_features, opp_features = self._extract_dual_utility_features(history_points, t_clipped)
            
            # Evaluate both multivariate equations
            our_raw = self._evaluate_multivariate_equation(our_features, our_weights)
            opp_raw = self._evaluate_multivariate_equation(opp_features, opp_weights)
            
            # Use tanh activation for smooth bounded output (standard in robotics)
            # Maps (-∞, +∞) → (-1, 1) with smooth gradients
            target_our_utility = float(np.tanh(our_raw))
            target_opponent_utility = float(np.tanh(opp_raw))
            
            return target_our_utility, target_opponent_utility
        
        return dual_utility_strategy
    
    def _find_closest_bid_to_utility_pair(self, target_our_utility: float, target_opponent_utility: float) -> nenv.Bid:
        """Find the bid that minimizes Euclidean distance to target utility pair.
        
        Args:
            target_our_utility: Target utility for our agent
            target_opponent_utility: Target utility for opponent
            
        Returns:
            The bid closest to the target utility pair in 2D utility space
        """
        best_bid = None
        min_distance = float('inf')
        
        # Search through all possible bids to find the closest one
        # This is computationally expensive but gives exact results
        for bid in self.preference.bids:
            our_utility = bid.utility
            opponent_utility = self.estimated_preference.get_utility(bid)
            
            # Calculate Euclidean distance in 2D utility space
            distance = ((our_utility - target_our_utility) ** 2 + 
                       (opponent_utility - target_opponent_utility) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_bid = bid
        
        return best_bid if best_bid is not None else self.preference.max_util_bid
    
    def _evaluate_multivariate_equation(self, features: np.ndarray, weights: np.ndarray) -> float:
        """Evaluate multivariate linear equation: raw_output = Σ(wi * fi).
        
        Standard robotics approach: Direct linear combination, no constraints.
        The tanh activation is applied later to bound the output to [-1,1].
        
        Args:
            features: Feature vector extracted from negotiation state
            weights: Weight vector learned by RL model
            
        Returns:
            Raw linear combination (unbounded)
        """
        # Ensure matching dimensions
        min_len = min(features.shape[0], weights.shape[0])
        features = features[:min_len] 
        weights = weights[:min_len]
        
        # Direct linear combination - no normalization, no constraints
        # This is the standard approach in robotics papers
        result = float(np.dot(weights, features))
        
        return result

    def _generate_bid_with_equation_fitting(self, t: float) -> nenv.Bid:
        """Generate the final bid using selected strategy and RL-learned Nash coefficients.

        Steps:
        1) Use discrete action to select a predefined strategy and simulate/append N offers as a patch.
        2) Build trajectory scaffold from real history + selected strategy patch.
        3) Use RL-learned Nash coefficients to define bidding strategy over the scaffold.
        4) Evaluate multivariate equation at current time to get next utility directly.
        """

        strategy_names = list(self.available_strategies.keys())
        strategy_index, our_weights, opp_weights = self._parse_action()
        selected_strategy_name = strategy_names[strategy_index]
        selected_strategy = self.available_strategies[selected_strategy_name]

        opponent_forecast = self.baseline_forecast

        selected_strategy_simulated_points = self._simulate_patch_for_strategy(selected_strategy, t, opponent_forecast)
        self.selected_strategy_simulated_points = selected_strategy_simulated_points

        # b) Collect the last sequence_length offers from history + patch to build trajectory scaffold
        sequence_length = config.core['sequence_length']
        
        # Use the last sequence_length items from the combined real+patch history
        combined_full = self.offer_history + selected_strategy_simulated_points
        scaffold = combined_full[-sequence_length:]

        # Validate scaffold has valid data
        if len(scaffold) == 0:
            raise RuntimeError("No valid offer points after building scaffold.")

        # c) Create the dual utility strategy function using multivariate features
        # Combine real history + simulated patches for feature extraction
        combined_history = self.offer_history + selected_strategy_simulated_points
        dual_utility_strategy = self._create_dual_utility_strategy(our_weights, opp_weights, combined_history)
        
        # Store for external access
        self._current_dual_utility_strategy = dual_utility_strategy
        
        # d) Generate target utility pair using the dual utility strategy
        target_our_utility, target_opponent_utility = dual_utility_strategy(t)

        # e) Find the bid closest to the target utility pair
        bid = self._find_closest_bid_to_utility_pair(target_our_utility, target_opponent_utility)

        return bid

    def _notify_strategies_of_sent_bid(self, sent_bid: nenv.Bid):
        """Notify all strategies about the bid that was actually sent to keep their states consistent."""
        for strategy_name, strategy in self.available_strategies.items():
            try:
                # Handle single last bid attribute
                if hasattr(strategy, 'my_last_bid'):
                    strategy.my_last_bid = sent_bid
                
                # Handle list of last bids - update the most recent one
                if hasattr(strategy, 'my_last_bids'):
                    if isinstance(strategy.my_last_bids, list) and len(strategy.my_last_bids) > 0:
                        strategy.my_last_bids[-1] = sent_bid
                    elif isinstance(strategy.my_last_bids, list):
                        # If empty list, append the bid (some strategies might expect this)
                        strategy.my_last_bids.append(sent_bid)
                
                # Handle other common bid tracking attributes
                if hasattr(strategy, 'lastOffer'):
                    strategy.lastOffer = sent_bid
                if hasattr(strategy, 'lastSentBid'):
                    strategy.lastSentBid = sent_bid
                    
            except Exception as e:
                # Failed to notify strategy - continue silently to avoid breaking the flow
                # In production, you might want to log this for debugging
                continue
    
    # ---------- Utils ----------
            
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self.available_strategies.keys())
    
    def get_current_dual_utility_strategy(self) -> callable:
        """Get the current RL dual utility strategy function.
        
        Returns:
            A callable function(t) -> (our_utility, opponent_utility) representing the current dual utility strategy.
            Returns None if called before strategy is created.
        """
        if not hasattr(self, '_current_dual_utility_strategy'):
            return None
        return self._current_dual_utility_strategy
    