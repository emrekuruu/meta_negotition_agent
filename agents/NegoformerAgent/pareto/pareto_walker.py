from typing import Dict, List, Union
import nenv
from .utils import extract_pareto_indices

class ParetoWalker:
    """
    Complete Pareto front management and ball selection.

    Handles:
    - Pareto front calculation and caching
    - Time-based Pareto walking
    - Nash-sorted ball selection with dual constraint expansion
    """
    
    @staticmethod
    def update_pareto_state(
                          preference: nenv.Preference,
                          t: float,
                          opponent_preference: nenv.Preference,
                          pareto_state: Dict,
                          desired_utility_point: Dict[str, float]) -> Dict:
        """
        Update Pareto front and walking state.

        Args:
            preference: Our preference
            t: Current time
            opponent_preference: Opponent's preference
            pareto_state: Dict containing all Pareto state variables
            desired_utility_point: Dict with 'our_utility' and 'opponent_utility' keys

        Returns:
            Updated pareto_state dict
        """
        # Extract state
        pareto_front = pareto_state['pareto_front']
        last_pareto_update = pareto_state['last_pareto_update']
        pareto_index = pareto_state['pareto_index']
        main_strategy_starting_t = pareto_state['main_strategy_starting_t']
        target_time = pareto_state['target_time']
        update_frequency = pareto_state['update_frequency']
        allow_pareto_refresh = pareto_state.get('allow_pareto_refresh', True)

        # Update Pareto front cache
        updated_front, updated_last_update = ParetoWalker._update_pareto_front_cache(
            preference,
            opponent_preference,
            pareto_front,
            last_pareto_update,
            update_frequency,
            allow_pareto_refresh
        )

        # Update Pareto walking state with desired point
        updated_index, updated_point, updated_starting_t = ParetoWalker._update_pareto_walking_state(
            t, updated_front, pareto_index, main_strategy_starting_t, target_time, desired_utility_point
        )

        # Find desired bid (closest to desired utility point)
        desired_bid = None
        desired_index = None
        if updated_front:
            desired_index = ParetoWalker._find_desired_index(desired_utility_point, updated_front)
            desired_bid = updated_front[desired_index]

        # Return updated state
        updated_state = {
            'pareto_front': updated_front,
            'last_pareto_update': updated_last_update,
            'pareto_index': updated_index,
            'current_pareto_point': updated_point,
            'main_strategy_starting_t': updated_starting_t,
            'target_time': target_time,
            'update_frequency': update_frequency,
            'allow_pareto_refresh': allow_pareto_refresh,
            'ball_index': pareto_state.get('ball_index', 0),
            'last_ball_point': pareto_state.get('last_ball_point', None),
            'desired_bid': desired_bid,
            'desired_utility_point': desired_utility_point,
            'desired_index': desired_index
        }

        return updated_state

    @staticmethod
    def _find_desired_index(desired_utility_point: Dict[str, float],
                           pareto_front: List[nenv.BidPoint]) -> int:
        """
        Find index of Pareto point closest to desired utility point.

        Args:
            desired_utility_point: Dict with 'our_utility' and 'opponent_utility' keys
            pareto_front: List of BidPoint objects on Pareto front

        Returns:
            Index of closest Pareto point using Euclidean distance
        """
        if not pareto_front:
            return 0

        # Create a BidPoint for the desired utility point (no bid, just utilities)
        desired_point = nenv.BidPoint(
            None,
            desired_utility_point['our_utility'],
            desired_utility_point['opponent_utility']
        )

        # Find closest point using Euclidean distance
        min_distance = float('inf')
        closest_index = 0

        for i, pareto_point in enumerate(pareto_front):
            distance = pareto_point.distance(desired_point)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    @staticmethod
    def _update_pareto_front_cache(preference: nenv.Preference,
                                  opponent_preference: nenv.Preference,
                                  cached_pareto_front,
                                  last_pareto_update: int,
                                  update_frequency: int,
                                  allow_pareto_refresh: bool = True) -> tuple:
        if cached_pareto_front is None:
            window_size = ParetoWalker._calculate_window_size(preference)
            minimum_utility = max(preference.reservation_value, 0.5)
            available_bids = preference.get_bids_at_range(minimum_utility)

            pareto_front = ParetoWalker._calculate_pareto_front(
                opponent_preference, available_bids, minimum_utility, window_size
            )
            return pareto_front, 0

        if not allow_pareto_refresh:
            return cached_pareto_front, last_pareto_update

        if last_pareto_update >= update_frequency:
            window_size = ParetoWalker._calculate_window_size(preference)
            minimum_utility = max(preference.reservation_value, 0.5)
            available_bids = preference.get_bids_at_range(minimum_utility)

            pareto_front = ParetoWalker._calculate_pareto_front(
                opponent_preference, available_bids, minimum_utility, window_size
            )
            return pareto_front, 0

        return cached_pareto_front, last_pareto_update + 1
    
    @staticmethod
    def _update_pareto_walking_state(t: float, pareto_front: List[nenv.BidPoint],
                                   pareto_index: int,
                                   main_strategy_starting_t: float,
                                   target_time: float,
                                   desired_utility_point: Dict[str, float]) -> tuple:
        if pareto_index == -1:
            new_index = 0
            new_point = pareto_front[0] if pareto_front else None
            new_starting_t = t
        else:
            new_point, new_index = ParetoWalker._get_pareto_point(
                t, pareto_front, main_strategy_starting_t, target_time, desired_utility_point
            )
            new_starting_t = main_strategy_starting_t

        return new_index, new_point, new_starting_t
    
    @staticmethod
    def _get_pareto_point(t: float, pareto_front: List[nenv.BidPoint],
                         main_strategy_starting_t: float,
                         target_time: float,
                         desired_utility_point: Dict[str, float]) -> tuple:
        """Returns (new_pareto_point, new_pareto_index)"""
        # Time-based pareto walking toward desired point
        desired_index = ParetoWalker._find_desired_index(desired_utility_point, pareto_front)

        # Calculate progress ratio
        # Handle edge case where target_time equals starting_t (immediate completion)
        denominator = target_time - main_strategy_starting_t
        if denominator <= 1e-9:  # Treat ~0 or negative duration as "done"
            time_progress = 1.0
        else:
            time_progress = (t - main_strategy_starting_t) / denominator
            
        new_pareto_index = round(time_progress * desired_index)

        if new_pareto_index > desired_index:
            print("Pareto walker cannot go beyond desired point: ", t)

        new_pareto_index = min(new_pareto_index, desired_index)

        return pareto_front[new_pareto_index], new_pareto_index
    
    
    @staticmethod
    def _calculate_window_size(preference: nenv.Preference) -> float:
        """Calculate window size based on domain size."""
        domain_size = len(preference.bids)
        if domain_size < 450:
            return 0.050
        elif domain_size < 1500:
            return 0.045
        elif domain_size < 4500:
            return 0.040
        elif domain_size < 18000:
            return 0.035
        elif domain_size < 33000:
            return 0.030
        else:
            return 0.025
    
    @staticmethod
    def _calculate_pareto_front(opponent_preference: nenv.Preference,
                              available_bids: List[nenv.Bid],
                              minimum_utility: float,
                              window_size: float) -> List[nenv.BidPoint]:
        """
        Calculate Pareto front.
        """
        # Compute fresh Pareto front
        pareto_front = []

        # Extract Pareto indices using utility pairs
        utility_pairs = [
            (bid.utility, opponent_preference.get_utility(bid))
            for bid in available_bids
        ]
        pareto_indices = extract_pareto_indices(utility_pairs, minimum_utility + window_size * 2)

        # Create BidPoint objects for Pareto front
        for i in pareto_indices:
            bid = available_bids[i]
            pareto_front.append(
                nenv.BidPoint(bid, bid.utility, opponent_preference.get_utility(bid))
            )

        return pareto_front
    
    
    @staticmethod
    def _get_pareto_ball(pareto_point: nenv.BidPoint,
                        opponent_preference: nenv.Preference,
                        preference: nenv.Preference,
                        window_size: float,
                        minimum_number_of_bids: int = 0) -> List[nenv.BidPoint]:
        """Generate bid pool (Pareto ball) around the given Pareto point."""
        center_utility_agent = pareto_point.utility_a - window_size
        center_utility_opp = pareto_point.utility_b
        
        # Center point below the Pareto point
        center_point = nenv.BidPoint(None, center_utility_agent, center_utility_opp)
        
        # Get candidate bids in utility range
        minimum_utility = max(preference.reservation_value, 0.5)
        bids = preference.get_bids_at_range(
            max(minimum_utility, center_utility_agent - window_size), 
            center_utility_agent + window_size
        )
        
        pool = []
        
        # Filter bids within window distance
        for bid in bids:
            bid_point = nenv.BidPoint(bid, bid.utility, opponent_preference.get_utility(bid))
            if bid_point - center_point <= window_size:
                pool.append(bid_point)
        
        # Ensure minimum number of bids in pool (if required)
        if minimum_number_of_bids > 0 and len(pool) < minimum_number_of_bids:
            additional_bids = preference.get_bids_at(center_utility_agent, window_size, 1.0)
            additional_bids.sort(
                key=lambda b: nenv.BidPoint(b, b.utility, opponent_preference.get_utility(b)) - pareto_point
            )
            
            while len(pool) < minimum_number_of_bids and len(additional_bids) > 0:
                bid = additional_bids.pop(0)
                if bid != pareto_point.bid:
                    pool.append(nenv.BidPoint(bid, bid.utility, opponent_preference.get_utility(bid)))
        
        # Always include the Pareto point itself
        pool.append(pareto_point)
        
        return pool
    
    @staticmethod
    def _expand_ball_with_limit(
        current_pareto_point: nenv.BidPoint,
        pareto_front: List[nenv.BidPoint],
        pareto_index: int,
        opponent_preference: nenv.Preference,
        preference: nenv.Preference,
        window_size: float,
        current_ball: List[nenv.BidPoint]
    ) -> List[nenv.BidPoint]:
        """
        Expand ball with quadruple symmetric constraints to prevent engulfing adjacent balls.

        Constraints (symmetric):
        1. Cannot exceed best opponent utility in NEXT ball (upper opponent utility bound)
        2. Cannot exceed best our utility in PREVIOUS ball (upper our utility bound)
        3. Cannot be worse for opponent than PREVIOUS pareto point (lower opponent utility bound - monotonicity)
        4. Cannot be worse for us than NEXT pareto point (lower our utility bound - monotonicity)

        Args:
            current_pareto_point: Current Pareto point
            pareto_front: Full Pareto front
            pareto_index: Current index on Pareto front
            opponent_preference: Opponent's preference
            preference: Our preference
            window_size: Window size for ball
            current_ball: Existing ball to expand

        Returns:
            Expanded ball (current_ball + new bids)
        """
        # Step 1: Find opponent utility upper limit (from next ball)
        max_opponent_utility_limit = 1.0  # Default: no limit

        if pareto_index < len(pareto_front) - 1:
            next_pareto_point = pareto_front[pareto_index + 1]

            # Generate next ball to find opponent utility limit
            next_ball = ParetoWalker._get_pareto_ball(
                next_pareto_point,
                opponent_preference,
                preference,
                window_size,
                minimum_number_of_bids=0
            )

            # Find max opponent utility in next ball
            if len(next_ball) == 0:
                max_opponent_utility_limit = next_pareto_point.utility_b
            else:
                max_opponent_utility_limit = max(bp.utility_b for bp in next_ball)

        # Step 2: Find our utility upper limit (from previous ball)
        max_our_utility_limit = 1.0  # Default: no limit

        if pareto_index > 0:
            prev_pareto_point = pareto_front[pareto_index - 1]

            # Generate previous ball to find our utility limit
            prev_ball = ParetoWalker._get_pareto_ball(
                prev_pareto_point,
                opponent_preference,
                preference,
                window_size,
                minimum_number_of_bids=0
            )

            # Find max our utility in previous ball
            if len(prev_ball) == 0:
                max_our_utility_limit = prev_pareto_point.utility_a
            else:
                max_our_utility_limit = max(bp.utility_a for bp in prev_ball)

        # Step 3: Find opponent utility lower limit (from previous pareto point - monotonicity)
        min_opponent_utility = 0.0  # Default: no constraint

        if pareto_index > 0:
            prev_pareto_point = pareto_front[pareto_index - 1]
            min_opponent_utility = prev_pareto_point.utility_b

        # Step 4: Find our utility lower limit (from next pareto point - monotonicity)
        min_our_utility = 0.0  # Default: no constraint

        if pareto_index < len(pareto_front) - 1:
            next_pareto_point = pareto_front[pareto_index + 1]
            min_our_utility = next_pareto_point.utility_a

        # Step 5: Get expansion candidates (broader search)
        center_utility_agent = current_pareto_point.utility_a - window_size
        expansion_candidates = preference.get_bids_at(center_utility_agent, window_size, 1.0)

        # Step 6: Filter by constraints
        expanded_ball = list(current_ball)  # Copy existing ball
        existing_bids = {bp.bid for bp in current_ball}  # Avoid duplicates

        for bid in expansion_candidates:
            if bid in existing_bids:
                continue

            our_utility = bid.utility
            opponent_utility = opponent_preference.get_utility(bid)

            # Quadruple symmetric constraints:
            # 1. Must be worse for opponent than next ball's best (upper opponent bound)
            # 2. Must be worse for us than previous ball's best (upper our bound)
            # 3. Must be at least as good for opponent as previous pareto point (lower opponent bound - monotonicity)
            # 4. Must be at least as good for us as next pareto point (lower our bound - monotonicity)
            if (opponent_utility < max_opponent_utility_limit and
                our_utility < max_our_utility_limit and
                opponent_utility >= min_opponent_utility and
                our_utility >= min_our_utility):
                bid_point = nenv.BidPoint(bid, our_utility, opponent_utility)
                expanded_ball.append(bid_point)

        return expanded_ball

    @staticmethod
    def get_bid_from_sorted_ball(
        preference: nenv.Preference,
        opponent_preference: nenv.Preference,
        pareto_state: Dict
    ) -> tuple:
        """
        Get next bid from Nash-sorted Pareto ball with dynamic expansion.

        - Sorted by Nash product (utility_a * utility_b) descending
        - Tracks position via ball_index
        - Resets ball_index when pareto_point changes
        - Expands ball when exhausted (dual constraints from adjacent balls)

        Args:
            preference: Our preference
            opponent_preference: Opponent's preference
            pareto_state: State dict with ball_index, last_ball_point, pareto_index, pareto_front

        Returns:
            (selected_bid, updated_pareto_state)
        """
        # Extract state
        current_pareto_point = pareto_state['current_pareto_point']
        pareto_index = pareto_state['pareto_index']
        pareto_front = pareto_state['pareto_front']
        ball_index = pareto_state.get('ball_index', 0)
        last_ball_point = pareto_state.get('last_ball_point', None)

        # Get initial ball
        window_size = ParetoWalker._calculate_window_size(preference)
        ball = ParetoWalker._get_pareto_ball(
            current_pareto_point,
            opponent_preference,
            preference,
            window_size,
            minimum_number_of_bids=0
        )

        # Sort by x product (decreasing)
        ball_sorted = sorted(
            ball,
            key=lambda bp: bp.utility_a * bp.utility_b,
            reverse=True
        )

        # Detect situations
        pareto_point_changed = (last_ball_point != current_pareto_point)
        ball_exhausted_same_point = (
            ball_index >= len(ball_sorted) and
            last_ball_point == current_pareto_point and
            ball_index > 0
        )

        # Handle different cases
        if pareto_point_changed:
            ball_index = 0  # New point, start fresh
        elif ball_exhausted_same_point:
            # Expand ball with limit
            ball = ParetoWalker._expand_ball_with_limit(
                current_pareto_point,
                pareto_front,
                pareto_index,
                opponent_preference,
                preference,
                window_size,
                ball
            )
            # Re-sort with expanded ball
            ball_sorted = sorted(
                ball,
                key=lambda bp: bp.utility_a * bp.utility_b,
                reverse=True
            )
            # Continue from where we left off (don't reset ball_index)

        # Handle wrap-around if still exceeds (after potential expansion)
        if ball_index >= len(ball_sorted):
            ball_index = ball_index % len(ball_sorted)

        # Select bid at current index
        selected_bid_point = ball_sorted[ball_index]

        # Update state
        updated_state = pareto_state.copy()
        updated_state['ball_index'] = ball_index + 1  # Increment (will wrap next iteration if needed)
        updated_state['last_ball_point'] = current_pareto_point

        return selected_bid_point.bid, updated_state

    # ============================================================================
    # Logging Methods
    # ============================================================================

    @staticmethod
    def get_all_pareto_balls(
        preference: nenv.Preference,
        opponent_preference: nenv.Preference,
        pareto_state: Dict
    ) -> List[List[nenv.BidPoint]]:
        """
        Generate all Pareto balls for the entire Pareto front.

        This method is primarily for logging/visualization purposes.
        It generates a ball for each point on the Pareto front.

        IMPORTANT: The BidPoints returned contain:
        - bid: The actual bid
        - utility_a: Agent's own utility (accurate)
        - utility_b: ESTIMATED opponent utility (from agent's opponent model)

        The utility_b values represent what the agent THINKS the opponent's utilities are,
        not the actual opponent utilities. This is useful for calculating precision/recall
        of the agent's opponent modeling.

        Args:
            preference: Agent's preference (own, accurate)
            opponent_preference: Agent's MODEL of opponent preference (estimated, may be inaccurate)
            pareto_state: Current Pareto state containing the front

        Returns:
            List of balls, where each ball is a List[BidPoint]
            Index i corresponds to the ball around pareto_front[i]
            Each BidPoint contains: (bid, self_utility, estimated_opponent_utility)
        """
        pareto_front = pareto_state.get('pareto_front', [])
        if pareto_front is None:
            return []
        if not pareto_front:
            return []

        window_size = ParetoWalker._calculate_window_size(preference)

        all_balls = []
        for pareto_point in pareto_front:
            ball = ParetoWalker._get_pareto_ball(
                pareto_point,
                opponent_preference,
                preference,
                window_size,
                minimum_number_of_bids=5
            )
            all_balls.append(ball)

        return all_balls
