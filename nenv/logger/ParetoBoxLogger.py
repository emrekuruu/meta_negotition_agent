from nenv.logger.AbstractLogger import AbstractLogger, Bid, SessionLogs, Session, LogRow
from typing import Union, List, Dict
import math
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle
import numpy as np
import os

class ParetoBoxLogger(AbstractLogger):
    """
    ParetoBoxLogger evaluates opponent model accuracy per Pareto box.
    It also visualizes the Pareto box structure at the beginning of the session.
    """

    def before_session_start(self, session: Union[Session, SessionLogs]) -> List[str]:
        """
        Calculate true Pareto boxes and visualize them before session starts.
        """
        # Import here to avoid circular import
        from agents.ParetoWalkerAgent.pareto import ParetoWalker

        # 1. Generate Boxes for Agent A (A's perspective)
        self.true_boxes_a = self._generate_agent_boxes(
            owner_pref=session.agentA.preference,
            opp_pref=session.agentB.preference,
            ParetoWalker=ParetoWalker
        )

        # 2. Generate Boxes for Agent B (B's perspective)
        self.true_boxes_b = self._generate_agent_boxes(
            owner_pref=session.agentB.preference,
            opp_pref=session.agentA.preference,
            ParetoWalker=ParetoWalker
        )

        # # 3. Visualize Structure
        # self._visualize_pareto_structure(session, self.true_boxes_a, self.true_boxes_b)

        return []

    def _generate_agent_boxes(self, owner_pref, opp_pref, ParetoWalker) -> List[List]:
        """
        Generate sorted, expanded Pareto boxes for an agent.
        """
        minimum_utility = max(owner_pref.reservation_value, 0.5)
        available_bids = owner_pref.get_bids_at_range(minimum_utility)
        window_size = ParetoWalker._calculate_window_size(owner_pref)

        # 1. Calculate Pareto Front (Oracle Mode - uses both prefs)
        pareto_front = ParetoWalker._calculate_pareto_front(
            opponent_preference=opp_pref,
            available_bids=available_bids,
            minimum_utility=minimum_utility,
            window_size=window_size
        )

        # 2. Sort Front by Owner Utility Descending (Strict Monotonicity)
        # This is crucial for the expansion logic to work correctly.
        pareto_front.sort(key=lambda bp: bp.utility_a, reverse=True)

        # 3. Generate and Expand Boxes
        all_boxes = []
        for pareto_index, pareto_point in enumerate(pareto_front):
            # Create initial ball
            box = ParetoWalker._get_pareto_ball(
                pareto_point=pareto_point,
                opponent_preference=opp_pref,
                preference=owner_pref,
                window_size=window_size,
                minimum_number_of_bids=5
            )

            # Expand ball
            expanded_box = ParetoWalker._expand_ball_with_limit(
                current_pareto_point=pareto_point,
                pareto_front=pareto_front,
                pareto_index=pareto_index,
                opponent_preference=opp_pref,
                preference=owner_pref,
                window_size=window_size,
                current_ball=box
            )

            all_boxes.append(expanded_box)

        return all_boxes

    def _visualize_pareto_structure(self, session, boxes_a, boxes_b):
        """
        Visualize the Pareto box structure for both agents.
        Creates a 2x2 grid:
        Row 1: Agent A (Own Utility), Agent A (Opponent Utility)
        Row 2: Agent B (Own Utility), Agent B (Opponent Utility)
        """
        from nenv import BidSpace
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))

        # Get Nash utilities
        bid_space = BidSpace(session.agentA.preference, session.agentB.preference)
        nash_bid = bid_space.nash_point
        nash_util_a = nash_bid.utility_a if nash_bid else 0.78
        nash_util_b = nash_bid.utility_b if nash_bid else 0.78

        # Plot Agent A
        # Own: A, Opponent: B
        self._plot_single_agent_structure(
            ax_own=axs[0, 0], 
            ax_opp=axs[0, 1], 
            boxes=boxes_a, 
            agent_name=session.agentA.name, 
            nash_own=nash_util_a, 
            nash_opp=nash_util_b
        )
        
        # Plot Agent B
        # Own: B, Opponent: A
        self._plot_single_agent_structure(
            ax_own=axs[1, 0], 
            ax_opp=axs[1, 1], 
            boxes=boxes_b, 
            agent_name=session.agentB.name, 
            nash_own=nash_util_b, 
            nash_opp=nash_util_a
        )

        plt.tight_layout()

        # Save with agent names
        output_dir = self.get_path("opponent model/pareto_boxes/")
        os.makedirs(output_dir, exist_ok=True)
        
        agent_a_name = session.agentA.name
        agent_b_name = session.agentB.name
        filename = f"{agent_a_name}_{agent_b_name}_pareto_structure.png"
        
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Pareto structure visualization saved to: {output_path}")

    def _plot_single_agent_structure(self, ax_own, ax_opp, boxes, agent_name, nash_own, nash_opp):
        """
        Plot structure for a single agent, split into Own and Opponent subplots.
        """
        if not boxes:
            return

        # Color map for boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(boxes)))
        num_boxes = len(boxes)

        for box_idx, box in enumerate(boxes):
            if not box:
                continue

            # Time position: box 0 at t=0, last box at t=1
            time_position = box_idx / max(num_boxes - 1, 1)
            
            # Calculate bar width
            bar_width = 1.0 / num_boxes if num_boxes > 1 else 0.1

            # --- Plot Owner Utility (Decreasing) on ax_own ---
            our_utils = [bp.utility_a for bp in box]
            min_util = min(our_utils)
            max_util = max(our_utils)

            # Draw box range for Owner
            rect = Rectangle(
                (time_position - bar_width/2, min_util),
                width=bar_width,
                height=max_util - min_util,
                facecolor='blue',
                alpha=0.3,
                edgecolor='blue',
                linewidth=1.0
            )
            ax_own.add_patch(rect)

            # Draw bids for Owner
            for bp in box:
                ax_own.scatter(time_position, bp.utility_a,
                          marker='_',
                          s=30,
                          color='blue',
                          alpha=0.6,
                          zorder=10)

            # --- Plot Opponent Utility (Increasing) on ax_opp ---
            opp_utils = [bp.utility_b for bp in box]
            min_opp = min(opp_utils)
            max_opp = max(opp_utils)

            # Draw box range for Opponent
            rect_opp = Rectangle(
                (time_position - bar_width/2, min_opp),
                width=bar_width,
                height=max_opp - min_opp,
                facecolor='orange',
                alpha=0.3,
                edgecolor='orange',
                linewidth=1.0
            )
            ax_opp.add_patch(rect_opp)

            # Draw bids for Opponent
            for bp in box:
                ax_opp.scatter(time_position, bp.utility_b,
                          marker='|',
                          s=30,
                          color='orange',
                          alpha=0.6,
                          zorder=10)

        # --- Formatting Own Axis ---
        ax_own.axhline(y=nash_own, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Nash ({nash_own:.3f})')
        ax_own.set_xlabel('Negotiation Time', fontsize=10)
        ax_own.set_ylabel('Utility', fontsize=10)
        ax_own.set_title(f'{agent_name}: Own Utility', fontsize=12, fontweight='bold')
        ax_own.set_xlim(-0.1, 1.1)
        ax_own.set_ylim(0, 1.05)
        ax_own.grid(True, alpha=0.3)
        ax_own.legend(loc='upper right', fontsize=8)

        # --- Formatting Opponent Axis ---
        ax_opp.axhline(y=nash_opp, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Nash ({nash_opp:.3f})')
        ax_opp.set_xlabel('Negotiation Time', fontsize=10)
        ax_opp.set_ylabel('Utility', fontsize=10)
        ax_opp.set_title(f'{agent_name}: Opponent Utility', fontsize=12, fontweight='bold')
        ax_opp.set_xlim(-0.1, 1.1)
        ax_opp.set_ylim(0, 1.05)
        ax_opp.grid(True, alpha=0.3)
        ax_opp.legend(loc='upper right', fontsize=8)

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        """Log per-box metrics at each offer."""
        return self.get_pareto_box_metrics(session)

    def on_accept(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        """Log per-box metrics on acceptance."""
        return self.get_pareto_box_metrics(session)

    def on_fail(self, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        """Log per-box metrics on failure."""
        return self.get_pareto_box_metrics(session)

    def get_pareto_box_metrics(self, session: Union[Session, SessionLogs]) -> LogRow:
        """
        Calculate metrics per Pareto box for each estimator.
        """
        row = {}

        # Ensure boxes are created if before_session_start wasn't called (e.g. in tests)
        if not hasattr(self, 'true_boxes_a'):
             # Import here to avoid circular import
            from agents.ParetoWalkerAgent.pareto import ParetoWalker
            
            self.true_boxes_a = self._generate_agent_boxes(
                owner_pref=session.agentA.preference,
                opp_pref=session.agentB.preference,
                ParetoWalker=ParetoWalker
            )
            self.true_boxes_b = self._generate_agent_boxes(
                owner_pref=session.agentB.preference,
                opp_pref=session.agentA.preference,
                ParetoWalker=ParetoWalker
            )

        for estimator_id in range(len(session.agentA.estimators)):
            estimator_name = session.agentA.estimators[estimator_id].name

            # Calculate metrics for Agent A (A estimates B)
            box_metrics_a = self._calculate_box_metrics(
                true_boxes=self.true_boxes_a,
                true_opponent_pref=session.agentB.preference,
                estimated_opponent_pref=session.agentA.estimators[estimator_id].preference,
                agent_suffix="A"
            )

            # Calculate metrics for Agent B (B estimates A)
            box_metrics_b = self._calculate_box_metrics(
                true_boxes=self.true_boxes_b,
                true_opponent_pref=session.agentA.preference,
                estimated_opponent_pref=session.agentB.estimators[estimator_id].preference,
                agent_suffix="B"
            )

            # Calculate averaged metrics (A+B) per box
            averaged_metrics = {}
            box_count = box_metrics_a.get('BoxCount_A', 0)

            for box_idx in range(box_count):
                # Average RMSE
                rmse_a = box_metrics_a.get(f'Box{box_idx}_RMSE_A')
                rmse_b = box_metrics_b.get(f'Box{box_idx}_RMSE_B')
                if rmse_a is not None and rmse_b is not None:
                    averaged_metrics[f'Box{box_idx}_RMSE'] = (rmse_a + rmse_b) / 2

                # Average RMSE_Std
                std_a = box_metrics_a.get(f'Box{box_idx}_RMSE_Std_A')
                std_b = box_metrics_b.get(f'Box{box_idx}_RMSE_Std_B')
                if std_a is not None and std_b is not None:
                    averaged_metrics[f'Box{box_idx}_RMSE_Std'] = (std_a + std_b) / 2

                # Average correlations (Spearman, Kendall, Pearson)
                for metric_name in ['Spearman', 'Kendall', 'Pearson']:
                    val_a = box_metrics_a.get(f'Box{box_idx}_{metric_name}_A')
                    val_b = box_metrics_b.get(f'Box{box_idx}_{metric_name}_B')
                    if val_a is not None and val_b is not None:
                        averaged_metrics[f'Box{box_idx}_{metric_name}'] = (val_a + val_b) / 2

            # Calculate overall metrics (pooling all bids from all boxes)
            overall_metrics = self._calculate_overall_metrics(
                true_boxes_a=self.true_boxes_a,
                true_boxes_b=self.true_boxes_b,
                true_pref_a=session.agentA.preference,
                true_pref_b=session.agentB.preference,
                est_pref_a=session.agentA.estimators[estimator_id].preference,
                est_pref_b=session.agentB.estimators[estimator_id].preference
            )

            # Combine metrics
            row[estimator_name] = {
                **box_metrics_a,
                **box_metrics_b,
                **averaged_metrics,
                **overall_metrics
            }

        return row

    def _calculate_box_metrics(self, true_boxes, true_opponent_pref, estimated_opponent_pref, agent_suffix):
        """
        Calculate RMSE, Spearman, Kendall, Pearson for each box.
        """
        metrics = {}

        for box_idx, box in enumerate(true_boxes):
            # Extract bids from BidPoint objects
            bids = [bid_point.bid for bid_point in box]
            bid_count = len(bids)

            # Skip boxes with too few bids
            if bid_count < 2:
                metrics[f"Box{box_idx}_RMSE_{agent_suffix}"] = None
                metrics[f"Box{box_idx}_RMSE_Std_{agent_suffix}"] = None
                metrics[f"Box{box_idx}_Spearman_{agent_suffix}"] = None
                metrics[f"Box{box_idx}_Kendall_{agent_suffix}"] = None
                metrics[f"Box{box_idx}_Pearson_{agent_suffix}"] = None
                metrics[f"Box{box_idx}_BidCount_{agent_suffix}"] = bid_count
                continue

            # Calculate utilities (true vs estimated) for each bid
            utilities = [[true_opponent_pref.get_utility(bid), estimated_opponent_pref.get_utility(bid)]
                        for bid in bids]

            # Calculate RMSE and RMSE_Std
            errors = [utility[0] - utility[1] for utility in utilities]
            rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
            rmse_std = np.std(errors)

            # Prepare ranking indices - both must be sorted to compare rankings
            org_indices = list(range(len(bids)))
            agent_indices = list(range(len(bids)))

            # Sort by TRUE utility (utilities[i][0]) - this is the ground truth ranking
            org_indices = sorted(org_indices, key=lambda i: utilities[i][0], reverse=True)
            # Sort by ESTIMATED utility (utilities[i][1]) - this is the predicted ranking
            agent_indices = sorted(agent_indices, key=lambda i: utilities[i][1], reverse=True)

            # Calculate correlations
            spearman, kendall, pearson = None, None, None

            try:
                spearman, _ = spearmanr(org_indices, agent_indices)
            except:
                pass

            try:
                kendall, _ = kendalltau(org_indices, agent_indices)
            except:
                pass

            try:
                original_utilities = [utility[0] for utility in utilities]
                estimated_utilities = [utility[1] for utility in utilities]
                
                # Check for zero variance to avoid Pearson NA/warning
                if np.std(original_utilities) == 0 or np.std(estimated_utilities) == 0:
                    pearson = 0.0
                else:
                    pearson, _ = pearsonr(original_utilities, estimated_utilities)
            except:
                pass

            # Store metrics
            metrics[f"Box{box_idx}_RMSE_{agent_suffix}"] = rmse
            metrics[f"Box{box_idx}_RMSE_Std_{agent_suffix}"] = rmse_std
            metrics[f"Box{box_idx}_Spearman_{agent_suffix}"] = spearman
            metrics[f"Box{box_idx}_Kendall_{agent_suffix}"] = kendall
            metrics[f"Box{box_idx}_Pearson_{agent_suffix}"] = pearson
            metrics[f"Box{box_idx}_BidCount_{agent_suffix}"] = bid_count

        # Add total box count
        metrics[f"BoxCount_{agent_suffix}"] = len(true_boxes)

        return metrics

    def _calculate_overall_metrics(self, true_boxes_a, true_boxes_b,
                                    true_pref_a, true_pref_b,
                                    est_pref_a, est_pref_b):
        """
        Calculate overall metrics by pooling all bids from all boxes.

        Args:
            true_boxes_a: List of boxes for Agent A
            true_boxes_b: List of boxes for Agent B
            true_pref_a: Agent A's true preference
            true_pref_b: Agent B's true preference
            est_pref_a: Agent A's estimated preference for B
            est_pref_b: Agent B's estimated preference for A

        Returns:
            Dict with Overall_* metrics for both agents and averaged
        """
        metrics = {}

        # ======= Agent A Perspective (A estimates B) =======
        # Pool all bids from Agent A's boxes
        all_bids_a = []
        for box in true_boxes_a:
            all_bids_a.extend([bp.bid for bp in box])

        if len(all_bids_a) >= 2:
            # Calculate utilities for all bids (Agent A perspective)
            utilities_a = [[true_pref_b.get_utility(bid), est_pref_a.get_utility(bid)]
                           for bid in all_bids_a]

            # Calculate RMSE and std for Agent A
            errors_a = [u[0] - u[1] for u in utilities_a]
            rmse_a = np.sqrt(np.mean([e**2 for e in errors_a]))
            rmse_std_a = np.std(errors_a)

            # Calculate correlations for Agent A
            org_indices_a = list(range(len(all_bids_a)))
            agent_indices_a = list(range(len(all_bids_a)))
            org_indices_a = sorted(org_indices_a, key=lambda i: utilities_a[i][0], reverse=True)
            agent_indices_a = sorted(agent_indices_a, key=lambda i: utilities_a[i][1], reverse=True)

            spearman_a, kendall_a, pearson_a = None, None, None

            try:
                spearman_a, _ = spearmanr(org_indices_a, agent_indices_a)
            except:
                pass

            try:
                kendall_a, _ = kendalltau(org_indices_a, agent_indices_a)
            except:
                pass

            try:
                original_utils_a = [u[0] for u in utilities_a]
                estimated_utils_a = [u[1] for u in utilities_a]
                if np.std(original_utils_a) == 0 or np.std(estimated_utils_a) == 0:
                    pearson_a = 0.0
                else:
                    pearson_a, _ = pearsonr(original_utils_a, estimated_utils_a)
            except:
                pass

            # Store Agent A metrics
            metrics['Overall_RMSE_A'] = rmse_a
            metrics['Overall_RMSE_Std_A'] = rmse_std_a
            metrics['Overall_Spearman_A'] = spearman_a
            metrics['Overall_Kendall_A'] = kendall_a
            metrics['Overall_Pearson_A'] = pearson_a
        else:
            metrics['Overall_RMSE_A'] = None
            metrics['Overall_RMSE_Std_A'] = None
            metrics['Overall_Spearman_A'] = None
            metrics['Overall_Kendall_A'] = None
            metrics['Overall_Pearson_A'] = None

        # ======= Agent B Perspective (B estimates A) =======
        # Pool all bids from Agent B's boxes
        all_bids_b = []
        for box in true_boxes_b:
            all_bids_b.extend([bp.bid for bp in box])

        if len(all_bids_b) >= 2:
            # Calculate utilities for all bids (Agent B perspective)
            utilities_b = [[true_pref_a.get_utility(bid), est_pref_b.get_utility(bid)]
                           for bid in all_bids_b]

            # Calculate RMSE and std for Agent B
            errors_b = [u[0] - u[1] for u in utilities_b]
            rmse_b = np.sqrt(np.mean([e**2 for e in errors_b]))
            rmse_std_b = np.std(errors_b)

            # Calculate correlations for Agent B
            org_indices_b = list(range(len(all_bids_b)))
            agent_indices_b = list(range(len(all_bids_b)))
            org_indices_b = sorted(org_indices_b, key=lambda i: utilities_b[i][0], reverse=True)
            agent_indices_b = sorted(agent_indices_b, key=lambda i: utilities_b[i][1], reverse=True)

            spearman_b, kendall_b, pearson_b = None, None, None

            try:
                spearman_b, _ = spearmanr(org_indices_b, agent_indices_b)
            except:
                pass

            try:
                kendall_b, _ = kendalltau(org_indices_b, agent_indices_b)
            except:
                pass

            try:
                original_utils_b = [u[0] for u in utilities_b]
                estimated_utils_b = [u[1] for u in utilities_b]
                if np.std(original_utils_b) == 0 or np.std(estimated_utils_b) == 0:
                    pearson_b = 0.0
                else:
                    pearson_b, _ = pearsonr(original_utils_b, estimated_utils_b)
            except:
                pass

            # Store Agent B metrics
            metrics['Overall_RMSE_B'] = rmse_b
            metrics['Overall_RMSE_Std_B'] = rmse_std_b
            metrics['Overall_Spearman_B'] = spearman_b
            metrics['Overall_Kendall_B'] = kendall_b
            metrics['Overall_Pearson_B'] = pearson_b
        else:
            metrics['Overall_RMSE_B'] = None
            metrics['Overall_RMSE_Std_B'] = None
            metrics['Overall_Spearman_B'] = None
            metrics['Overall_Kendall_B'] = None
            metrics['Overall_Pearson_B'] = None

        # ======= Calculate Averaged Overall Metrics =======
        rmse_a = metrics.get('Overall_RMSE_A')
        rmse_b = metrics.get('Overall_RMSE_B')
        if rmse_a is not None and rmse_b is not None:
            metrics['Overall_RMSE'] = (rmse_a + rmse_b) / 2
            metrics['Overall_RMSE_Std'] = (metrics['Overall_RMSE_Std_A'] + metrics['Overall_RMSE_Std_B']) / 2
        else:
            metrics['Overall_RMSE'] = None
            metrics['Overall_RMSE_Std'] = None

        spearman_a = metrics.get('Overall_Spearman_A')
        spearman_b = metrics.get('Overall_Spearman_B')
        if spearman_a is not None and spearman_b is not None:
            metrics['Overall_Spearman'] = (spearman_a + spearman_b) / 2
        else:
            metrics['Overall_Spearman'] = None

        kendall_a = metrics.get('Overall_Kendall_A')
        kendall_b = metrics.get('Overall_Kendall_B')
        if kendall_a is not None and kendall_b is not None:
            metrics['Overall_Kendall'] = (kendall_a + kendall_b) / 2
        else:
            metrics['Overall_Kendall'] = None

        pearson_a = metrics.get('Overall_Pearson_A')
        pearson_b = metrics.get('Overall_Pearson_B')
        if pearson_a is not None and pearson_b is not None:
            metrics['Overall_Pearson'] = (pearson_a + pearson_b) / 2
        else:
            metrics['Overall_Pearson'] = None

        return metrics
