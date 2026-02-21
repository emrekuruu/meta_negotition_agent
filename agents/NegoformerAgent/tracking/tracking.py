import os
import matplotlib.pyplot as plt
import numpy as np
from nenv import Bid
from typing import Tuple


class NegoformerTrackerMixin:
    def _determine_plot_dir(self, opponent_name: str) -> str:
        return f'plots/{self.deadline}/{self.horizon}/{opponent_name}_{os.getpid()}'

    def plot_prediction_schedule(self):
        """
        Plot a standalone prediction schedule, independent from negotiation data.

        Shows:
        - When each prediction is made (forecast round)
        - Which timesteps evaluate that prediction via the ACI buffer window
        - Total number of predictions and frequency
        """
        prediction_rounds = self.forecasting_rounds
        if not prediction_rounds:
            return

        total_predictions = len(prediction_rounds)
        forecast_frequency = max(1, self.forecast_frequency)
        eval_offset_start = self.horizon - self.aci_buffer
        eval_offset_end = self.horizon + self.aci_buffer

        fig_height = min(max(4.0, 1.0 + (0.35 * total_predictions)), 16.0)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        for idx, prediction_round in enumerate(prediction_rounds):
            eval_start = prediction_round + eval_offset_start
            eval_end_exclusive = prediction_round + eval_offset_end

            clipped_start = max(0, eval_start)
            clipped_end = min(self.deadline, eval_end_exclusive - 1)

            if clipped_start <= clipped_end:
                ax.hlines(
                    y=idx,
                    xmin=clipped_start,
                    xmax=clipped_end,
                    color='tab:blue',
                    linewidth=4,
                    alpha=0.65,
                    label='Evaluation timesteps' if idx == 0 else None,
                )

            if 0 <= prediction_round <= self.deadline:
                ax.scatter(
                    prediction_round,
                    idx,
                    color='tab:red',
                    s=28,
                    zorder=3,
                    label='Prediction made' if idx == 0 else None,
                )

        ax.axvline(
            x=self.forecasting_ready_threshold,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label=f'Forecasting start round ({self.forecasting_ready_threshold})',
        )

        ax.set_xlim(0, self.deadline)
        ax.set_ylim(-1, total_predictions)
        ax.set_xlabel('Round (linear scale)')
        ax.set_ylabel('Prediction index')

        tick_step = max(1, total_predictions // 20)
        tick_indices = [i for i in range(total_predictions) if (i % tick_step == 0) or (i == total_predictions - 1)]
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([f'P{i} @ r={prediction_rounds[i]}' for i in tick_indices])

        if self.deadline > 0:
            top_axis = ax.secondary_xaxis(
                'top',
                functions=(lambda round_idx: round_idx / self.deadline, lambda norm_t: norm_t * self.deadline),
            )
            top_axis.set_xlabel('Normalized time')

        ax.set_title(
            f'Total predictions: {total_predictions} | '
            f'Frequency: approx every {forecast_frequency} rounds | '
            f'Buffer: +/-{self.aci_buffer} around horizon {self.horizon}'
        )
        ax.grid(axis='x', alpha=0.25)
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plot_path = f'{self.plot_dir}/prediction_schedule.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(
            f"Saved prediction schedule to {plot_path} | "
            f"total_predictions={total_predictions}, "
            f"frequency={forecast_frequency}, "
            f"evaluation_offset=[{eval_offset_start}, {eval_offset_end})"
        )

    def _fit_forecast_line(self, forecast: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a line to the forecast and extend it to the end of negotiation.

        Args:
            forecast: Predicted opponent utilities
            t: Current time

        Returns:
            Tuple of (extended_times, extended_utilities) from current time to 1.0
        """
        num_forecast_points = len(forecast)
        forecast_times = np.linspace(t, min(t + (num_forecast_points / self.deadline), 1.0), num_forecast_points)

        # Fit linear regression to the forecast
        # y = mx + b
        coeffs = np.polyfit(forecast_times, forecast, deg=1)
        slope, intercept = coeffs

        # Extend line from current time to end of negotiation
        extended_times = np.linspace(t, 1.0, 100)
        extended_utilities = slope * extended_times + intercept

        return extended_times, extended_utilities

    def _plot_forecast(self, forecast: np.ndarray, t: float, target_time: float):
        """
        Plot opponent utility forecast with historical offers.

        Args:
            forecast: Median forecast array
            t: Current time
            target_time: Calculated target time to reach desired point
        """
        median = forecast

        # Extract historical opponent offers
        historical_utilities = [utility for utility, _ in self.forecast_data]
        historical_times = [time for _, time in self.forecast_data]

        # Create future time points (normalized 0-1)
        num_forecast_points = len(median)
        future_times = np.linspace(t, min(t + (num_forecast_points / self.deadline), 1.0), num_forecast_points)

        # Get desired opponent utility threshold
        desired_utility_point = self.pareto_state['desired_utility_point']
        desired_opponent_utility = desired_utility_point['opponent_utility']

        # Also get Nash for comparison (optional visualization)
        nash_opponent_utility = None
        if self.bid_space:
            nash_bid = self.get_nash_bid()
            if nash_bid:
                nash_opponent_utility = nash_bid.utility_b

        # Plot
        plt.figure(figsize=(14, 7))

        # Plot historical opponent offers
        if historical_times:
            plt.scatter(historical_times, historical_utilities, c='blue', s=30, alpha=0.6,
                       label='Opponent Historical Offers', marker='o')
            plt.plot(historical_times, historical_utilities, 'b-', alpha=0.3, linewidth=1)

        # Plot forecast median
        plt.plot(future_times, median, 'r-', label='Forecasted Median', linewidth=2.5)

        # Fit and plot extended forecast line
        extended_times, extended_utilities = self._fit_forecast_line(median, t)
        plt.plot(extended_times, extended_utilities, 'r--', label='Forecast Trend Line', linewidth=2, alpha=0.7)

        # Current time marker
        plt.axvline(x=t, color='g', linestyle=':', label=f'Current Time (t={t:.2f})', linewidth=1.5)

        # Add desired threshold line (adaptive)
        if desired_opponent_utility is not None:
            plt.axhline(y=desired_opponent_utility, color='orange', linestyle='--',
                       label=f'Adaptive Desired Opponent Utility ({desired_opponent_utility:.3f})', linewidth=1.5)

        # Add Nash threshold line for reference
        if nash_opponent_utility is not None:
            plt.axhline(y=nash_opponent_utility, color='red', linestyle=':',
                       label=f'Nash Opponent Utility ({nash_opponent_utility:.3f})', linewidth=1.0, alpha=0.7)

        # Add target time line
        if target_time < 1.0:
            plt.axvline(x=target_time, color='purple', linestyle='-.',
                       label=f'Target Time (t={target_time:.3f})', linewidth=1.5)

        plt.xlabel('Normalized Time', fontsize=12)
        plt.ylabel('Opponent Utility', fontsize=12)

        plt.title(f'Opponent Utility Forecast at t={t:.2f}', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/opponent_forecast_t_{t:.2f}.png', dpi=150)
        plt.close()

    def _track_metrics(self, t: float, bid: Bid, opponent_utility: float):
        """Track metrics for final visualization."""
        # Get current target_time
        target_time = self.pareto_state['target_time'] if self.pareto_state else 1.0

        # Get Nash utility
        nash_bid = self.get_nash_bid()
        nash_utility = nash_bid.utility_b if nash_bid else 0.5

        # Get our utility for opponent's bid
        our_utility = self.preference.get_utility(bid)

        # Store
        self.tracking['time'].append(t)
        self.tracking['target_time'].append(target_time)
        self.tracking['opponent_utility'].append(opponent_utility)
        self.tracking['nash_utility'].append(nash_utility)
        self.tracking['our_utility'].append(our_utility)

    def plot_negotiation_summary(self, opponent_name: str = "Opponent"):
        """
        Create a comprehensive final plot showing:
        - Target time progression
        - Opponent utility progression
        - Nash point
        """
        if not self.tracking['time']:
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f'Negotiation Summary: {self.name} vs {opponent_name}', fontsize=14, fontweight='bold')

        times = self.tracking['time']

        # Plot 1: Target Time
        ax1 = axes[0]
        # Plot safe position FIRST (behind target time)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Safe Position (1.0)', zorder=1)
        # Plot target time with higher z-order and markers to make it visible even at 1.0
        ax1.plot(times, self.tracking['target_time'], 'b-', linewidth=3, label='Target Time', zorder=2)
        ax1.scatter(times[::max(1, len(times)//20)], [self.tracking['target_time'][i] for i in range(0, len(times), max(1, len(times)//20))],
                    c='blue', s=30, zorder=3, alpha=0.7)
        ax1.set_ylabel('Target Time')
        ax1.set_ylim(0, 1.1)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Target Time')

        # Plot 2: Opponent Utility vs Nash
        ax2 = axes[1]
        ax2.plot(times, self.tracking['opponent_utility'], 'r-', linewidth=2, label='Opponent Utility (their bids)')
        ax2.plot(times, self.tracking['nash_utility'], 'purple', linestyle='--', linewidth=2, label='Nash Utility')
        ax2.set_ylabel('Utility')
        ax2.set_xlabel('Normalized Time')
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Opponent Utility vs Nash Point')

        plt.tight_layout()
        plot_path = f'{self.plot_dir}/negotiation_summary_{opponent_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved negotiation summary to {plot_path}")

    def plot_observation_intervals(self, opponent_name: str = "Opponent"):
        """
        Plot vanilla ACI-issued intervals with actual values (hits/misses).
        """
        history = self.aci_tracker.get_observation_history()
        if not history:
            return

        all_rounds = [int(t * self.deadline) for t in self.tracking.get('time', [])]
        all_actuals = self.tracking.get('opponent_utility', [])
        matured_round_set = {int(d['round']) for d in history}
        non_matured_rounds = [r for r in all_rounds if r not in matured_round_set]
        non_matured_actuals = [a for r, a in zip(all_rounds, all_actuals) if r not in matured_round_set]

        rounds = [d['round'] for d in history]
        actuals = [d['actual'] for d in history]
        predicted = [d['predicted'] for d in history]
        ignored_flags = [bool(d.get('ignored', False)) for d in history]
        evaluated_indices = [i for i, ignored in enumerate(ignored_flags) if not ignored]
        ignored_indices = [i for i, ignored in enumerate(ignored_flags) if ignored]

        lowers = []
        uppers = []
        upper_interval_widths = []
        for d in history:
            interval = d.get('interval')
            p = float(d['predicted'])
            if isinstance(interval, (tuple, list, np.ndarray)) and len(interval) == 2:
                lower = float(interval[0])
                upper = float(interval[1])
            else:
                lower = p
                upper = p

            lowers.append(lower)
            uppers.append(upper)
            stored_upper_width = d.get('upper_interval_width')
            if stored_upper_width is None:
                upper_interval_widths.append(max(0.0, upper - p))
            else:
                upper_interval_widths.append(float(stored_upper_width))

        fig, (ax, ax_width) = plt.subplots(
            2,
            1,
            figsize=(14, 9),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1]},
        )

        # Clamp for display: upper should not go below median, lower should not go above median
        display_uppers = [max(u, p) for u, p in zip(uppers, predicted)]
        display_lowers = [min(l, p) for l, p in zip(lowers, predicted)]
        fixed_uppers = [p + 0.05 for p in predicted]
        ax.fill_between(rounds, predicted, fixed_uppers, alpha=0.2, color='red', label='Fixed interval (0.05)')
        ax.fill_between(rounds, predicted, display_uppers, alpha=0.3, color='blue', label='Upper interval (ACI)')
        ax.fill_between(rounds, display_lowers, predicted, alpha=0.3, color='orange', label='Lower interval (ACI)')
        ax.plot(rounds, predicted, 'k-', alpha=0.5, linewidth=0.7, label='Predicted (median)')

        hit_rounds = [rounds[i] for i in evaluated_indices if bool(history[i]['hit'])]
        hit_actuals = [actuals[i] for i in evaluated_indices if bool(history[i]['hit'])]
        miss_rounds = [rounds[i] for i in evaluated_indices if not bool(history[i]['hit'])]
        miss_actuals = [actuals[i] for i in evaluated_indices if not bool(history[i]['hit'])]
        ignored_rounds = [rounds[i] for i in ignored_indices]
        ignored_actuals = [actuals[i] for i in ignored_indices]

        if hit_rounds:
            ax.scatter(hit_rounds, hit_actuals, c='green', s=20, alpha=0.7, label=f'Hit ({len(hit_rounds)})', zorder=3)
        if miss_rounds:
            ax.scatter(miss_rounds, miss_actuals, c='red', s=20, alpha=0.7, label=f'Miss ({len(miss_rounds)})', zorder=3)
        if ignored_rounds:
            ax.scatter(ignored_rounds, ignored_actuals, c='goldenrod', s=18, alpha=0.75, label=f'Ignored ({len(ignored_rounds)})', zorder=3)
        if non_matured_rounds:
            ax.scatter(
                non_matured_rounds,
                non_matured_actuals,
                c='gray',
                s=12,
                alpha=0.45,
                label=f'Not matured ({len(non_matured_rounds)})',
                zorder=2,
            )

        total = len(evaluated_indices)
        misses = len(miss_rounds)
        err_rate = misses / total if total > 0 else 0.0
        alpha_history = self.aci_tracker.get_alpha_history()
        final_alpha = float(alpha_history[-1]['alpha_next']) if alpha_history else self.aci_tracker.target_alpha

        ax.set_ylabel('Utility', fontsize=10)
        ax.set_title(
            f'{total} evaluated, {misses} misses ({err_rate:.1%}), '
            f'ignored={len(ignored_rounds)}, final alpha_t={final_alpha:.3f}'
        )
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        ax_width.plot(rounds, upper_interval_widths, color='blue', linewidth=1.6, label='Upper interval width (ACI)')
        ax_width.fill_between(rounds, 0, upper_interval_widths, alpha=0.15, color='blue')
        ax_width.axhline(y=0.05, color='red', linestyle='--', linewidth=1.2, label='Fixed width (0.05)')
        ax_width.set_xlabel('Round', fontsize=12)
        ax_width.set_ylabel('Width', fontsize=10)
        ax_width.set_ylim(bottom=0)
        ax_width.grid(True, alpha=0.3)
        ax_width.legend(loc='best', fontsize=8)

        fig.suptitle(f'Prediction Intervals vs Actual Values - {opponent_name}', fontsize=14)
        plt.tight_layout()

        plot_path = f'{self.plot_dir}/observation_intervals_{opponent_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved observation intervals plot to {plot_path}")

    def plot_interval_alphas(self, opponent_name: str = "Opponent"):
        """Plot alpha_t trajectories produced by vanilla ACI trackers."""
        history = self.aci_tracker.get_alpha_history()
        if not history:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        rounds = [d['round'] for d in history]
        alphas = [d['alpha_next'] for d in history]
        ax.plot(rounds, alphas, linewidth=2, label='alpha_t')

        ax.axhline(
            y=self.aci_tracker.target_alpha,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'Target alpha={self.aci_tracker.target_alpha:.3f}',
        )
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('alpha_t', fontsize=12)
        ax.set_title(f'ACI Alpha Trajectory - {opponent_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        plot_path = f'{self.plot_dir}/interval_alphas_{opponent_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved interval alpha plot to {plot_path}")

    def _save_aci_csv(self, opponent_name: str) -> None:
        history = self.aci_tracker.get_observation_history()
        if not history:
            return

        csv_path = f'{self.plot_dir}/aci.csv'

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('horizon,opponent,round,predicted,actual,upper_interval_width,alpha_t,ignored,hit\n')
            for d in history:
                f.write(
                    f"{self.horizon},{opponent_name},{int(d['round'])},"
                    f"{float(d['predicted']):.6f},"
                    f"{float(d['actual']):.6f},"
                    f"{float(d['upper_interval_width']):.6f},"
                    f"{float(d['alpha_next']):.6f},"
                    f"{int(bool(d['ignored']))},"
                    f"{int(bool(d['hit']))}\n"
                )
        print(f"Saved ACI csv to {csv_path}")
