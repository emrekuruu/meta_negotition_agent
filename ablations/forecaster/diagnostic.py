#!/usr/bin/env python3
"""
Simple diagnostic using ACTUAL session data and the SAME evaluation logic.
No multithreading, no GPU complexity - just one session, one round at a time.
Plots Full Timeline with Forecast vs. Self vs. Oracle.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

os.environ['WORKER_GPU_ID'] = '0'

from agents.NegoformerAgent.forecasting.forecaster import Forecaster
from nenv.EditablePreference import EditablePreference
from nenv.Bid import Bid
from dotenv import load_dotenv
from ablations.forecaster.metrics import calculate_rmse, calculate_mape, calculate_crps, calculate_mase

load_dotenv()


def load_one_session(base_dir: str = "/Users/emrekuru/Developer/Thesis/Negoformer/strategy/results/oracle/1000", preferred_model: str = 'Oracle'):
    """Load the first valid ParetoWalker session with weights, prioritizing preferred_model."""
    from pathlib import Path
    
    # Model types mapping: (filename suffix, sheet name)
    # Put preferred model first
    all_models = [
        ('Bayesian', '_BayesianOpponentModel.xlsx', 'Bayesian Opponent Model Weights'),
        ('ClassicFrequency', '_ClassicFrequencyOpponentModel.xlsx', 'Classic Frequency Opponent Model Weights'),
        ('CUHK', '_CUHKOpponentModel.xlsx', 'CUHK Frequency Opponent Model Weights'),
        ('WindowedFrequency', '_WindowedFrequencyOpponentModel.xlsx', 'Windowed Frequency Opponent Model Weights'),
        ('ConflictBased', '_ConflictBasedOpponentModel.xlsx', 'Conflict-Based Opponent Model Weights'),
        ('Expectation', '_ExpectationCOMBOpponentModel.xlsx', 'Expectation COMB Opponent Model Weights')
    ]
    
    # Sort models to prioritize preferred_model
    sorted_models = sorted(all_models, key=lambda x: 0 if x[0] == preferred_model else 1)
    
    for domain_dir in Path(base_dir).iterdir():
        if not domain_dir.is_dir():
            continue
        
        sessions_dir = domain_dir / "sessions"
        if not sessions_dir.exists():
            continue
        
        for session_file in sessions_dir.iterdir():
            if not session_file.name.endswith('.xlsx'):
                continue
            if 'ParetoWalker' not in session_file.name:
                continue
            
            # Try each opponent model type in order
            for mode_name, suffix, weight_sheet_name in sorted_models:
                if suffix not in session_file.name:
                    continue
                    
                try:
                    session_df = pd.read_excel(session_file, sheet_name="Session")
                    weights_df = pd.read_excel(session_file, sheet_name=weight_sheet_name)
                    
                    print(f"Loaded: {session_file.name}")
                    print(f"  Mode: {mode_name}")
                    print(f"  Session rows: {len(session_df)}")
                    print(f"  Weights rows: {len(weights_df)}")
                    
                    # Determine perspective
                    parts = session_file.name.replace('.xlsx', '').split('_')
                    agent_a = parts[0]
                    agent_b = parts[1]
                    
                    if 'ParetoWalker' in agent_a:
                        opponent_who = 'B'
                        actual_col = 'AgentBUtility'
                    else:
                        opponent_who = 'A'
                        actual_col = 'AgentAUtility'
                    
                    return {
                        'session_df': session_df,
                        'weights_df': weights_df,
                        'opponent_who': opponent_who,
                        'actual_col': actual_col,
                        'domain': domain_dir.name,
                        'filename': session_file.name,
                        'mode': mode_name
                    }
                except Exception as e:
                    print(f"Error loading {session_file.name}: {e}")
                    continue
    
    return None


def recalculate_utilities_for_round_simple(
    round_idx: int,
    weights_df: pd.DataFrame,
    session_df: pd.DataFrame,
    opponent_who: str
) -> list:
    """
    Recalculate utilities for ALL opponent bids (Past + Future) 
    using the weights known at round_idx.
    
    Returns list of (estimated_utility, bid_index) tuples for the ENTIRE session.
    """
    # Map opponent bid index to weights row
    opponent_mask = session_df['Who'] == opponent_who
    opponent_action_indices = session_df[opponent_mask].index.tolist()
    
    if round_idx >= len(opponent_action_indices):
        print(f"round_idx {round_idx} >= len(opponent_action_indices) {len(opponent_action_indices)}")
        return []
    
    overall_action_idx = opponent_action_indices[round_idx]
    
    if overall_action_idx >= len(weights_df):
        # Fallback for end of session mismatch
        overall_action_idx = len(weights_df) - 1
    
    weights_row = weights_df.iloc[overall_action_idx]
    
    # Build preference from weights
    issue_weights = {}
    issues = {}
    
    prefix = f"A_estimates_{opponent_who}_" if opponent_who == "B" else f"B_estimates_{opponent_who}_"
    
    for col in weights_row.index:
        if not col.startswith(prefix):
            continue
        
        suffix = col[len(prefix):]
        
        if suffix.endswith("_issue_weight"):
            issue_name = suffix.replace("_issue_weight", "")
            issue_weights[issue_name] = weights_row[col]
            if issue_name not in issues:
                issues[issue_name] = {}
        
        elif suffix.endswith("_weight") and not suffix.endswith("_issue_weight"):
            parts = suffix.rsplit("_", 1)[0].rsplit("_", 1)
            if len(parts) == 2:
                issue_name, value_name = parts
                if issue_name not in issues:
                    issues[issue_name] = {}
                issues[issue_name][value_name] = weights_row[col]
    
    # Create preference
    try:
        estimated_pref = EditablePreference(
            issue_weights=issue_weights,
            issues=issues,
            reservation_value=0.0,
            generate_bids=False
        )
    except Exception as e:
        print(f"Error creating preference: {e}")
        return []
    
    # Recalculate utilities for ALL opponent bids (Past AND Future)
    results = []
    
    for idx in opponent_action_indices:
        bid_content = session_df.iloc[idx]['BidContent']
        
        try:
            bid_str = str(bid_content).replace("'", '"')
            bid_dict = json.loads(bid_str)
            bid = Bid(bid_dict)
            utility = estimated_pref.get_utility(bid)
            results.append((utility, idx))
        except Exception as e:
            results.append((0.0, idx))
    
    return results


def run_simple_diagnostic(preferred_model='Bayesian', use_oracle_mode=False):
    """
    Run diagnostic on ONE session.
    Calculates metrics for BOTH targets (Self and Oracle) and plots a single timeline.

    Args:
        preferred_model: Name of the opponent model to prioritize loading.
        use_oracle_mode: If True, use Oracle->Oracle (input=oracle, target=oracle).
                        If False, use OpponentModel->Oracle/Self (input=estimates, targets=both).
    """

    mode_str = "Oracle -> Oracle" if use_oracle_mode else f"{preferred_model} -> Self & Oracle"
    print("=" * 80)
    print(f"SIMPLE DIAGNOSTIC: {mode_str}")
    print("=" * 80)
    
    # Load one session
    session_data = load_one_session(preferred_model=preferred_model)
    if session_data is None:
        print(f"No valid session found for model {preferred_model}!")
        return
    
    session_df = session_data['session_df']
    weights_df = session_data['weights_df']
    opponent_who = session_data['opponent_who']
    actual_col = session_data['actual_col']
    domain = session_data['domain']
    
    print(f"Domain: {domain}")
    
    # Get opponent bids only
    opponent_mask = session_df['Who'] == opponent_who
    session_df_opponent = session_df[opponent_mask].reset_index(drop=True)
    
    print(f"\nOpponent ({opponent_who}) bids: {len(session_df_opponent)}")
    print(f"Actual utility column: {actual_col}")
    
    # Target 2: Oracle Truth (Actual Utilities from File)
    oracle_utilities_full = session_df_opponent[actual_col].tolist()
    timestamps = session_df_opponent['Time'].tolist() if 'Time' in session_df_opponent.columns else list(range(len(session_df_opponent)))
    
    # Initialize forecaster
    print("\nInitializing forecaster...")
    deadline = 1000
    forecaster = Forecaster(max_context=deadline // 2, prediction_length=deadline // 4, deadline=deadline)
    
    # Test at different rounds
    test_rounds = [50, 100, 200, 300, 400, 500]
    test_rounds = [r for r in test_rounds if r < len(session_df_opponent) - 10]
    
    if not test_rounds:
        print("Session too short for testing!")
        return
    
    print(f"\nTesting at rounds: {test_rounds}")
    
    for round_idx in test_rounds:
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_idx}")
        print(f"{'=' * 60}")

        if use_oracle_mode:
            # Oracle Mode: Input = Oracle, Target = Oracle
            input_history = oracle_utilities_full[:round_idx + 1]
            full_estimated_utilities = None  # Not used in Oracle mode
            target_future_self = None  # Not applicable in Oracle mode
        else:
            # Opponent Model Mode: Input = Estimates, Targets = Self & Oracle
            # Get recalculated (estimated) utilities using weights at this round
            recalc_results = recalculate_utilities_for_round_simple(
                round_idx, weights_df, session_df, opponent_who
            )

            if not recalc_results:
                print("  No recalculated results!")
                continue

            full_estimated_utilities = [u for u, _ in recalc_results]
            input_history = full_estimated_utilities[:round_idx + 1]
            target_future_self = full_estimated_utilities[round_idx + 1:]

        # Target: Oracle (Actual Future) - used in both modes
        target_future_oracle = oracle_utilities_full[round_idx + 1:]

        timestamps_historical = timestamps[:round_idx + 1]
        timestamps_future = timestamps[round_idx + 1:]

        # Build input for forecaster
        input_data = list(zip(input_history, timestamps_historical))
        
        # Get forecast
        forecast = forecaster(input_data)

        # === Calculate Metrics ===
        if use_oracle_mode:
            comparison_length = min(len(forecast), len(target_future_oracle))
        else:
            comparison_length = min(len(forecast), len(target_future_self), len(target_future_oracle))

        # Initialize all metrics
        metrics_self = {}
        metrics_oracle = {}

        if comparison_length > 0:
            forecast_slice = forecast[:comparison_length]

            # Vs Self (only in opponent model mode)
            if not use_oracle_mode and target_future_self is not None:
                self_slice = np.array(target_future_self[:comparison_length])
                try:
                    metrics_self['rmse'] = calculate_rmse(self_slice, forecast_slice)
                except ValueError:
                    pass

                try:
                    metrics_self['mape'] = calculate_mape(self_slice, forecast_slice)
                except ValueError:
                    pass

                try:
                    metrics_self['crps'] = calculate_crps(self_slice, forecast_slice)
                except ValueError:
                    pass

                try:
                    training_series = np.array(full_estimated_utilities[:round_idx + 1])
                    metrics_self['mase'] = calculate_mase(self_slice, forecast_slice, training_series, seasonality=1)
                except ValueError:
                    pass

            # Vs Oracle
            oracle_slice = np.array(target_future_oracle[:comparison_length])
            try:
                metrics_oracle['rmse'] = calculate_rmse(oracle_slice, forecast_slice)
            except ValueError:
                pass

            try:
                metrics_oracle['mape'] = calculate_mape(oracle_slice, forecast_slice)
            except ValueError:
                pass

            try:
                metrics_oracle['crps'] = calculate_crps(oracle_slice, forecast_slice)
            except ValueError:
                pass

            try:
                training_series = np.array(oracle_utilities_full[:round_idx + 1])
                metrics_oracle['mase'] = calculate_mase(oracle_slice, forecast_slice, training_series, seasonality=1)
            except ValueError:
                pass

            # Print metrics
            if metrics_self:
                print(f"  Target: SELF")
                for metric_name, value in metrics_self.items():
                    if metric_name == 'mape':
                        print(f"    {metric_name.upper():6s}: {value:.2f}%")
                    else:
                        print(f"    {metric_name.upper():6s}: {value:.4f}")

            if metrics_oracle:
                print(f"  Target: ORACLE")
                for metric_name, value in metrics_oracle.items():
                    if metric_name == 'mape':
                        print(f"    {metric_name.upper():6s}: {value:.2f}%")
                    else:
                        print(f"    {metric_name.upper():6s}: {value:.4f}")
        
        # === PLOT: Full Timeline Only ===
        fig, ax = plt.subplots(figsize=(14, 8))

        if use_oracle_mode:
            # Oracle Mode: Input = Oracle
            ax.plot(timestamps_historical, input_history, 'k-', label='Input (Oracle History)', alpha=0.7, linewidth=2)
        else:
            # Opponent Model Mode: Input = Estimates
            ax.plot(timestamps_historical, input_history, 'b-', label='Input (Estimated History)', alpha=0.9, linewidth=1.5)
            # Show oracle as reference
            ax.plot(timestamps_historical, oracle_utilities_full[:round_idx+1], 'k--', label='Ref: Oracle History', alpha=0.3)

        # Future Self Target (only in opponent model mode)
        if not use_oracle_mode and target_future_self is not None and len(timestamps_future) > 0:
            ax.plot(timestamps_future, target_future_self[:len(timestamps_future)], 'g:', label='Target 1: Self (Model Future)', alpha=0.8, linewidth=2)

        # Future Oracle Target
        if len(timestamps_future) > 0:
            ax.plot(timestamps_future, target_future_oracle[:len(timestamps_future)], 'k-', label='Target: Oracle (Actual Future)', alpha=0.2, linewidth=4)
        
        # 5. Forecast
        forecast_timestamps = np.linspace(
            timestamps_historical[-1],
            min(1.0, timestamps_historical[-1] + (len(forecast)/deadline)),
            len(forecast)
        )
        ax.plot(forecast_timestamps, forecast, 'r--', label='Forecast', alpha=0.9, linewidth=2.5)
            
        # Current time marker
        ax.axvline(x=timestamps_historical[-1], color='purple', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('Time (0 to 1)')
        ax.set_ylabel('Utility')

        # Build title with available metrics
        mode_label = "Oracle" if use_oracle_mode else session_data['mode']
        title_parts = [f"Forecast Diagnostic | Round {round_idx} | Mode: {mode_label}"]

        # Format SELF metrics (only in opponent model mode)
        if metrics_self:
            self_str = "VS SELF: " + ", ".join([
                f"{k.upper()}={v:.3f}" if k != 'mape' else f"{k.upper()}={v:.1f}%"
                for k, v in metrics_self.items()
            ])
            title_parts.append(self_str)

        # Format ORACLE metrics
        if metrics_oracle:
            oracle_str = "VS ORACLE: " + ", ".join([
                f"{k.upper()}={v:.3f}" if k != 'mape' else f"{k.upper()}={v:.1f}%"
                for k, v in metrics_oracle.items()
            ])
            title_parts.append(oracle_str)

        ax.set_title("\n".join(title_parts))
        
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        forecaster_name = os.environ.get('FORECASTER_MODEL', 'toto').lower()
        filename = f"ablations/forecaster/plots/{forecaster_name}_{round_idx}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Oracle', help='Preferred opponent model to load (ignored if --oracle is used)')
    parser.add_argument('--oracle', action='store_true', help='Use Oracle->Oracle mode instead of OpponentModel mode')
    args = parser.parse_args()

    run_simple_diagnostic(preferred_model=args.model, use_oracle_mode=args.oracle)
