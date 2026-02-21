import os
import sys
import argparse
import pickle
from pathlib import Path
from multiprocessing import Pool, set_start_method
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ablations.forecaster.metrics import calculate_rmse, calculate_mape, calculate_crps, calculate_mase


# ============================================================================
# Configuration
# ============================================================================

OPPONENT_MODEL_MAPPINGS = {
    'Oracle': ('Oracle', None),
    'ClassicFrequencyOpponentModel': ('ClassicFrequency', 'Classic Frequency Opponent Model'),
    'CUHKOpponentModel': ('CUHK', 'CUHK Frequency Opponent Model'),
    'BayesianOpponentModel': ('Bayesian', 'Bayesian Opponent Model'),
    'WindowedFrequencyOpponentModel': ('WindowedFrequency', 'Frequency Window Opponent Model'),
    'ConflictBasedOpponentModel': ('ConflictBased', 'Conflict-Based Opponent Model'),
    'StepwiseCOMBOpponentModel': ('StepwiseCOMB', 'Stepwise COMB Opponent Model'),
    'ExpectationCOMBOpponentModel': ('ExpectationCOMB', 'Expectation COMB Opponent Model'),
}

ALL_MODES = [mapping[0] for mapping in OPPONENT_MODEL_MAPPINGS.values()]


# ============================================================================
# Session Loading (runs ONCE in main process)
# ============================================================================

def get_domains(base_dir: str) -> set:
    return {d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))}


def detect_mode_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    for suffix, (mode_name, sheet_name) in OPPONENT_MODEL_MAPPINGS.items():
        if f'_{suffix}.xlsx' in filename:
            return mode_name, sheet_name
    return None, None


def load_session_data(file_path: Path) -> Optional[Dict]:
    result = {'filename': file_path.name, 'opponent_models': {}}
    try:
        mode_name, sheet_name = detect_mode_from_filename(file_path.name)
        if mode_name is None:
            return None
        
        result['mode'] = mode_name
        result['sheet_name'] = sheet_name
        
        parts = file_path.name.replace('.xlsx', '').split('_')
        result['agent_a'] = parts[0]
        result['agent_b'] = parts[1]
        result['domain'] = parts[2]
        
        if 'ParetoWalker' not in result['agent_a'] and 'ParetoWalker' not in result['agent_b']:
            return None
        
        session_df = pd.read_excel(file_path, sheet_name="Session")
        result['session_df'] = session_df

        # Load bid contents for utility recalculation
        if 'BidContent' in session_df.columns:
            result['bid_contents'] = session_df['BidContent'].tolist()
        else:
            return None  # Cannot recalculate without bids

        accept_idx = session_df[session_df['Action'] == 'Accept'].index
        result['accept_row_idx'] = accept_idx[0] if len(accept_idx) > 0 else len(session_df) - 1
        
        if sheet_name is not None:
            try:
                opp_model_df = pd.read_excel(file_path, sheet_name=sheet_name)
                result['opponent_models'][sheet_name] = opp_model_df

                # Load weights sheet for utility recalculation
                weights_sheet_name = f"{sheet_name} Weights"
                weights_df = pd.read_excel(file_path, sheet_name=weights_sheet_name)
                result['weights_df'] = weights_df
            except Exception as e:
                print(f"Error loading sheets for {file_path.name}: {e}")
                return None
    except:
        return None
    return result


def get_paretowaker_perspective(session_data: Dict) -> Optional[Dict]:
    agent_a = session_data['agent_a']
    agent_b = session_data['agent_b']
    if 'ParetoWalker' in agent_a:
        return {
            'opponent_name': agent_b,
            'is_agent_a': True,
            'opponent_who': 'B',
            'estimated_column': 'EstimatedOpponentUtilityB',
            'actual_column': 'AgentBUtility'
        }
    elif 'ParetoWalker' in agent_b:
        return {
            'opponent_name': agent_a,
            'is_agent_a': False,
            'opponent_who': 'A',
            'estimated_column': 'EstimatedOpponentUtilityA',
            'actual_column': 'AgentAUtility'
        }
    return None


def get_domain_profile(domain: str, is_agent_a: bool):
    """
    Load the domain profile for the forecasting agent.

    Args:
        domain: Domain name (e.g., "Domain0")
        is_agent_a: True if ParetoWalker is AgentA, False if AgentB

    Returns:
        Preference object for domain structure reference
    """
    from nenv.Preference import Preference

    domain = domain.lower()
    profile_letter = "A" if is_agent_a else "B"
    profile_path = f"domains/{domain}/profile{profile_letter}.json"

    try:
        return Preference(profile_path)
    except Exception as e:
        print(f"Error loading domain profile {profile_path}: {e}")
        return None


def recalculate_utilities_for_round(
    round_idx: int,
    weights_df: pd.DataFrame,
    bid_contents: List[str],
    opponent_who: str,
    session_df: pd.DataFrame,
    reference_pref,
    cache: dict,
    cache_lock: threading.Lock
) -> List[float]:
    """
    Recalculate utilities for all historical bids using weights from round_idx.

    Args:
        round_idx: The round whose weights to use for recalculation (index into opponent bids)
        weights_df: DataFrame containing opponent model weights per action (all actions, not just opponent)
        bid_contents: List of JSON bid strings from Session sheet
        opponent_who: 'A' or 'B' - which agent's bids to evaluate
        session_df: Full session DataFrame (contains 'Who' column)
        reference_pref: Preference object for domain structure
        cache: Shared cache dict {(round, bid_json): utility}
        cache_lock: Thread lock for cache access

    Returns:
        List of recalculated utilities for opponent's bids from rounds 0 to round_idx
    """
    import json
    from nenv.EditablePreference import EditablePreference
    from nenv.Bid import Bid

    # Map round_idx (opponent bid index) to the correct row in weights_df
    # weights_df has one row per action (both agents), but round_idx is an index into opponent bids only
    # We need to find which overall action row corresponds to the round_idx-th opponent bid
    opponent_mask = session_df['Who'] == opponent_who
    opponent_action_indices = session_df[opponent_mask].index.tolist()
    
    if round_idx >= len(opponent_action_indices):
        return []
    
    # The overall action index corresponding to the round_idx-th opponent bid
    overall_action_idx = opponent_action_indices[round_idx]
    
    # Use this to index into weights_df (weights_df rows align with session_df rows)
    if overall_action_idx >= len(weights_df):
        return []

    weights_row = weights_df.iloc[overall_action_idx]

    # Build issue_weights and issues dicts from weights_row
    issue_weights = {}
    issues = {}

    # Parse column names to extract structure
    # Format: "A_estimates_B_IssueName_issue_weight" or "A_estimates_B_IssueName_ValueName_weight"
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

    # Create EditablePreference with these weights
    try:
        estimated_pref = EditablePreference(
            issue_weights=issue_weights,
            issues=issues,
            reservation_value=0.0,
            generate_bids=False
        )
    except Exception as e:
        print(f"Error creating preference at round {round_idx}: {e}")
        return []

    # Recalculate utilities for opponent's bids
    # Reuse opponent_action_indices computed earlier (indices into session_df where opponent bid)
    utilities = []

    for idx in opponent_action_indices:  # Evaluate ALL opponent bids (past and future) with CURRENT weights
        bid_json = bid_contents[idx]

        # Check cache first
        cache_key = (round_idx, bid_json)

        with cache_lock:
            if cache_key in cache:
                utilities.append(cache[cache_key])
                continue

        # Parse bid
        try:
            bid_dict = json.loads(bid_json.replace("'", '"'))
            bid = Bid(bid_dict)
            utility = estimated_pref.get_utility(bid)

            # Cache result
            with cache_lock:
                cache[cache_key] = utility

            utilities.append(utility)
        except Exception as e:
            print(f"Error parsing/evaluating bid at index {idx}: {e}")
            utilities.append(0.0)  # Fallback

    return utilities


def load_all_sessions(base_dir: str, n_jobs: int = 32) -> Dict[str, List[Dict]]:
    """
    Load ALL session files ONCE across all domains, grouped by mode.
    
    Returns: {mode: [session_data, ...]}
    """
    all_data = {mode: [] for mode in ALL_MODES}
    domains = get_domains(base_dir)
    
    print(f"\nLoading ParetoWalker sessions from {len(domains)} domains...")
    
    for domain in domains:
        try:
            sessions_dir = Path(base_dir) / domain / "sessions"
            if not sessions_dir.exists():
                continue
            
            files = list(sessions_dir.iterdir())
            
            # Parallel loading
            session_data_list = Parallel(n_jobs=n_jobs)(
                delayed(load_session_data)(file) 
                for file in tqdm(files, desc=f"Loading {domain}")
            )
            
            # Group by mode
            for session_data in session_data_list:
                if session_data is None:
                    continue
                mode = session_data['mode']
                all_data[mode].append(session_data)
                
        except Exception as e:
            print(f"Error loading {domain}: {e}")
    
    # Print statistics
    print("\nLoaded sessions by mode:")
    for mode in ALL_MODES:
        total = len(all_data[mode])
        print(f"  {mode:20s}: {total:4d} sessions")
    
    return all_data


# ============================================================================
# Evaluation (runs in GPU processes)
# ============================================================================

def evaluate_session(session_data: Dict, forecaster, mode: str, deadline: int, min_round_threshold: int, eval_target: str) -> Optional[Dict]:
    """Evaluate forecaster on a single session with multithreaded utility recalculation."""
    try:
        perspective = get_paretowaker_perspective(session_data)
        if perspective is None:
            return None

        session_df = session_data['session_df'].copy()
        accept_row_idx = session_data['accept_row_idx']
        slice_end = min(accept_row_idx + 1, len(session_df))
        session_df = session_df.iloc[:slice_end]

        opponent_who = perspective['opponent_who']
        mask = session_df['Who'] == opponent_who
        session_df_opponent = session_df[mask].reset_index(drop=True)

        # Define sampling step (proportional to deadline: 1% sampling rate)
        sampling_step = max(1, deadline // 100)

        # Ensure enough rounds for at least one forecast
        if len(session_df_opponent) <= min_round_threshold + sampling_step:
            return None

        # Get actual utilities (always from session)
        actual_utilities = session_df_opponent[perspective['actual_column']].tolist()

        # Sample rounds
        valid_rounds = list(range(min_round_threshold, len(session_df_opponent) - sampling_step, sampling_step))

        if not valid_rounds:
            return None

        # Determine input utilities based on mode
        if mode == 'Oracle':
            # Oracle uses actual utilities (no recalculation needed)
            round_evaluations = []

            for round_idx in valid_rounds:
                input_data = []
                for i in range(round_idx + 1):
                    utility = actual_utilities[i]
                    timestamp = session_df_opponent.iloc[i]['Time'] if 'Time' in session_df_opponent.columns else i / deadline
                    input_data.append((utility, timestamp))

                forecast = forecaster(input_data)
                actual_after = actual_utilities[round_idx + 1:]
                comparison_length = min(len(forecast), len(actual_after))

                if comparison_length == 0:
                    continue

                forecast_slice = forecast[:comparison_length]
                actual_slice = np.array(actual_after[:comparison_length])

                # Oracle -> Oracle Metrics
                metrics = {'round': round_idx, 'comparison_length': comparison_length}

                try:
                    metrics['rmse'] = calculate_rmse(actual_slice, forecast_slice)
                except ValueError:
                    pass

                try:
                    metrics['mape'] = calculate_mape(actual_slice, forecast_slice)
                except ValueError:
                    pass

                try:
                    metrics['crps'] = calculate_crps(actual_slice, forecast_slice)
                except ValueError:
                    pass

                try:
                    # For MASE, use history as training series
                    training_series = np.array(actual_utilities[:round_idx + 1])
                    metrics['mase'] = calculate_mase(actual_slice, forecast_slice, training_series, seasonality=1)
                except ValueError:
                    pass

                # Only add if at least one metric was calculated
                if any(k in metrics for k in ['rmse', 'mape', 'crps', 'mase']):
                    round_evaluations.append(metrics)

        else:
            # NON-ORACLE: Recalculate utilities using weights

            # Load domain profile for structure
            domain = session_data['domain']
            reference_pref = get_domain_profile(domain, perspective['is_agent_a'])
            if reference_pref is None:
                return None

            # Get weights dataframe
            weights_df = session_data.get('weights_df')
            if weights_df is None:
                return None

            # Prepare for multithreaded recalculation
            bid_contents = session_data['bid_contents']

            # Shared cache and lock
            utility_cache = {}
            cache_lock = threading.Lock()

            # Multithreaded utility recalculation
            round_to_utilities = {}

            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_round = {
                    executor.submit(
                        recalculate_utilities_for_round,
                        round_idx,
                        weights_df,
                        bid_contents,
                        opponent_who,
                        session_df,
                        reference_pref,
                        utility_cache,
                        cache_lock
                    ): round_idx
                    for round_idx in valid_rounds
                }

                for future in as_completed(future_to_round):
                    round_idx = future_to_round[future]
                    try:
                        utilities = future.result()
                        round_to_utilities[round_idx] = utilities
                    except Exception as e:
                        print(f"Error recalculating utilities for round {round_idx}: {e}")

            # Now evaluate forecaster sequentially using recalculated utilities
            round_evaluations = []

            for round_idx in valid_rounds:
                if round_idx not in round_to_utilities:
                    continue

                input_utilities = round_to_utilities[round_idx]

                # Input data is HISTORY only
                input_utilities_history = input_utilities[:round_idx + 1]

                # Build input data for forecaster
                input_data = []
                for i, utility in enumerate(input_utilities_history):
                    timestamp = session_df_opponent.iloc[i]['Time'] if 'Time' in session_df_opponent.columns else i / deadline
                    input_data.append((utility, timestamp))

                # Forecast
                forecast = forecaster(input_data)
                
                round_metrics = {'round': round_idx}
                
                if eval_target == 'oracle':
                    # Target 1: Actual Future (Opponent -> Oracle)
                    actual_after = actual_utilities[round_idx + 1:]
                    comparison_length = min(len(forecast), len(actual_after))

                    if comparison_length > 0:
                        forecast_slice = forecast[:comparison_length]
                        actual_slice = np.array(actual_after[:comparison_length])
                        round_metrics['comparison_length'] = comparison_length

                        try:
                            round_metrics['rmse'] = calculate_rmse(actual_slice, forecast_slice)
                        except ValueError:
                            pass

                        try:
                            round_metrics['mape'] = calculate_mape(actual_slice, forecast_slice)
                        except ValueError:
                            pass

                        try:
                            round_metrics['crps'] = calculate_crps(actual_slice, forecast_slice)
                        except ValueError:
                            pass

                        try:
                            # For MASE, use actual history as training series
                            training_series = np.array(actual_utilities[:round_idx + 1])
                            round_metrics['mase'] = calculate_mase(actual_slice, forecast_slice, training_series, seasonality=1)
                        except ValueError:
                            pass

                elif eval_target == 'self':
                    # Target 2: Estimated Future (Opponent -> Opponent)
                    estimated_after = input_utilities[round_idx + 1:]
                    comparison_length = min(len(forecast), len(estimated_after))

                    if comparison_length > 0:
                        forecast_slice = forecast[:comparison_length]
                        estimated_slice = np.array(estimated_after[:comparison_length])
                        round_metrics['comparison_length'] = comparison_length

                        try:
                            round_metrics['rmse'] = calculate_rmse(estimated_slice, forecast_slice)
                        except ValueError:
                            pass

                        try:
                            round_metrics['mape'] = calculate_mape(estimated_slice, forecast_slice)
                        except ValueError:
                            pass

                        try:
                            round_metrics['crps'] = calculate_crps(estimated_slice, forecast_slice)
                        except ValueError:
                            pass

                        try:
                            # For MASE, use estimated history as training series
                            training_series = np.array(input_utilities[:round_idx + 1])
                            round_metrics['mase'] = calculate_mase(estimated_slice, forecast_slice, training_series, seasonality=1)
                        except ValueError:
                            pass

                # Add if at least one metric was calculated
                if any(k in round_metrics for k in ['rmse', 'mape', 'crps', 'mase']):
                    round_evaluations.append(round_metrics)
        
        if not round_evaluations:
            return None
        
        return {
            'domain': session_data['domain'],
            'opponent': perspective['opponent_name'],
            'mode': mode,
            'round_evaluations': round_evaluations
        }
    
    except Exception as e:
        print(f"Error evaluating session {session_data.get('filename', 'unknown')}: {e}")
        return None


def process_mode(args: Tuple[str, int, List[Dict], int, str]) -> Tuple[str, List[Dict]]:
    """
    Process all sessions for a single mode on a dedicated GPU.

    Args:
        args: (mode, gpu_id, sessions_list, deadline, eval_target)

    Returns:
        (mode, list of session results)
    """
    mode, gpu_id, sessions, deadline, eval_target = args
    
    # Set GPU for this process BEFORE importing torch/forecaster
    os.environ['WORKER_GPU_ID'] = str(gpu_id)
    
    # Import forecaster after setting GPU
    from agents.NegoformerAgent.forecasting.forecaster import Forecaster
    
    print(f"[GPU {gpu_id}] Starting {mode} mode with {len(sessions)} sessions...")
    
    # Initialize forecaster on assigned GPU
    forecaster = Forecaster(
        max_context=deadline // 2,
        prediction_length=deadline // 4,
        deadline=deadline
    )
    
    min_round_threshold = int(deadline * 0.05)
    
    if not sessions:
        return mode, []
    
    # Evaluate sessions
    results = []
    for session_data in tqdm(sessions, desc=f"[GPU {gpu_id}] {mode}"):
        result = evaluate_session(session_data, forecaster, mode, deadline, min_round_threshold, eval_target)
        if result is not None:
            results.append(result)
    
    print(f"[GPU {gpu_id}] {mode}: {len(results)} sessions evaluated successfully")
    return mode, results


def aggregate_by_round(all_mode_results: Dict[str, List[Dict]]) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """Aggregate results by round for all modes and metrics."""
    round_data = {}

    for mode, session_results in all_mode_results.items():
        round_data[mode] = {}

        for session in session_results:
            for round_eval in session['round_evaluations']:
                round_num = round_eval['round']
                if round_num not in round_data[mode]:
                    round_data[mode][round_num] = {}
                
                # Collect all metrics present in the evaluation
                for key, value in round_eval.items():
                    if key in ['round', 'comparison_length']:
                        continue
                        
                    if key not in round_data[mode][round_num]:
                        round_data[mode][round_num][key] = []
                    
                    round_data[mode][round_num][key].append(value)

    return round_data


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Forecaster Ablation Evaluation")
    parser.add_argument('--sessions_path', type=str,
                        default="/Users/emrekuru/Developer/Thesis/Negoformer/strategy/results/oracle/1000",
                        help="Path to sessions directory")
    parser.add_argument('--deadline', type=int, default=1000, help="Negotiation deadline")
    parser.add_argument('--output', type=str, default=None,
                        help="Output pickle file path (default: forecaster_ablation_results.pkl)")
    parser.add_argument('--num_gpus', type=int, default=8, help="Number of GPUs to use")
    parser.add_argument('--target', type=str, required=True, 
                        choices=['oracle_baseline', 'oracle_target', 'self_target'],
                        help="Evaluation target mode:\n"
                             "  oracle_baseline: Run Oracle mode only (Oracle -> Oracle)\n"
                             "  oracle_target: Run all Opponent Models -> Target is Oracle (Actual)\n"
                             "  self_target: Run all Opponent Models -> Target is Self (Estimated)")
    args = parser.parse_args()

    # Determine selected modes and evaluation type based on target
    if args.target == 'oracle_baseline':
        selected_modes = ['Oracle']
        eval_target = 'oracle' # Evaluate vs Actual
    elif args.target == 'oracle_target':
        selected_modes = [m for m in ALL_MODES if m != 'Oracle']
        eval_target = 'oracle' # Evaluate vs Actual
    elif args.target == 'self_target':
        selected_modes = [m for m in ALL_MODES if m != 'Oracle']
        eval_target = 'self'   # Evaluate vs Estimated

    # Set default output filename based on target
    if args.output is None:
        args.output = f"forecaster_ablation_{args.target}.pkl"
    
    print("="*80)
    print("Multi-GPU Forecaster Ablation Evaluation")
    print("="*80)
    print(f"Sessions path: {args.sessions_path}")
    print(f"Deadline: {args.deadline}")
    print(f"Target Mode: {args.target}")
    print(f"Modes Running: {selected_modes}")
    print(f"Metrics: RMSE, MAPE")
    print(f"Output: {args.output}")
    
    # Auto-detect available GPUs
    import torch
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {available_gpus}")
    else:
        available_gpus = 0
        print("No GPUs available, using CPU")
    
    num_gpus = min(args.num_gpus, max(available_gpus, 1))
    print(f"Using {num_gpus} GPU(s)")
    print("="*80)
    
    # =========================================================================
    # LOAD ALL SESSIONS ONCE (before forking to GPU processes)
    # =========================================================================
    all_sessions = load_all_sessions(args.sessions_path, n_jobs=32)
    
    # =========================================================================
    # Distribute sessions to GPU processes
    # =========================================================================
    mode_args = [
        (mode, i % num_gpus, all_sessions[mode], args.deadline, eval_target)
        for i, mode in enumerate(selected_modes)
    ]
    
    print(f"\nProcessing {len(selected_modes)} modes across {num_gpus} GPU(s)...")
    
    with Pool(processes=min(len(selected_modes), num_gpus)) as pool:
        results = pool.map(process_mode, mode_args)
    
    # Collect results
    all_mode_results = {mode: result_list for mode, result_list in results}

    # Aggregate by round
    round_by_round_data = aggregate_by_round(all_mode_results)
    
    # Print summary
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    for mode in selected_modes:
        if mode in all_mode_results:
            num_sessions = len(all_mode_results[mode])
            num_rounds = len(round_by_round_data.get(mode, {}))
            print(f"  {mode:20s}: {num_sessions:4d} sessions, {num_rounds:4d} unique rounds")
    
    # Save results
    output_data = {
        'all_mode_results': all_mode_results,
        'round_by_round_data': round_by_round_data,
        'config': {
            'sessions_path': args.sessions_path,
            'deadline': args.deadline,
            'target': args.target,
            'eval_target': eval_target,
            'modes': selected_modes,
        }
    }
    
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    main()
