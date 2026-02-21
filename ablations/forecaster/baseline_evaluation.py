import os
import sys
import argparse
import pickle
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ablations.forecaster.data_preparation import (
    get_paretowaker_perspective
)
from ablations.forecaster.zero_shot_evaluation import (
    load_all_sessions,
    aggregate_by_round
)
from ablations.forecaster.metrics import calculate_rmse, calculate_mape, calculate_crps, calculate_mase

from neuralforecast import NeuralForecast


def build_history_dataframe(
    session_data: Dict,
    round_idx: int,
    mode: str,
    deadline: int,
    custom_unique_id: str = None
) -> Optional[pd.DataFrame]:
    """
    Build NeuralForecast input DataFrame for history up to round_idx.

    Args:
        session_data: Session dictionary
        round_idx: Round index to predict from (history = 0 to round_idx inclusive)
        mode: Mode name
        deadline: Negotiation deadline
        custom_unique_id: Optional custom unique_id for batching

    Returns:
        DataFrame with [unique_id, ds, y] for history or None if invalid
    """
    # Get perspective
    perspective = get_paretowaker_perspective(session_data)
    if perspective is None:
        return None

    # Get session dataframe
    session_df = session_data['session_df'].copy()
    accept_row_idx = session_data.get('accept_row_idx')

    if accept_row_idx is not None:
        slice_end = min(accept_row_idx + 1, len(session_df))
        session_df = session_df.iloc[:slice_end]

    # Filter to opponent actions only
    opponent_who = perspective['opponent_who']
    mask = session_df['Who'] == opponent_who
    session_df_opponent = session_df[mask].reset_index(drop=True)

    if round_idx + 1 > len(session_df_opponent):
        return None

    # Extract utilities up to round_idx (inclusive)
    actual_column = perspective['actual_column']
    utilities = session_df_opponent.iloc[:round_idx + 1][actual_column].values

    if len(utilities) == 0:
        return None

    # Check for NaN
    if np.any(np.isnan(utilities)):
        raise ValueError(
            f"NaN values in history for session {session_data['filename']} at round {round_idx}"
        )

    # Generate unique_id for this test session
    if custom_unique_id:
        unique_id = custom_unique_id
    else:
        domain = session_data['domain']
        filename = session_data['filename']
        session_hash = abs(hash(filename)) % (10 ** 8)
        unique_id = f"{domain}_{mode}_{session_hash}"

    # Create regular timestamps
    timestamps = pd.date_range(
        start='2024-01-01',
        periods=len(utilities),
        freq='1s'
    )

    # Create panel format DataFrame
    history_df = pd.DataFrame({
        'unique_id': [unique_id] * len(utilities),
        'ds': timestamps,
        'y': utilities
    })

    return history_df


def evaluate_session_neuralforecast(
    session_data: Dict,
    nf: NeuralForecast,
    model_name: str,
    mode: str,
    deadline: int,
    min_round_threshold: int
) -> Optional[Dict]:
    """
    Evaluate NeuralForecast model on a single session with per-round sampling.

    Args:
        session_data: Session dictionary
        nf: Loaded NeuralForecast instance with trained models
        model_name: Model name ('PatchTST' or 'LSTM')
        mode: Mode name
        deadline: Negotiation deadline
        min_round_threshold: Minimum round to start evaluation

    Returns:
        Dictionary with evaluation results or None if session invalid
    """
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

        # Define sampling step
        sampling_step = max(1, deadline // 100)

        # Ensure enough rounds
        if len(session_df_opponent) <= min_round_threshold + sampling_step:
            return None

        # Get actual utilities
        actual_utilities = session_df_opponent[perspective['actual_column']].tolist()

        # Sample rounds
        valid_rounds = list(range(min_round_threshold, len(session_df_opponent) - sampling_step, sampling_step))

        if not valid_rounds:
            return None

        # --- BATCH OPTIMIZATION START ---
        # Instead of calling predict() in a loop, we create a batch of histories
        batch_dfs = []
        round_map = {} # Map unique_id back to round_idx

        domain = session_data['domain']
        filename = session_data['filename']
        session_hash = abs(hash(filename)) % (10 ** 8)

        for round_idx in valid_rounds:
            # Create a unique ID for this round's history
            # Format: {domain}_{mode}_{hash}_r{round}
            uid = f"{domain}_{mode}_{session_hash}_r{round_idx}"
            
            # Build history DataFrame
            history_df = build_history_dataframe(
                session_data, 
                round_idx, 
                mode, 
                deadline, 
                custom_unique_id=uid
            )
            
            if history_df is not None:
                batch_dfs.append(history_df)
                round_map[uid] = round_idx

        if not batch_dfs:
            return None

        # Combine all rounds into one large panel dataset
        full_batch_df = pd.concat(batch_dfs, ignore_index=True)

        # Execute Single Prediction for the whole session
        try:
            forecast_df = nf.predict(df=full_batch_df, verbose=False)
        except Exception as e:
            print(f"Batch prediction error for session {filename}: {e}")
            return None
        
        # Handle unique_id being in index or column
        if 'unique_id' not in forecast_df.columns and forecast_df.index.name == 'unique_id':
             forecast_df = forecast_df.reset_index()

        # Check model name exists
        if model_name not in forecast_df.columns:
             raise ValueError(
                f"Model '{model_name}' not found in forecast columns: {list(forecast_df.columns)}"
            )

        # Process results from batch
        round_evaluations = []
        
        for uid, group in forecast_df.groupby('unique_id'):
            if uid not in round_map:
                continue
                
            round_idx = round_map[uid]
            forecast = group[model_name].values

            # Get actual future values
            actual_after = actual_utilities[round_idx + 1:]
            comparison_length = min(len(forecast), len(actual_after))

            if comparison_length == 0:
                continue

            # Calculate metrics
            forecast_slice = forecast[:comparison_length]
            actual_slice = np.array(actual_after[:comparison_length])

            metrics = {
                'round': round_idx,
                'comparison_length': comparison_length
            }

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
                # For MASE, use actual history as training series
                training_series = np.array(actual_utilities[:round_idx + 1])
                metrics['mase'] = calculate_mase(actual_slice, forecast_slice, training_series, seasonality=1)
            except ValueError:
                pass

            # Add if at least one metric was calculated
            if any(k in metrics for k in ['rmse', 'mape', 'crps', 'mase']):
                round_evaluations.append(metrics)
        # --- BATCH OPTIMIZATION END ---

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



def process_mode_neuralforecast(
    args: Tuple[str, int, List[Dict], int, Path, str]
) -> Tuple[str, List[Dict]]:
    """
    Process all test sessions for a single mode with trained NeuralForecast model.

    Args:
        args: (mode, gpu_id, sessions_list, deadline, model_path, model_name)

    Returns:
        (mode_model_name, list of session results)
    """
    mode, gpu_id, sessions, deadline, model_path, model_name = args

    # Set GPU for this process
    os.environ['WORKER_GPU_ID'] = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Silence PyTorch Lightning and NeuralForecast
    import logging
    import warnings
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    os.environ["rdma_prediction_backend"] = "0"  # Suppress some potential distributed warnings

    print(f"[GPU {gpu_id}] Loading {model_name} model for {mode}...")

    # Load trained NeuralForecast model
    try:
        nf = NeuralForecast.load(path=str(model_path))
    except Exception as e:
        print(f"[GPU {gpu_id}] Error loading model from {model_path}: {e}")
        return f"{mode}_{model_name}", []

    print(f"[GPU {gpu_id}] Starting {mode} - {model_name} with {len(sessions)} test sessions...")

    min_round_threshold = int(deadline * 0.05)

    if not sessions:
        return f"{mode}_{model_name}", []

    # Evaluate sessions
    results = []
    # Disable tqdm in inner loop to key logs clean? Or keep it?
    # User complained about "Predicting DataLoader" logs which come from PL.
    # The tqdm here is for SESSIONS, which is useful.
    for session_data in tqdm(sessions, desc=f"[GPU {gpu_id}] {mode} - {model_name}", position=gpu_id):
        result = evaluate_session_neuralforecast(
            session_data, nf, model_name, mode, deadline, min_round_threshold
        )
        if result is not None:
            results.append(result)

    print(f"[GPU {gpu_id}] {mode} - {model_name}: {len(results)} sessions evaluated successfully")
    return f"{mode}_{model_name}", results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained NeuralForecast baseline models"
    )
    parser.add_argument(
        '--sessions_path',
        type=str,
        required=True,
        help="Path to session files (e.g., results/oracle/1000)"
    )
    parser.add_argument(
        '--deadline',
        type=int,
        required=True,
        help="Negotiation deadline (e.g., 1000)"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Directory containing trained models (e.g., models/neuralforecast)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output pickle file path (default: forecaster_neuralforecast_oracle_target.pkl)"
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Validate paths
    sessions_path = Path(args.sessions_path)
    if not sessions_path.exists():
        raise ValueError(f"Sessions path does not exist: {sessions_path}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    # Determine modes to evaluate
    selected_modes = ['Oracle']

    # Set default output filename
    if args.output is None:
        args.output = f"forecaster_neuralforecast_oracle_target.pkl"

    print("=" * 80)
    print("NeuralForecast Baseline Evaluation (Oracle Only)")
    print("=" * 80)
    print(f"Sessions Path: {sessions_path}")
    print(f"Model Directory: {model_dir}")
    print(f"Deadline: {args.deadline}")
    print(f"Modes: {', '.join(selected_modes)}")
    print(f"Models: PatchTST, LSTM")
    print(f"Output: {args.output}")
    print(f"Random Seed: {args.seed}")
    print("=" * 80)

    # Auto-detect GPUs
    import torch
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {available_gpus}")
    else:
        available_gpus = 0
        print("No GPUs available, using CPU")

    num_gpus = min(args.num_gpus, max(available_gpus, 1))
    print(f"Using {num_gpus} GPU(s)")
    print("=" * 80)

    # Load all sessions
    print("\nLoading all sessions...")
    all_sessions = load_all_sessions(str(sessions_path), n_jobs=32)

    # Use all sessions for evaluation (no splitting)
    # Filter to selected modes
    test_sessions = {mode: sessions for mode, sessions in all_sessions.items()
                     if mode in selected_modes}

    # Create evaluation tasks for each (mode, model) pair
    models_to_eval = ['PatchTST', 'LSTM']
    eval_tasks = []
    gpu_counter = 0

    for mode in selected_modes:
        mode_model_path = model_dir / mode
        if not mode_model_path.exists():
            print(f"Warning: No trained models found for {mode} at {mode_model_path}")
            continue

        for model_name in models_to_eval:
            eval_tasks.append((
                mode,
                gpu_counter % num_gpus,
                test_sessions[mode],
                args.deadline,
                mode_model_path,
                model_name
            ))
            gpu_counter += 1

    print(f"\nProcessing {len(eval_tasks)} (mode, model) pairs across {num_gpus} GPU(s)...")
    print(f"  {len(selected_modes)} modes × {len(models_to_eval)} models = {len(eval_tasks)} tasks")

    # Process in parallel
    with Pool(processes=min(len(eval_tasks), num_gpus * 2)) as pool:
        results = pool.map(process_mode_neuralforecast, eval_tasks)

    # Collect results
    all_mode_results = {mode_model: result_list for mode_model, result_list in results}

    # Aggregate by round
    round_by_round_data = aggregate_by_round(all_mode_results)

    # Save results
    output_data = {
        'round_by_round': round_by_round_data,
        'all_results': all_mode_results,
        'config': {
            'deadline': args.deadline,
            'mode': 'Oracle',
            'models': models_to_eval,
            'seed': args.seed
        }
    }

    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n{'=' * 80}")
    print("Evaluation Complete!")
    print(f"{'=' * 80}")
    print(f"Results saved to: {args.output}")
    print(f"\nSummary:")
    for mode_model, results in all_mode_results.items():
        print(f"  {mode_model:30s}: {len(results):4d} sessions evaluated")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
