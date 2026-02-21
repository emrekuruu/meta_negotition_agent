import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ablations.forecaster.data_preparation import (
    sessions_to_panel_dataset
)
from ablations.forecaster.zero_shot_evaluation import (
    load_all_sessions
)

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, LSTM


def train_global_models(
    sessions: Dict[str, List],
    modes: List[str],
    deadline: int,
    output_dir: Path,
    max_steps: int = 1000,
    gpu_id: int = 0
):
    """
    Train global PatchTST and LSTM models for each mode on ALL available data.

    Args:
        sessions: Dictionary of sessions by mode
        modes: List of modes to train
        deadline: Negotiation deadline for context/prediction length calculation
        output_dir: Directory to save trained models
        max_steps: Maximum training iterations
        gpu_id: GPU device ID to use

    Raises:
        ValueError: If any mode has no training data or if training fails
    """
    # Set GPU environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Calculate model parameters based on deadline
    h = deadline // 4  # prediction_length (250 for deadline=1000)
    input_size = deadline // 4  # max_context (250 for deadline=1000) 

    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Deadline: {deadline}")
    print(f"  Prediction Length (h): {h}")
    print(f"  Input Size (context): {input_size}")
    print(f"  Max Training Steps: {max_steps}")
    print(f"  GPU: {gpu_id}")
    print(f"{'='*70}\n")

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Training models for mode: {mode}")
        print(f"{'='*70}")

        # Validate mode has data
        if mode not in sessions or len(sessions[mode]) == 0:
            raise ValueError(f"No sessions found for mode {mode}")

        # Convert sessions to panel format
        print(f"\nConverting {len(sessions[mode])} sessions to panel format...")
        train_df = sessions_to_panel_dataset(sessions[mode], mode, deadline, min_length=h)
        print(f"Training data: {len(train_df)} rows, {train_df['unique_id'].nunique()} unique sessions")

        # Initialize models with EXPLICIT parameters (no defaults)
        print("\nInitializing models...")
        models = [
            PatchTST(
                h=h,
                input_size=input_size,
                encoder_layers=3,
                n_heads=16,
                hidden_size=128,
                patch_len=16,
                stride=8,
                dropout=0.2,
                fc_dropout=0.2,
                max_steps=max_steps,
                learning_rate=1e-3,
                scaler_type='standard',  # Critical for [0,1] utility values
                start_padding_enabled=True,  # Allow short series to be padded
                batch_size=32,  # Reduce batch size to manage memory
                random_seed=42
            ),
            LSTM(
                h=h,
                input_size=input_size,
                encoder_n_layers=2,
                encoder_hidden_size=128,
                encoder_dropout=0.1,
                max_steps=max_steps,
                learning_rate=1e-3,
                scaler_type='standard',
                start_padding_enabled=True,  # Allow short series to be padded
                batch_size=32,  # Reduce batch size to manage memory
                random_seed=42
            )
        ]

        print("Models initialized:")
        print("  - PatchTST (Transformer with patching)")
        print("  - LSTM (Encoder-Decoder)")

        # Train models
        print(f"\nTraining models (max_steps={max_steps})...")
        nf = NeuralForecast(models=models, freq='1s')

        # Train on ALL data
        try:
            nf.fit(df=train_df, val_size=0)
        except Exception as e:
            raise ValueError(f"Training failed for mode {mode}: {str(e)}")

        # Save trained models
        save_path = output_dir / mode
        save_path.mkdir(parents=True, exist_ok=True)
        nf.save(path=str(save_path), overwrite=True)
        print(f"\n✓ Models saved to: {save_path}")

        # Verify saved models can be loaded
        print("Verifying saved models...")
        try:
            test_nf = NeuralForecast.load(path=str(save_path))
            print("✓ Models successfully loaded from checkpoint")
        except Exception as e:
            raise ValueError(f"Failed to load saved models for {mode}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train NeuralForecast baseline models (PatchTST and LSTM)"
    )
    parser.add_argument(
        '--sessions_path',
        type=str,
        required=True,
        help='Path to session files (e.g., results/oracle/1000)'
    )
    parser.add_argument(
        '--deadline',
        type=int,
        required=True,
        help='Negotiation deadline (e.g., 1000)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save trained models (e.g., models/neuralforecast)'
    )

    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='Maximum training iterations (default: 1000)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Import pandas here to avoid import before argparse
    import pandas as pd
    globals()['pd'] = pd

    # Validate sessions path exists
    sessions_path = Path(args.sessions_path)
    if not sessions_path.exists():
        raise ValueError(f"Sessions path does not exist: {sessions_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine modes to train
    modes_to_train = ['Oracle']

    print(f"{'='*70}")
    print(f"NeuralForecast Baseline Training (Oracle Only)")
    print(f"{'='*70}")
    print(f"Sessions Path: {sessions_path}")
    print(f"Deadline: {args.deadline}")
    print(f"Output Directory: {output_dir}")
    print(f"Modes: {', '.join(modes_to_train)}")
    print(f"Split: Training on ALL sessions (no holdout)")
    print(f"Random Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Load all sessions
    print("Loading sessions...")
    all_sessions = load_all_sessions(str(sessions_path), n_jobs=32)

    # Filter to requested modes
    all_sessions = {mode: sessions for mode, sessions in all_sessions.items()
                    if mode in modes_to_train}

    # Train models for each mode on ALL sessions
    train_global_models(
        sessions=all_sessions,
        modes=modes_to_train,
        deadline=args.deadline,
        output_dir=output_dir,
        max_steps=args.max_steps,
        gpu_id=args.gpu
    )

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
