import random
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def get_paretowaker_perspective(session_data: Dict) -> Optional[Dict]:
    """
    Determine ParetoWalker perspective and opponent information.

    Args:
        session_data: Session dictionary with agent_a, agent_b keys

    Returns:
        Dictionary with opponent information or None if ParetoWalker not found
    """
    agent_a = session_data['agent_a']
    agent_b = session_data['agent_b']
    if 'ParetoWalker' in agent_a:
        return {
            'opponent_name': agent_b,
            'is_agent_a': True,
            'opponent_who': 'B',
            'actual_column': 'AgentBUtility'
        }
    elif 'ParetoWalker' in agent_b:
        return {
            'opponent_name': agent_a,
            'is_agent_a': False,
            'opponent_who': 'A',
            'actual_column': 'AgentAUtility'
        }
    return None


def session_to_panel_row(
    session_data: Dict,
    mode: str,
    deadline: int,
    min_length: int = 0
) -> Optional[pd.DataFrame]:
    """
    Convert single session to NeuralForecast panel format.

    Args:
        session_data: Session dictionary from load_session_data
        mode: Mode name (e.g., 'Oracle', 'Bayesian')
        deadline: Negotiation deadline for timestamp normalization
        min_length: Minimum required session length (default: 0)

    Returns:
        DataFrame with columns [unique_id, ds, y] or None if session invalid
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

    if len(session_df_opponent) == 0:
        return None

    # Extract opponent utilities (actual/oracle utilities)
    actual_column = perspective['actual_column']
    if actual_column not in session_df_opponent.columns:
        raise ValueError(
            f"Column '{actual_column}' not found in session {session_data['filename']}"
        )

    utilities = session_df_opponent[actual_column].values

    # Check minimum length
    if len(utilities) < min_length:
        return None

    # Check for NaN values (NO fallbacks!)
    if np.any(np.isnan(utilities)):
        raise ValueError(
            f"NaN values found in utilities for session {session_data['filename']}"
        )

    # Generate unique_id for this session
    domain = session_data['domain']
    filename = session_data['filename']
    # Use hash of filename for uniqueness
    session_hash = abs(hash(filename)) % (10 ** 8)
    unique_id = f"{domain}_{mode}_{session_hash}"

    # Create regular timestamps with 1-second intervals
    # NeuralForecast requires inferrable frequency
    timestamps = pd.date_range(
        start='2024-01-01',
        periods=len(utilities),
        freq='1s'
    )

    # Create panel format DataFrame
    panel_df = pd.DataFrame({
        'unique_id': [unique_id] * len(utilities),
        'ds': timestamps,
        'y': utilities
    })

    return panel_df


def sessions_to_panel_dataset(
    sessions: List[Dict],
    mode: str,
    deadline: int,
    min_length: int = 0,
    n_jobs: int = 32
) -> pd.DataFrame:
    """
    Convert list of sessions to full NeuralForecast panel dataset.

    Args:
        sessions: List of session dictionaries
        mode: Mode name (e.g., 'Oracle', 'Bayesian')
        deadline: Negotiation deadline
        min_length: Minimum required session length (default: 0)
        n_jobs: Number of parallel jobs for processing

    Returns:
        Concatenated DataFrame with all sessions in panel format

    Raises:
        ValueError: If any session contains NaN values or if result is empty
    """
    # Process sessions in parallel
    panel_dfs = Parallel(n_jobs=n_jobs)(
        delayed(session_to_panel_row)(session, mode, deadline, min_length)
        for session in sessions
    )

    # Filter out None results
    panel_dfs = [df for df in panel_dfs if df is not None]

    if len(panel_dfs) == 0:
        raise ValueError(f"No valid sessions found for mode {mode}")

    # Concatenate all DataFrames
    full_df = pd.concat(panel_dfs, ignore_index=True)

    # Sort by unique_id and timestamp
    full_df = full_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    # Final validation: check for NaN
    if full_df['y'].isnull().any():
        raise ValueError(
            f"NaN values found in final dataset for mode {mode}"
        )

    return full_df

