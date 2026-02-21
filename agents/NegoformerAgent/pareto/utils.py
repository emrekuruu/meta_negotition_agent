from typing import List, Tuple
from numba import prange, njit
import warnings

warnings.filterwarnings("ignore")


@njit(parallel=True)
def extract_pareto_indices(candidates: List[Tuple[float, float]], lower_bound: float) -> List[int]:
    """
    This method extracts the pareto points from a list of bids.
    
    Args:
        candidates: List of bids as utility tuple (U_A, U_B).
        lower_bound: Utility lower bound
        
    Returns:
        List of indices of pareto bids.
    """
    indices = []

    for i in range(len(candidates)):
        is_pareto = True
        candidate_i = candidates[i]

        for j in prange(len(candidates)):
            if not is_pareto:
                break

            candidate_j = candidates[j]

            if candidate_i[0] > candidate_j[0]:
                break

            if (candidate_j[0] >= candidate_i[0] and candidate_j[1] >= candidate_i[1]) and (
                    candidate_j[0] > candidate_i[0] or candidate_j[1] > candidate_i[1]):
                is_pareto = False
                break

        if is_pareto and candidate_i[0] >= lower_bound:
            indices.append(i)

    return indices
