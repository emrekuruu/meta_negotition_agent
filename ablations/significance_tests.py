import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, levene, ks_2samp
from statsmodels.stats.anova import AnovaRM
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_best_model_per_metric(ratings_df: pd.DataFrame) -> Dict[str, str]:
    best_models = {}
    for metric in ratings_df.columns:
        # Assumes values are negated if lower is better (so high is always good here)
        mean_scores = ratings_df[metric].apply(lambda x: np.mean(np.array(x)))
        best_models[metric] = mean_scores.idxmax()
    return best_models

def prepare_long_format_data(ratings_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    long_data_list = []
    # Identify number of subjects from first available model
    first_model = ratings_df.index[0]
    n_subjects = len(ratings_df.loc[first_model, metric])
    
    for subject_idx in range(n_subjects):
        for model_name in ratings_df.index:
            value = ratings_df.loc[model_name, metric][subject_idx]
            long_data_list.append({
                'Subject': subject_idx,
                'Model': model_name,
                'Value': value
            })
    return pd.DataFrame(long_data_list)

def test_normality(data: np.ndarray) -> bool:
    if len(data) < 3: return False 
    # KS test against normal distribution with same mean/std
    _, p_value = ks_2samp(data, np.random.normal(np.mean(data), np.std(data), len(data)))
    return p_value >= 0.05

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0: return 0.0
    d = (np.mean(group1) - np.mean(group2)) / pooled_sd
    return d

def perform_group_significance_test(ratings_df: pd.DataFrame, metric: str) -> Tuple[str, float]:
    methods = list(ratings_df.index)
    
    # Check normality
    normal_tests = []
    for method in methods:
        data = np.array(ratings_df.loc[method, metric])
        is_normal = test_normality(data)
        normal_tests.append(is_normal)

    if all(normal_tests):
        try:
            long_data = prepare_long_format_data(ratings_df, metric)
            anova = AnovaRM(long_data, 'Value', 'Subject', within=['Model']).fit()
            return "Repeated Measures ANOVA", anova.anova_table['Pr > F'][0]
        except Exception as e:
            logger.warning(f"RM ANOVA failed: {e}, falling back to Friedman")

    # Fallback to Friedman
    friedman_data = [np.array(ratings_df.loc[m, metric]) for m in methods]
    stat, p_value = friedmanchisquare(*friedman_data)
    return "Friedman Test", p_value

def perform_pairwise_comparison(best_scores: List[float], comparison_scores: List[float]) -> Tuple[str, float, float]:
    best_array = np.array(best_scores)
    comp_array = np.array(comparison_scores)
    
    effect_size = calculate_cohens_d(best_array, comp_array)
    
    norm1 = test_normality(best_array)
    norm2 = test_normality(comp_array)
    
    if norm1 and norm2:
        _, lev_p = levene(best_array, comp_array)
        if lev_p >= 0.05:
            _, p_value = ttest_rel(best_array, comp_array)
            return "Paired t-test", p_value, effect_size
            
    _, p_value = wilcoxon(best_array, comp_array)
    return "Wilcoxon Signed-Rank Test", p_value, effect_size

def interpret_effect_size(effect_size: float) -> str:
    if effect_size is None: return "N/A"
    abs_e = abs(effect_size)
    if abs_e < 0.2: return "negligible"
    elif abs_e < 0.5: return "small"
    elif abs_e < 0.8: return "medium"
    else: return "large"

# --- Main Iterative Logic (Per Metric) ---

def iterative_compare_models_per_metric(ratings_df: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    
    for metric in ratings_df.columns:
        # Start with full set of models for this metric
        current_models = list(ratings_df.index)
        iteration = 1
        
        while len(current_models) > 1:
            print(f"Metric: {metric}, Iteration: {iteration}, Models: {len(current_models)}")
            
            # Subset data
            current_data = ratings_df.loc[current_models, [metric]]
            
            # Find best model
            # get_best_model_per_metric expects a DF, returns {metric: best_model}
            best_model_name = get_best_model_per_metric(current_data)[metric]
            
            # Group Test Logic
            if len(current_models) > 2:
                group_test_name, group_p = perform_group_significance_test(current_data, metric)
            else:
                group_test_name = "Skipped (Only 2 models)"
                group_p = 0.0 # Force pairwise comparison 
            
            # Pairwise comparisons
            # Determine if we should proceed with pairwise comparisons
            # We proceed if group test is significant OR if we skipped it because only 2 models left
            is_significant_group = (group_p < 0.05)
            
            if is_significant_group:
                for model in current_models:
                    if model == best_model_name:
                        continue
                    
                    try:
                        test_used, p_val, effect_size = perform_pairwise_comparison(
                            ratings_df.loc[best_model_name, metric],
                            ratings_df.loc[model, metric]
                        )
                        effect_interp = interpret_effect_size(effect_size)
                        
                        all_results.append({
                            "Metric": metric,
                            "Iteration": iteration,
                            "Group Test": group_test_name,
                            "Group p": group_p,
                            "Best Method": best_model_name,
                            "Compared Method": model,
                            "Pair Test": test_used,
                            "Pair p": p_val,
                            "Significant": p_val < 0.05,
                            "Effect Size": effect_size,
                            "Interpretation": effect_interp
                        })
                    except Exception as e:
                        logger.error(f"Error comparing {best_model_name} vs {model}: {e}")
            else:
                 # No significant difference among group, record best model info anyway
                 all_results.append({
                    "Metric": metric,
                    "Iteration": iteration,
                    "Group Test": group_test_name,
                    "Group p": group_p,
                    "Best Method": best_model_name,
                    "Compared Method": None,
                    "Pair Test": None,
                    "Pair p": None,
                    "Significant": False,
                    "Effect Size": None,
                    "Interpretation": None
                })
            
            # Remove best model from THIS metric's pool
            current_models.remove(best_model_name)
            iteration += 1
            
    return pd.DataFrame(all_results)


# --- Data Loading & Prep ---

def merge_round_data(data_dict, sources):
    merged = {}
    for source_key in sources:
        if source_key in data_dict:
            if 'round_by_round_data' in data_dict[source_key]:
                rd = data_dict[source_key]['round_by_round_data']
                for mode, mode_data in rd.items():
                    merged[mode] = mode_data
    return merged

def clean_mape_data(round_data, threshold=200):
    cleaned_count = 0
    for mode, mode_rounds in round_data.items():
        for round_num, metrics in mode_rounds.items():
            if 'mape' in metrics:
                original_len = len(metrics['mape'])
                metrics['mape'] = [v for v in metrics['mape'] if v is None or np.isnan(v) or v <= threshold]
                cleaned_count += original_len - len(metrics['mape'])
    return cleaned_count

def prepare_data(round_data, end_of_session_only):
    data_rmse = {}
    data_mape = {}
    
    for mode, mode_rounds in round_data.items():
        if end_of_session_only:
            if mode_rounds:
                max_round = max(mode_rounds.keys())
                if max_round in mode_rounds:
                    if 'rmse' in mode_rounds[max_round]:
                        vals = mode_rounds[max_round]['rmse']
                        vals = [v for v in vals if v is not None and not np.isnan(v)]
                        if vals: data_rmse[mode] = vals
                    if 'mape' in mode_rounds[max_round]:
                        vals = mode_rounds[max_round]['mape']
                        vals = [v for v in vals if v is not None and not np.isnan(v)]
                        if vals: data_mape[mode] = vals
        else:
            all_rmse = []
            all_mape = []
            for round_num in mode_rounds.keys():
                if 'rmse' in mode_rounds[round_num]:
                    vals = mode_rounds[round_num]['rmse']
                    vals = [v for v in vals if v is not None and not np.isnan(v)]
                    all_rmse.extend(vals)
                if 'mape' in mode_rounds[round_num]:
                    vals = mode_rounds[round_num]['mape']
                    vals = [v for v in vals if v is not None and not np.isnan(v)]
                    all_mape.extend(vals)
            if all_rmse: data_rmse[mode] = all_rmse
            if all_mape: data_mape[mode] = all_mape
            
    if not data_rmse and not data_mape:
        return None
        
    # Intersection for RMSE
    if data_rmse:
        min_length_rmse = min(len(v) for v in data_rmse.values())
        for mode in data_rmse:
            data_rmse[mode] = data_rmse[mode][:min_length_rmse]
            
    # Intersection for MAPE
    if data_mape:
        min_length_mape = min(len(v) for v in data_mape.values())
        for mode in data_mape:
            data_mape[mode] = data_mape[mode][:min_length_mape]
            
    modes = sorted(set(list(data_rmse.keys()) + list(data_mape.keys())))
    df = pd.DataFrame(index=modes, columns=['rmse', 'mape'])
    
    for mode in modes:
        if mode in data_rmse:
            df.loc[mode, 'rmse'] = data_rmse[mode]
        if mode in data_mape:
            df.loc[mode, 'mape'] = data_mape[mode]
            
    return df

