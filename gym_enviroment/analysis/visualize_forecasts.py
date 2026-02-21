import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import torch
from nenv.utils.DynamicImport import load_agent_class
from gym_enviroment.env import NegotiationEnv
from gym_enviroment.config.config import config


class ForecastCollector:
    """Collects forecasts, Nash utilities, and actual behavior throughout an episode."""
    
    def __init__(self):
        self.forecasts_history = []  # List of forecasts for candidate [0] at each step
        self.nash_baseline_history = []  # List of Nash baseline forecasts
        self.actual_opponent_utilities = []  # What opponent actually gave itself
        self.actual_nash_utilities = []  # Actual Nash utilities achieved
        self.strategy_selections = []  # Which strategy was selected each step
        self.step_info = []  # List of step information (round, time, etc.)
        self.final_utility = 0.0
        self.episode_done = False
        
    def reset(self):
        """Reset collector for new episode."""
        self.forecasts_history = []
        self.nash_baseline_history = []
        self.actual_opponent_utilities = []
        self.actual_nash_utilities = []
        self.strategy_selections = []
        self.step_info = []
        self.final_utility = 0.0
        self.episode_done = False
        
    def collect_step(self, obs: Dict, info: Dict, reward: float, done: bool, env: NegotiationEnv):
        """Collect forecast data, Nash utilities, and actual behavior from a single step."""
        candidate_forecasts = obs.get('candidate_forecasts', None)
        
        if candidate_forecasts is not None and len(candidate_forecasts) > 0:
            # Extract forecasts for candidate [0]
            candidate_0_forecast = candidate_forecasts[0]  # Shape: (336,)
            self.forecasts_history.append(candidate_0_forecast.copy())
            
            # Get Nash baseline forecast from agent if available
            if hasattr(env.our_agent, 'nash_baseline_forecast') and env.our_agent.nash_baseline_forecast is not None:
                self.nash_baseline_history.append(env.our_agent.nash_baseline_forecast.copy())
            else:
                self.nash_baseline_history.append(np.zeros_like(candidate_0_forecast))
            
            # Get strategy selection if available
            if hasattr(env.our_agent, 'selected_strategy'):
                self.strategy_selections.append(env.our_agent.selected_strategy)
            else:
                self.strategy_selections.append(0)
            
            # Get actual opponent utility from their last bid
            if hasattr(env, 'last_opponent_bid') and env.last_opponent_bid is not None:
                # What utility did the opponent give itself with their bid?
                opponent_utility = env.opponent_preference.get_utility(env.last_opponent_bid)
                self.actual_opponent_utilities.append(opponent_utility)
                
                # Calculate actual Nash utility
                our_utility = env.our_preference.get_utility(env.last_opponent_bid)
                nash_utility = our_utility * opponent_utility
                self.actual_nash_utilities.append(nash_utility)
            else:
                # No opponent bid yet (first step)
                self.actual_opponent_utilities.append(0.0)
                self.actual_nash_utilities.append(0.0)
            
            # Store step information
            step_info = {
                'round': info.get('round', len(self.forecasts_history) - 1),
                'time_left': info.get('time_left', 1.0),
                'domain': info.get('domain', 'unknown'),
                'opponent': info.get('opponent', 'unknown')
            }
            self.step_info.append(step_info)
            
        if done:
            self.final_utility = reward
            self.episode_done = True


def run_episode_with_forecast_collection(env: NegotiationEnv, agent_policy=None) -> ForecastCollector:
    """Run a single episode while collecting forecast data."""
    collector = ForecastCollector()
    
    obs, info = env.reset()
    collector.collect_step(obs, info, 0.0, False, env)
    
    step_count = 0
    max_steps = 10000  # Safety limit
    
    while step_count < max_steps:
        # Simple policy: select candidate based on some strategy
        if agent_policy is not None:
            action = agent_policy(obs, info)
        else:
            # Default: always select candidate [0] for this visualization
            action = 0
            
        obs, reward, done, truncated, info = env.step(action)
        collector.collect_step(obs, info, reward, done or truncated, env)
        
        if done or truncated:
            break
            
        step_count += 1
    
    return collector


def visualize_forecast_vs_actual(collector: ForecastCollector, save_path: str = None):
    """Create comprehensive visualization comparing forecasted vs actual behavior with Nash utilities."""
    if not collector.forecasts_history:
        print("No forecast data collected!")
        return
        
    # Convert to numpy arrays for easier manipulation
    forecasts_array = np.array(collector.forecasts_history)  # Shape: (n_steps, 336)
    nash_baseline_array = np.array(collector.nash_baseline_history) if collector.nash_baseline_history else None
    actual_utilities = np.array(collector.actual_opponent_utilities)  # Shape: (n_steps,)
    actual_nash = np.array(collector.actual_nash_utilities)  # Shape: (n_steps,)
    strategy_selections = np.array(collector.strategy_selections) if collector.strategy_selections else None
    n_steps, forecast_length = forecasts_array.shape
    
    # Create the plot with more subplots for comprehensive analysis
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    steps = range(n_steps)
    
    # Top plot: Strategy forecasts vs Nash baseline forecasts
    mean_forecasts = np.mean(forecasts_array, axis=1)
    std_forecasts = np.std(forecasts_array, axis=1)
    
    axes[0].plot(steps, mean_forecasts, 'b-', linewidth=2, label='Strategy Forecast (Mean)', alpha=0.8)
    axes[0].fill_between(steps, mean_forecasts - std_forecasts, mean_forecasts + std_forecasts, 
                        alpha=0.2, color='blue', label='Strategy Forecast ±1 Std')
    
    if nash_baseline_array is not None and nash_baseline_array.size > 0:
        mean_nash_baseline = np.mean(nash_baseline_array, axis=1)
        std_nash_baseline = np.std(nash_baseline_array, axis=1)
        axes[0].plot(steps, mean_nash_baseline, 'g--', linewidth=2, label='Nash Baseline (Mean)', alpha=0.8)
        axes[0].fill_between(steps, mean_nash_baseline - std_nash_baseline, mean_nash_baseline + std_nash_baseline, 
                            alpha=0.2, color='green', label='Nash Baseline ±1 Std')
    
    axes[0].set_ylabel('Predicted Utility')
    axes[0].set_title('Forecasting Comparison: Strategy vs Nash Baseline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Middle plot: Actual utilities (opponent and Nash)
    axes[1].plot(steps, actual_utilities, 'r-', linewidth=2, marker='o', markersize=3, 
                label='Opponent Utility', alpha=0.8)
    axes[1].plot(steps, actual_nash, 'purple', linewidth=2, marker='s', markersize=3, 
                label='Nash Utility', alpha=0.8)
    axes[1].set_ylabel('Actual Utility')
    axes[1].set_title('Actual Negotiation Outcomes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Bottom plot: Strategy selection over time
    if strategy_selections is not None:
        # Create strategy name mapping
        strategy_names = ['SAGA', 'Hybrid', 'Conceder', 'Boulware', 'AgentKN']  # Based on main_strategy.py
        unique_strategies = np.unique(strategy_selections)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_strategies)))
        
        for i, strategy_idx in enumerate(unique_strategies):
            mask = strategy_selections == strategy_idx
            strategy_name = strategy_names[strategy_idx] if strategy_idx < len(strategy_names) else f'Strategy {strategy_idx}'
            axes[2].scatter(np.array(steps)[mask], [strategy_idx] * np.sum(mask), 
                           c=[colors[i]], label=strategy_name, alpha=0.7, s=30)
        
        axes[2].set_ylabel('Selected Strategy')
        axes[2].set_xlabel('Negotiation Step')
        axes[2].set_title('Strategy Selection Over Time')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yticks(range(len(strategy_names)))
        axes[2].set_yticklabels(strategy_names)
    else:
        axes[2].text(0.5, 0.5, 'Strategy selection data not available', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_xlabel('Negotiation Step')
    
    # Add episode info
    if collector.step_info:
        info_text = f"Domain: {collector.step_info[0]['domain']} | Opponent: {collector.step_info[0]['opponent']} | Final Utility: {collector.final_utility:.3f}"
        fig.suptitle(info_text, fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    # Print comprehensive summary statistics
    print(f"\n=== Episode Analysis ===")
    print(f"Total Steps: {n_steps}")
    print(f"Final Utility: {collector.final_utility:.3f}")
    
    print(f"\n=== Forecasting Performance ===")
    print(f"Mean Strategy Forecast: {np.mean(forecasts_array):.3f}")
    if nash_baseline_array is not None and nash_baseline_array.size > 0:
        print(f"Mean Nash Baseline Forecast: {np.mean(nash_baseline_array):.3f}")
    print(f"Mean Opponent Utility: {np.mean(actual_utilities):.3f}")
    print(f"Mean Nash Utility: {np.mean(actual_nash):.3f}")
    
    print(f"\n=== Correlations ===")
    strategy_vs_opponent = np.corrcoef(mean_forecasts, actual_utilities)[0,1]
    print(f"Strategy Forecast vs Opponent Utility: {strategy_vs_opponent:.3f}")
    
    if nash_baseline_array is not None and nash_baseline_array.size > 0:
        nash_vs_actual = np.corrcoef(mean_nash_baseline, actual_nash)[0,1]
        print(f"Nash Baseline vs Actual Nash: {nash_vs_actual:.3f}")
    
    print(f"\n=== Strategy Usage ===")
    if strategy_selections is not None:
        strategy_counts = np.bincount(strategy_selections, minlength=5)
        strategy_names = ['SAGA', 'Hybrid', 'Conceder', 'Boulware', 'AgentKN']
        for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
            percentage = (count / n_steps) * 100 if n_steps > 0 else 0
            print(f"{name}: {count} times ({percentage:.1f}%)")
    
    return fig


def main():
    """Main function to run forecast collection and visualization."""
    print("Setting up environment...")
    
    # Load agent class
    agent_class = load_agent_class('gym_enviroment.agent.main_strategy.MainStrategy')
    
    # Setup environment with more diverse testing
    domains = ['5']  # Use domain0 for consistent testing
    opponents = ['agents.BoulwareAgent']  # Updated path based on project structure
    deadline_round = 1000  # Shorter episodes for faster testing
    
    env = NegotiationEnv(
        our_agent_class=agent_class,
        domains=domains,
        deadline_round=deadline_round,
        opponent_names=opponents,
        mode="oracle"  # Use oracle mode for cleaner Nash calculations
    )
    
    print("Running episode with comprehensive forecast collection...")
    
    # Simple policy for testing: use random strategy selection
    def simple_policy(obs, info):
        """Simple policy that varies strategy selection for testing."""
        # Create a simple action vector: [strategy_index, weights...]
        strategy_idx = np.random.randint(0, 5)  # Random strategy
        weights = np.random.uniform(-1, 1, 97)  # Random weights for Nash equation
        return np.concatenate([[strategy_idx], weights])
    
    # Run episode and collect forecasts
    collector = run_episode_with_forecast_collection(env, agent_policy=simple_policy)
    
    if not collector.forecasts_history:
        print("No forecasts were collected! Check if the agent is generating forecasts properly.")
        return
        
    print(f"Collected {len(collector.forecasts_history)} forecast observations")
    print(f"Nash baseline history length: {len(collector.nash_baseline_history)}")
    print(f"Strategy selections length: {len(collector.strategy_selections)}")
    
    # Create comprehensive visualization
    print("Creating enhanced visualization with Nash utilities and strategy analysis...")
    opponent_name = opponents[0].split('.')[-1] if '.' in opponents[0] else opponents[0]
    visualize_forecast_vs_actual(collector, save_path=f'comprehensive_forecast_analysis_{opponent_name}.png')
 

if __name__ == "__main__":
    main() 