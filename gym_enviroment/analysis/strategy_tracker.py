#!/usr/bin/env python3
"""
Strategy Tracker for RL Negotiation Agent
Tracks and visualizes the strategy selection throughout negotiation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os


class StrategyTracker:
    """Tracks RL model strategy selection throughout negotiation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracker for new negotiation."""
        self.timestamps = []
        self.strategy_indices = []   # Selected strategy at each step
        self.step_count = 0
    
    def record_strategy(self, timestamp: float, strategy_index: int):
        """Record strategy selection at a specific timestamp."""
        self.timestamps.append(timestamp)
        self.strategy_indices.append(strategy_index)
        self.step_count += 1
    
    def plot_strategy_selection(self, opponent_name: str, domain: str, save_dir: str = "plots"):
        """Create visualization of strategy selection evolution."""
        if not self.timestamps:
            print("⚠️  No strategy data to plot")
            return
        
        # Create domain-specific plots directory
        domain_dir = f"{save_dir}/domain{domain}"
        os.makedirs(domain_dir, exist_ok=True)
        
        # Convert to numpy arrays for easier manipulation
        timestamps = np.array(self.timestamps)
        strategy_indices = np.array(self.strategy_indices)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Strategy selection over time
        strategy_names = ['SAGA', 'Hybrid', 'Conceder', 'Boulware']
        unique_strategies = np.unique(strategy_indices)
        strategy_colors = ['blue', 'green', 'orange', 'purple']
        
        # Plot strategy selection as scatter points
        for strategy_idx in unique_strategies:
            if strategy_idx < len(strategy_names):
                # Create binary mask for this strategy
                strategy_mask = strategy_indices == strategy_idx
                strategy_times = timestamps[strategy_mask]
                strategy_values = np.full_like(strategy_times, strategy_idx)
                
                # Plot as scatter points
                ax.scatter(strategy_times, strategy_values, 
                          c=strategy_colors[strategy_idx], 
                          label=strategy_names[strategy_idx], 
                          alpha=0.8, s=60, marker='o')
        
        # Connect the strategy points with lines to show transitions
        ax.plot(timestamps, strategy_indices, 'gray', linewidth=2, alpha=0.6, linestyle='-')
        
        # Styling
        ax.set_ylabel('Selected Strategy')
        ax.set_xlabel('Negotiation Time')
        ax.set_title(f'Strategy Selection Over Time - vs {opponent_name} on domain {domain}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_yticks(range(len(strategy_names)))
        ax.set_yticklabels(strategy_names)
        ax.set_ylim(-0.5, len(strategy_names) - 0.5)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"{domain_dir}/strategy_vs_{opponent_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Strategy selection plot saved as {filename}")
        
        # Print summary statistics
        self._print_strategy_stats()
    
    def _print_strategy_stats(self):
        """Print summary statistics about strategy selection."""
        if not self.timestamps:
            return
        
        strategy_indices = np.array(self.strategy_indices)
        strategy_names = ['SAGA', ]
        
        print(f"\n📈 Strategy Selection Statistics:")
        print(f"   Total steps recorded: {len(self.timestamps)}")
        
        # Strategy usage
        strategy_counts = np.bincount(strategy_indices, minlength=4)
        print(f"   Strategy usage:")
        for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
            percentage = (count / len(strategy_indices)) * 100 if strategy_indices.size > 0 else 0
            print(f"     {name}: {count} times ({percentage:.1f}%)")
        
        # Most common strategy
        if strategy_indices.size > 0:
            most_common_idx = np.argmax(strategy_counts)
            print(f"   Most used strategy: {strategy_names[most_common_idx]} ({strategy_counts[most_common_idx]} times)")
            
            # Strategy transitions
            transitions = np.sum(np.diff(strategy_indices) != 0)
            print(f"   Strategy transitions: {transitions}")


def create_strategy_comparison_plot(trackers: Dict[str, StrategyTracker], domain: str, save_dir: str = "plots"):
    """Create comparison plot of strategy usage across multiple opponents."""
    if not trackers:
        return
    
    # Create domain-specific plots directory
    domain_dir = f"{save_dir}/domain{domain}"
    os.makedirs(domain_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    strategy_names = ['SAGA', 'Hybrid', 'Conceder', 'Boulware']
    colors = ['blue', 'green', 'orange', 'purple']
    
    # Plot 1: Strategy usage percentage by opponent
    opponents = list(trackers.keys())
    strategy_percentages = np.zeros((len(opponents), 4))
    
    for i, (opponent, tracker) in enumerate(trackers.items()):
        if tracker.strategy_indices:
            strategy_counts = np.bincount(tracker.strategy_indices, minlength=4)
            total_steps = len(tracker.strategy_indices)
            strategy_percentages[i] = (strategy_counts / total_steps) * 100
    
    # Create stacked bar chart
    bottom = np.zeros(len(opponents))
    for j, (strategy_name, color) in enumerate(zip(strategy_names, colors)):
        ax1.bar(opponents, strategy_percentages[:, j], bottom=bottom, 
               label=strategy_name, color=color, alpha=0.8)
        bottom += strategy_percentages[:, j]
    
    ax1.set_ylabel('Strategy Usage (%)')
    ax1.set_title(f'Strategy Usage by Opponent - Domain {domain}')
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Rotate x-axis labels if they're too long
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Average strategy transitions
    transition_counts = []
    for opponent, tracker in trackers.items():
        if len(tracker.strategy_indices) > 1:
            transitions = np.sum(np.diff(tracker.strategy_indices) != 0)
            transition_counts.append(transitions)
        else:
            transition_counts.append(0)
    
    ax2.bar(opponents, transition_counts, color='gray', alpha=0.7)
    ax2.set_ylabel('Number of Strategy Transitions')
    ax2.set_xlabel('Opponent')
    ax2.set_title(f'Strategy Transitions by Opponent - Domain {domain}')
    
    # Rotate x-axis labels if they're too long
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save comparison plot
    filename = f"{domain_dir}/strategy_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Strategy comparison plot saved as {filename}")
