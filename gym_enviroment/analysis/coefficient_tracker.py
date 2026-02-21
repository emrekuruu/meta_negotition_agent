#!/usr/bin/env python3
"""
Coefficient Tracker for RL Negotiation Agent
Tracks and visualizes the Nash equation coefficients throughout negotiation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os


class CoefficientTracker:
    """Tracks RL model coefficients throughout negotiation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracker for new negotiation."""
        self.timestamps = []
        self.nash_coefficients = []  # w1-w96 (96 coefficients)
        self.time_coefficients = []  # w97 (time coefficient)
        self.strategy_indices = []   # Selected strategy at each step
        self.step_count = 0
    
    def record_coefficients(self, timestamp: float, action_vector: np.ndarray, strategy_index: int):
        """Record coefficients at a specific timestamp."""
        self.timestamps.append(timestamp)
        self.step_count += 1
        
        # Parse action vector: [strategy_index, w1, w2, ..., w96, w97]
        if len(action_vector) >= 98:  # strategy_index + 96 nash + 1 time
            nash_weights = action_vector[1:97]  # w1-w96
            time_weight = action_vector[97] if len(action_vector) > 97 else 0.0  # w97
        else:
            # Fallback for shorter vectors
            nash_weights = action_vector[1:min(97, len(action_vector))]
            # Pad with zeros if needed
            while len(nash_weights) < 96:
                nash_weights = np.append(nash_weights, 0.0)
            time_weight = action_vector[-1] if len(action_vector) > 1 else 0.0
        
        self.nash_coefficients.append(nash_weights)
        self.time_coefficients.append(time_weight)
        self.strategy_indices.append(strategy_index)
    
    def plot_coefficients(self, opponent_name: str, domain: str, save_dir: str = "plots"):
        """Create visualization of coefficient evolution with line plots."""
        if not self.timestamps:
            print("⚠️  No coefficient data to plot")
            return
        
        # Create domain-specific plots directory
        domain_dir = f"{save_dir}/domain{domain}"
        os.makedirs(domain_dir, exist_ok=True)
        
        # Convert to numpy arrays for easier manipulation
        nash_coeffs = np.array(self.nash_coefficients)  # Shape: (n_steps, 96)
        time_coeffs = np.array(self.time_coefficients)   # Shape: (n_steps,)
        timestamps = np.array(self.timestamps)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 1. Nash coefficients as individual line plots (w1-w96)
        if nash_coeffs.size > 0:
            # Create a colormap for the 96 coefficients
            colors = plt.cm.viridis(np.linspace(0, 1, 96))
            
            # Plot each coefficient as a line
            for i in range(96):
                ax1.plot(timestamps, nash_coeffs[:, i], color=colors[i], 
                        linewidth=1, alpha=0.7)
            
            ax1.set_ylabel('Coefficient Value')
            ax1.set_title(f'Nash Coefficients Evolution (w1-w96) - vs {opponent_name} on domain {domain}')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar to show which coefficient corresponds to which color
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=96))
            sm.set_array([])
            cbar1 = plt.colorbar(sm, ax=ax1)
            cbar1.set_label('Coefficient Index (w1-w96)')
        
        # 2. Time coefficient (w97) as a prominent line plot
        ax2.plot(timestamps, time_coeffs, 'red', linewidth=3, marker='o', 
                markersize=6, alpha=0.9, label='Time Coefficient (w97)')
        ax2.set_ylabel('Time Coefficient (w97)')
        ax2.set_xlabel('Negotiation Time')
        ax2.set_title(f'Time Coefficient Evolution (w97) - vs {opponent_name} on domain {domain}')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"{domain_dir}/coefficients_vs_{opponent_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Coefficients plot saved as {filename}")
        
        # Print summary statistics
        self._print_coefficient_stats()
    
    def _print_coefficient_stats(self):
        """Print summary statistics about coefficients."""
        if not self.timestamps:
            return
        
        nash_coeffs = np.array(self.nash_coefficients)
        time_coeffs = np.array(self.time_coefficients)
        
        print(f"\n📈 Coefficient Statistics:")
        print(f"   Total steps recorded: {len(self.timestamps)}")
        print(f"   Time coefficient range: [{np.min(time_coeffs):.3f}, {np.max(time_coeffs):.3f}]")
        print(f"   Time coefficient final: {time_coeffs[-1]:.3f}")
        print(f"   Nash coefficients range: [{np.min(nash_coeffs):.3f}, {np.max(nash_coeffs):.3f}]")
        print(f"   Most active Nash coefficients: {np.argsort(np.std(nash_coeffs, axis=0))[-5:] + 1}")  # +1 for 1-indexed
        
        # Strategy usage
        strategy_counts = np.bincount(self.strategy_indices, minlength=4)
        strategy_names = ['SAGA', 'Hybrid', 'Conceder', 'Boulware']
        print(f"   Strategy usage:")
        for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
            percentage = (count / len(self.strategy_indices)) * 100 if self.strategy_indices else 0
            print(f"     {name}: {count} times ({percentage:.1f}%)")


def create_coefficient_comparison_plot(trackers: Dict[str, CoefficientTracker], domain: str, save_dir: str = "plots"):
    """Create comparison plot across multiple opponents."""
    if not trackers:
        return
    
    # Create domain-specific plots directory
    domain_dir = f"{save_dir}/domain{domain}"
    os.makedirs(domain_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trackers)))
    
    # Plot time coefficients for all opponents
    for i, (opponent, tracker) in enumerate(trackers.items()):
        if tracker.timestamps:
            ax1.plot(tracker.timestamps, tracker.time_coefficients, 
                    color=colors[i], linewidth=2, alpha=0.7, label=f'vs {opponent}')
    
    ax1.set_ylabel('Time Coefficient (w97)')
    ax1.set_xlabel('Negotiation Time')
    ax1.set_title(f'Time Coefficient Comparison - Domain {domain}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot final Nash coefficient distributions
    all_final_nash = []
    opponent_labels = []
    
    for opponent, tracker in trackers.items():
        if tracker.nash_coefficients:
            final_nash = tracker.nash_coefficients[-1]  # Last coefficients
            all_final_nash.append(final_nash)
            opponent_labels.append(opponent)
    
    if all_final_nash:
        # Create heatmap of final Nash coefficients
        nash_matrix = np.array(all_final_nash)
        im = ax2.imshow(nash_matrix, aspect='auto', cmap='RdBu_r', origin='lower')
        ax2.set_ylabel('Opponent')
        ax2.set_xlabel('Nash Coefficient Index (w1-w96)')
        ax2.set_title('Final Nash Coefficients by Opponent')
        ax2.set_yticks(range(len(opponent_labels)))
        ax2.set_yticklabels(opponent_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Coefficient Value')
    
    plt.tight_layout()
    
    # Save comparison plot
    filename = f"{domain_dir}/coefficient_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Coefficient comparison plot saved as {filename}")