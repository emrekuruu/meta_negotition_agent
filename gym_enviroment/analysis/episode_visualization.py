"""
Comprehensive Episode Visualization Script
Visualizes all episode dynamics: rewards, utilities, strategies, and coefficients.

Now supports timestep range filtering for fair comparison across environments with different speeds.
You can specify both lower and upper bounds to analyze specific training periods.

Example usage:
    # Visualize data from timestep 50000 onwards
    python episode_visualization.py --min-timestep 50000 --type all
    
    # Visualize data between timesteps 50000-100000
    python episode_visualization.py --min-timestep 50000 --max-timestep 100000 --type rewards --run_id my_run
    
    # List available timesteps
    python episode_visualization.py --list
    
    # Compare specific training periods
    python episode_visualization.py --min-timestep 100000 --max-timestep 150000 --type strategies
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


class EpisodeVisualizer:
    """Comprehensive visualizer for all episode dynamics data."""
    
    def __init__(self, log_dir: str = "episode_logs", run_id: str = ""):
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.reward_dir = self.log_dir / run_id / "reward_tracking"
        self.utility_dir = self.log_dir / run_id / "utility_tracking"
        self.strategy_dir = self.log_dir / run_id / "strategy_tracking"
        self.coefficient_dir = self.log_dir / run_id / "coefficient_tracking"
    
    def plot_episode_rewards(self, min_timestep: int = 0, max_timestep: int = None):
        """Plot Nash and Strategy Fit reward dynamics for all opponents within a timestep range."""
        
        # Load reward data
        global_csv = self.reward_dir / "global_episode_dynamics.csv"
        if not global_csv.exists():
            print(f"❌ Reward CSV not found: {global_csv}")
            return
        
        df = pd.read_csv(global_csv)
        
        # Apply timestep filtering
        if max_timestep is None:
            timestep_data = df[df['timestep'] >= min_timestep]
            range_desc = f"timestep >= {min_timestep}"
        else:
            timestep_data = df[(df['timestep'] >= min_timestep) & (df['timestep'] <= max_timestep)]
            range_desc = f"timestep {min_timestep}-{max_timestep}"
        
        if len(timestep_data) == 0:
            print(f"❌ No reward data found for {range_desc}")
            available_timesteps = sorted(df['timestep'].unique())
            print(f"Available timesteps: {available_timesteps}")
            return
        
        # Get unique domains and opponents for domain-based organization
        domains = timestep_data['domain'].unique()
        opponents = timestep_data['opponent'].unique()
        
        # Create base plots directory
        plots_dir = self.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print(f"📊 Creating reward plots for {len(domains)} domains: {list(domains)}")
        
        # Process each domain separately
        for domain in domains:
            domain_data = timestep_data[timestep_data['domain'] == domain]
            if len(domain_data) == 0:
                continue
                
            # Create domain-specific directory
            domain_plots_dir = plots_dir / str(domain)
            domain_plots_dir.mkdir(exist_ok=True)
            
            # Create separate subplots for Nash and Strategy Fit rewards
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            domain_opponents = domain_data['opponent'].unique()
            
            for opponent in domain_opponents:
                opp_data = domain_data[domain_data['opponent'] == opponent]
                nash_data = opp_data.dropna(subset=['nash_reward'])
                strategy_fit_data = opp_data.dropna(subset=['strategy_fit_reward'])
                terminal_data = opp_data.dropna(subset=['terminal_reward'])
                
                # Get terminal reward for legend
                terminal_reward = terminal_data['terminal_reward'].iloc[0] if len(terminal_data) > 0 else "N/A"
                terminal_round = terminal_data['round'].iloc[0] if len(terminal_data) > 0 else "N/A"
                
                # Create labels
                if terminal_reward != "N/A":
                    nash_label = f"{opponent} (Terminal: {terminal_reward:.3f} @ round {terminal_round})"
                    strategy_fit_label = f"{opponent}"
                else:
                    nash_label = f"{opponent} (No terminal reward)"
                    strategy_fit_label = f"{opponent}"
                
                # Plot Nash rewards in first subplot
                if len(nash_data) > 0:
                    ax1.plot(nash_data['round'], nash_data['nash_reward'], 
                            'o-', label=nash_label, linewidth=2, markersize=4, alpha=0.8)
                
                # Plot Strategy Fit rewards in second subplot
                if len(strategy_fit_data) > 0:
                    ax2.plot(strategy_fit_data['round'], strategy_fit_data['strategy_fit_reward'], 
                            's-', label=strategy_fit_label, linewidth=2, markersize=4, alpha=0.8)
            
            # Configure Nash rewards subplot (top)
            ax1.set_xlabel('Round', fontsize=12)
            ax1.set_ylabel('Nash Reward', fontsize=12)
            ax1.set_title(f'Domain {domain} - Nash Reward Dynamics ({range_desc})', fontsize=13, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Configure Strategy Fit rewards subplot (bottom)
            ax2.set_xlabel('Round', fontsize=12)
            ax2.set_ylabel('Strategy Fit Reward', fontsize=12)
            ax2.set_title(f'Domain {domain} - Strategy Fit Reward Dynamics ({range_desc})', fontsize=13, fontweight='bold')
            ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Baseline (Perfect Strategy Fit)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Create filename based on range
            if max_timestep is None:
                filename = f'rewards_timestep{min_timestep}+_domain_{domain}.png'
            else:
                filename = f'rewards_timestep{min_timestep}-{max_timestep}_domain_{domain}.png'
            
            output_path = domain_plots_dir / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Domain {domain} reward plot saved: {output_path}")
            
            # Print domain-specific statistics
            print(f"📈 Domain {domain} Reward Statistics for {range_desc}:")
            for opponent in domain_opponents:
                opp_data = domain_data[domain_data['opponent'] == opponent]
                nash_data = opp_data.dropna(subset=['nash_reward'])
                strategy_fit_data = opp_data.dropna(subset=['strategy_fit_reward'])
                
                if len(nash_data) > 0:
                    nash_mean = nash_data['nash_reward'].mean()
                    nash_std = nash_data['nash_reward'].std()
                    print(f"   {opponent} Nash: {len(nash_data)} points, avg={nash_mean:.4f}, std={nash_std:.4f}")
                
                if len(strategy_fit_data) > 0:
                    fit_mean = strategy_fit_data['strategy_fit_reward'].mean()
                    fit_std = strategy_fit_data['strategy_fit_reward'].std()
                    print(f"   {opponent} Strategy Fit: {len(strategy_fit_data)} points, avg={fit_mean:.4f}, std={fit_std:.4f}")
            print()
    
    def plot_episode_utilities(self, min_timestep: int = 0, max_timestep: int = None):
        """Plot utility dynamics for data within a timestep range."""
        
        # Load utility data
        global_csv = self.utility_dir / "global_utility_dynamics.csv"
        if not global_csv.exists():
            print(f"❌ Utility CSV not found: {global_csv}")
            return
        
        df = pd.read_csv(global_csv)
        
        # Apply timestep filtering
        if max_timestep is None:
            timestep_data = df[df['timestep'] >= min_timestep]
            range_desc = f"timestep >= {min_timestep}"
        else:
            timestep_data = df[(df['timestep'] >= min_timestep) & (df['timestep'] <= max_timestep)]
            range_desc = f"timestep {min_timestep}-{max_timestep}"
        
        if len(timestep_data) == 0:
            print(f"❌ No utility data found for {range_desc}")
            available_timesteps = sorted(df['timestep'].unique())
            print(f"Available timesteps: {available_timesteps}")
            return
        
        # Create plots directory
        plots_dir = self.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Get unique domains for domain-based organization
        domains = timestep_data['domain'].unique()
        print(f"📊 Creating utility plots for {len(domains)} domains: {list(domains)}")
        
        # Process each domain separately
        for domain in domains:
            domain_data = timestep_data[timestep_data['domain'] == domain]
            if len(domain_data) == 0:
                continue
                
            # Create domain-specific directory
            domain_plots_dir = plots_dir / str(domain)
            domain_plots_dir.mkdir(exist_ok=True)
            
            opponents = domain_data['opponent'].unique()
            
            for opponent in opponents:
                # Get data for this opponent - DON'T SORT to preserve alternating pattern
                opp_data = domain_data[domain_data['opponent'] == opponent]
                
                if len(opp_data) == 0:
                    continue
                
                print(f"Processing {len(opp_data)} rows for {opponent} in domain {domain}")
                
                # Separate data by bid type using the new bid_type column
                our_bid_data = opp_data[opp_data['bid_type'] == 'our_bid']
                opp_bid_data = opp_data[opp_data['bid_type'] == 'opp_bid']
            
            print(f"Found {len(our_bid_data)} our_bid entries and {len(opp_bid_data)} opp_bid entries")
            
            # Debug: show what bid types we actually have
            unique_bid_types = opp_data['bid_type'].unique()
            print(f"Available bid types: {unique_bid_types}")
            
            # Blue dots = our offers
            blue_times_left = []    # our_utility from our offers
            blue_values_left = []
            blue_times_right = []   # opp_utility from our offers
            blue_values_right = []
            
            # Red dots = opponent offers
            red_times_left = []     # our_utility from opponent offers
            red_values_left = []
            red_times_right = []    # opp_utility from opponent offers
            red_values_right = []
            
            # Process our bid data (BLUE dots)
            for _, row in our_bid_data.iterrows():
                round_num = float(row['round'])  # Ensure it's a number
                time = round_num / 1000.0
                
                blue_times_left.append(time)
                blue_values_left.append(row['our_utility'])
                blue_times_right.append(time)
                blue_values_right.append(row['opp_utility'])
            
            # Process opponent bid data (RED dots)
            for _, row in opp_bid_data.iterrows():
                round_num = float(row['round'])  # Ensure it's a number
                time = round_num / 1000.0
                
                red_times_left.append(time)
                red_values_left.append(row['our_utility'])
                red_times_right.append(time)
                red_values_right.append(row['opp_utility'])
            
            print(f"Extracted {len(blue_times_left)} blue dots, {len(red_times_left)} red dots")
            print(f"Blue values left (our utility from our offers): {blue_values_left[:5]} ... {blue_values_left[-5:]}")
            print(f"Blue values right (opp utility from our offers): {blue_values_right[:5]} ... {blue_values_right[-5:]}")
            print(f"Red values left (our utility from their offers): {red_values_left[:5]} ... {red_values_left[-5:]}")
            print(f"Red values right (opp utility from their offers): {red_values_right[:5]} ... {red_values_right[-5:]}")
            
            if not blue_times_left or not red_times_left:
                print(f"❌ No valid offers found for {opponent}")
                continue
            
            # Final agreement = use the last available data point (could be from either bid type)
            # Find the row with the highest round number for the final agreement
            final_row = opp_data.loc[opp_data['round'].idxmax()]
            final_our_utility = final_row['our_utility']
            final_opp_utility = final_row['opp_utility']
            final_time = float(final_row['round']) / 1000.0
            
            # Create the dual perspective plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # === LEFT SUBPLOT: Our Utility ===
            # Blue dots = first row of each round (our offers)
            ax1.scatter(blue_times_left, blue_values_left, c='blue', s=50, alpha=0.7, 
                       label=f'Our Offers ({len(blue_times_left)} total)', marker='o')
            # Red dots = second row of each round (their offers)
            ax1.scatter(red_times_left, red_values_left, c='red', s=50, alpha=0.7, 
                       label=f'Opponent Offers ({len(red_times_left)} total)', marker='s')
            
            ax1.scatter([final_time], [final_our_utility], c='green', s=200, alpha=0.9,
                       label=f'Final Agreement (Ours: {final_our_utility:.3f}, Opponents: {final_opp_utility:.3f})', 
                       marker='*', edgecolors='darkgreen', linewidth=2)
            
            ax1.set_xlabel('Negotiation Time (0 = start, 1 = deadline)')
            ax1.set_ylabel('Our Utility')
            ax1.set_title(f'Our Utility Perspective vs {opponent}')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.set_xlim(0, 1)
            
            # === RIGHT SUBPLOT: Opponent Utility ===
            # Blue dots = first row of each round (our offers, their perspective)
            ax2.scatter(blue_times_right, blue_values_right, c='blue', s=50, alpha=0.7, marker='o')
            # Red dots = second row of each round (their offers, their perspective)
            ax2.scatter(red_times_right, red_values_right, c='red', s=50, alpha=0.7, marker='s')
            
            ax2.scatter([final_time], [final_opp_utility], c='green', s=200, alpha=0.9,
                       marker='*', edgecolors='darkgreen', linewidth=2)
            
            ax2.set_xlabel('Negotiation Time (0 = start, 1 = deadline)')
            ax2.set_ylabel('Opponent Utility')
            ax2.set_title(f'Opponent Utility Perspective vs {opponent}')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            ax2.set_xlim(0, 1)
            
            # Legend and statistics
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(0.85, 0.85), loc='upper left')
            plt.subplots_adjust(right=0.75)
            
            # Statistics
            stats_text = f"OUR AGENT:\n"
            stats_text += f"• Avg utility: {np.mean(blue_values_left):.3f}\n" if blue_values_left else "• No our bids\n"
            stats_text += f"• Last offer: {blue_values_left[-1]:.3f}\n" if blue_values_left else ""
            stats_text += f"• Final agreement: {final_our_utility:.3f}\n"
            stats_text += f"• Avg opp utility: {np.mean(blue_values_right):.3f}\n" if blue_values_right else ""
            
            stats_text += f"\nOPPONENT:\n"
            stats_text += f"• Avg utility: {np.mean(red_values_right):.3f}\n" if red_values_right else "• No opp bids\n"
            stats_text += f"• Last offer: {red_values_right[-1]:.3f}\n" if red_values_right else ""
            stats_text += f"• Final agreement: {final_opp_utility:.3f}\n"
            stats_text += f"• Avg our utility: {np.mean(red_values_left):.3f}\n" if red_values_left else ""
            
            fig.text(0.85, 0.65, stats_text, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, fontfamily='monospace')
            
            # Save plot
            if max_timestep is None:
                filename = domain_plots_dir / f"negotiation_vs_{opponent}_timestep{min_timestep}+.png"
            else:
                filename = domain_plots_dir / f"negotiation_vs_{opponent}_timestep{min_timestep}-{max_timestep}.png"
            
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Negotiation plot saved as {filename}")
            print(f"   Our offers: {len(blue_times_left)}, Opponent offers: {len(red_times_left)}")
            print(f"   Final agreement utility (ours): {final_our_utility:.3f}")
            print(f"   Final agreement utility (opponent): {final_opp_utility:.3f}")
    
    def plot_episode_strategies(self, min_timestep: int = 0, max_timestep: int = None):
        """Plot strategy selection for all opponents within a timestep range."""
        
        # Load strategy data
        global_csv = self.strategy_dir / "global_strategy_dynamics.csv"
        if not global_csv.exists():
            print(f"❌ Strategy CSV not found: {global_csv}")
            return
        
        df = pd.read_csv(global_csv)
        
        # Apply timestep filtering
        if max_timestep is None:
            timestep_data = df[df['timestep'] >= min_timestep]
            range_desc = f"timestep >= {min_timestep}"
        else:
            timestep_data = df[(df['timestep'] >= min_timestep) & (df['timestep'] <= max_timestep)]
            range_desc = f"timestep {min_timestep}-{max_timestep}"
        
        if len(timestep_data) == 0:
            print(f"❌ No strategy data found for {range_desc}")
            available_timesteps = sorted(df['timestep'].unique())
            print(f"Available timesteps: {available_timesteps}")
            return
        
        # Create plots directory
        plots_dir = self.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Get unique domains for domain-based organization
        domains = timestep_data['domain'].unique()
        print(f"📊 Creating strategy plots for {len(domains)} domains: {list(domains)}")
        
        strategy_names = ['SAGA', 'Hybrid', 'Conceder', 'Boulware', 'MICRO', 'KAWAII']
        strategy_colors = ['blue', 'green', 'orange', 'purple', 'red', 'yellow']
        
        # Process each domain separately
        for domain in domains:
            domain_data = timestep_data[timestep_data['domain'] == domain]
            if len(domain_data) == 0:
                continue
                
            # Create domain-specific directory
            domain_plots_dir = plots_dir / str(domain)
            domain_plots_dir.mkdir(exist_ok=True)
            
            opponents = domain_data['opponent'].unique()
            
            for opponent in opponents:
                opp_data = domain_data[domain_data['opponent'] == opponent]
                
                # Convert to numpy arrays (same as strategy_tracker)
                timestamps = np.array(opp_data['time_progress'])
                strategy_indices = np.array(opp_data['strategy_index'])
                
                # Create figure (same layout as strategy_tracker)
                fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            
            # Strategy selection over time
            unique_strategies = np.unique(strategy_indices)
            
            # Plot strategy selection as scatter points (same as strategy_tracker)
            for strategy_idx in unique_strategies:
                if strategy_idx < len(strategy_names):
                    # Create binary mask for this strategy
                    strategy_mask = strategy_indices == strategy_idx
                    strategy_times = timestamps[strategy_mask]
                    strategy_values = np.full_like(strategy_times, strategy_idx)
                    
                    # Calculate count and percentage for label
                    count = np.sum(strategy_mask)
                    percentage = (count / len(strategy_indices)) * 100 if len(strategy_indices) > 0 else 0
                    label = f"{strategy_names[strategy_idx]} ({count} times, {percentage:.1f}%)"
                    
                    # Plot as scatter points
                    ax.scatter(strategy_times, strategy_values, 
                              c=strategy_colors[strategy_idx], 
                              label=label, 
                              alpha=0.8, s=60, marker='o')
            
            # Connect the strategy points with lines to show transitions (same as strategy_tracker)
            ax.plot(timestamps, strategy_indices, 'gray', linewidth=2, alpha=0.6, linestyle='-')
            
            # Styling (same as strategy_tracker)
            ax.set_ylabel('Selected Strategy')
            ax.set_xlabel('Negotiation Time')
            ax.set_title(f'Strategy Selection Over Time - {range_desc.capitalize()} vs {opponent}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_yticks(range(len(strategy_names)))
            ax.set_yticklabels(strategy_names)
            ax.set_ylim(-0.5, len(strategy_names) - 0.5)
            ax.set_xlim(0, 1)
            
            plt.tight_layout()
                
            # Save plot (same style as strategy_tracker)
            if max_timestep is None:
                filename = domain_plots_dir / f"strategy_timestep{min_timestep}+_vs_{opponent}.png"
            else:
                filename = domain_plots_dir / f"strategy_timestep{min_timestep}-{max_timestep}_vs_{opponent}.png"
            
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()  # Close to prevent display during batch processing
            
            print(f"📊 Domain {domain} strategy selection plot saved as {filename}")
            
            # Print summary statistics (same style as strategy_tracker)
            print(f"\n📈 Strategy Selection Statistics - {range_desc.capitalize()} vs {opponent}:")
            print(f"   Total steps recorded: {len(timestamps)}")
            
            # Strategy usage
            strategy_counts = np.bincount(strategy_indices, minlength=len(strategy_names))
            print(f"   Strategy usage:")
            for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
                percentage = (count / len(strategy_indices)) * 100 if len(strategy_indices) > 0 else 0
                print(f"     {name}: {count} times ({percentage:.1f}%)")
            
            # Most common strategy
            if len(strategy_indices) > 0:
                most_common_idx = np.argmax(strategy_counts)
                print(f"   Most used strategy: {strategy_names[most_common_idx]} ({strategy_counts[most_common_idx]} times)")
                
                # Strategy transitions
                transitions = np.sum(np.diff(strategy_indices) != 0)
                print(f"   Strategy transitions: {transitions}")
    
    def plot_episode_coefficients(self, min_timestep: int = 0, max_timestep: int = None):
        """Plot coefficient evolution for data within a timestep range."""
        
        # Load coefficient data
        global_csv = self.coefficient_dir / "global_coefficient_dynamics.csv"
        if not global_csv.exists():
            print(f"❌ Coefficient CSV not found: {global_csv}")
            return
        
        df = pd.read_csv(global_csv)
        
        # Apply timestep filtering
        if max_timestep is None:
            timestep_data = df[df['timestep'] >= min_timestep]
            range_desc = f"timestep >= {min_timestep}"
        else:
            timestep_data = df[(df['timestep'] >= min_timestep) & (df['timestep'] <= max_timestep)]
            range_desc = f"timestep {min_timestep}-{max_timestep}"
        
        if len(timestep_data) == 0:
            print(f"❌ No coefficient data found for {range_desc}")
            available_timesteps = sorted(df['timestep'].unique())
            print(f"Available timesteps: {available_timesteps}")
            return
        
        # Create plots directory
        plots_dir = self.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Get unique domains for domain-based organization
        domains = timestep_data['domain'].unique()
        print(f"📊 Creating coefficient plots for {len(domains)} domains: {list(domains)}")
        
        # Process each domain separately
        for domain in domains:
            domain_data = timestep_data[timestep_data['domain'] == domain]
            if len(domain_data) == 0:
                continue
                
            # Create domain-specific directory
            domain_plots_dir = plots_dir / str(domain)
            domain_plots_dir.mkdir(exist_ok=True)
            
            opponents = domain_data['opponent'].unique()
            
            for opponent in opponents:
                opp_data = domain_data[domain_data['opponent'] == opponent].sort_values('time_progress')
                
                if len(opp_data) == 0:
                    continue
            
            # Extract data exactly like coefficient_tracker.py expects
            timestamps = list(opp_data['time_progress'])
            nash_coefficients = []
            time_coefficients = list(opp_data['time_coefficient'])
            
            # Build nash_coefficients as list of arrays (like coefficient_tracker.py)
            nash_cols = [col for col in df.columns if col.startswith('nash_coeff_')]
            for _, row in opp_data.iterrows():
                nash_weights = np.array([row[col] for col in nash_cols])
                nash_coefficients.append(nash_weights)
            
            # Check if we have data
            if not timestamps:
                print("⚠️  No coefficient data to plot")
                continue
            
            # EXACT CODE FROM coefficient_tracker.py plot_coefficients method
            import os
            save_dir = str(plots_dir)
            
            # Create timestep range string for filename
            if max_timestep is None:
                timestep_range = f"{min_timestep}+"
            else:
                timestep_range = f"{min_timestep}-{max_timestep}"
            
            # Create plots directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Convert to numpy arrays for easier manipulation
            nash_coeffs = np.array(nash_coefficients)  # Shape: (n_steps, 96)
            time_coeffs = np.array(time_coefficients)   # Shape: (n_steps,)
            timestamps_array = np.array(timestamps)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # 1. Nash coefficients as individual line plots (w1-w96)
            if nash_coeffs.size > 0:
                # Create a colormap for the 96 coefficients
                colors = plt.cm.viridis(np.linspace(0, 1, 96))
                
                # Plot each coefficient as a line
                for i in range(96):
                    ax1.plot(timestamps_array, nash_coeffs[:, i], color=colors[i], 
                            linewidth=1, alpha=0.7)
                
                ax1.set_ylabel('Coefficient Value')
                ax1.set_title(f'Nash Coefficients Evolution (w1-w96) - vs {opponent} timestep {timestep_range}')
                ax1.grid(True, alpha=0.3)
                
                # Add colorbar to show which coefficient corresponds to which color
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=96))
                sm.set_array([])
                cbar1 = plt.colorbar(sm, ax=ax1)
                cbar1.set_label('Coefficient Index (w1-w96)')
            
            # 2. Time coefficient (w97) as a prominent line plot
            ax2.plot(timestamps_array, time_coeffs, 'red', linewidth=3, marker='o', 
                    markersize=6, alpha=0.9, label='Time Coefficient (w97)')
            ax2.set_ylabel('Time Coefficient (w97)')
            ax2.set_xlabel('Negotiation Time')
            ax2.set_title(f'Time Coefficient Evolution (w97) - vs {opponent} timestep {timestep_range}')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"{str(domain_plots_dir)}/coefficients_vs_{opponent}_{timestep_range}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()  # Don't show, just save
            
            print(f"📊 Domain {domain} coefficients plot saved as {filename}")
            
            # Print summary statistics (EXACT CODE from coefficient_tracker.py)
            print(f"\n📈 Coefficient Statistics:")
            print(f"   Total steps recorded: {len(timestamps)}")
            print(f"   Time coefficient range: [{np.min(time_coeffs):.3f}, {np.max(time_coeffs):.3f}]")
            print(f"   Time coefficient final: {time_coeffs[-1]:.3f}")
            print(f"   Nash coefficients range: [{np.min(nash_coeffs):.3f}, {np.max(nash_coeffs):.3f}]")
            print(f"   Most active Nash coefficients: {np.argsort(np.std(nash_coeffs, axis=0))[-5:] + 1}")  # +1 for 1-indexed  
    
    def plot_all_dynamics_after_timestep(self, min_timestep: int = 0, max_timestep: int = None):
        """Plot all dynamics for data within a timestep range in one comprehensive view."""
        if max_timestep is None:
            range_desc = f"timestep {min_timestep}+"
        else:
            range_desc = f"timestep {min_timestep}-{max_timestep}"
        
        print(f"🎯 Generating comprehensive visualization for {range_desc}")
        print("=" * 60)
        
        try:
            self.plot_episode_rewards(min_timestep, max_timestep)
        except Exception as e:
            print(f"❌ Error plotting rewards: {e}")
        
        try:
            self.plot_episode_utilities(min_timestep, max_timestep)
        except Exception as e:
            print(f"❌ Error plotting utilities: {e}")
        
        try:
            self.plot_episode_strategies(min_timestep, max_timestep)
        except Exception as e:
            print(f"❌ Error plotting strategies: {e}")
        
        try:
            self.plot_episode_coefficients(min_timestep, max_timestep)
        except Exception as e:
            print(f"❌ Error plotting coefficients: {e}")
        
        print("=" * 60)
        print(f"✅ Comprehensive visualization complete for {range_desc}")
    
    def list_available_timesteps(self):
        """List all available timesteps across all data types."""
        print("📊 Available Timesteps Summary:")
        print("=" * 40)
        
        data_types = [
            ("Rewards", self.reward_dir / "global_episode_dynamics.csv"),
            ("Utilities", self.utility_dir / "global_utility_dynamics.csv"), 
            ("Strategies", self.strategy_dir / "global_strategy_dynamics.csv"),
            ("Coefficients", self.coefficient_dir / "global_coefficient_dynamics.csv")
        ]
        
        for data_type, csv_path in data_types:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                timesteps = sorted(df['timestep'].unique())
                print(f"{data_type:12}: {len(timesteps)} timesteps - Range: {min(timesteps)} to {max(timesteps)}")
                if len(timesteps) <= 10:
                    print(f"                All timesteps: {timesteps}")
                else:
                    print(f"                First 5: {timesteps[:5]}, Last 5: {timesteps[-5:]}")
            else:
                print(f"{data_type:12}: ❌ No data file found")
    
    def list_available_episodes(self):
        """List all available episodes across all data types."""
        print("📊 Available Episodes Summary:")
        print("=" * 40)
        
        data_types = [
            ("Rewards", self.reward_dir / "global_episode_dynamics.csv"),
            ("Utilities", self.utility_dir / "global_utility_dynamics.csv"), 
            ("Strategies", self.strategy_dir / "global_strategy_dynamics.csv"),
            ("Coefficients", self.coefficient_dir / "global_coefficient_dynamics.csv")
        ]
        
        for data_type, csv_path in data_types:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                episodes = sorted(df['episode'].unique())
                print(f"{data_type:12}: {len(episodes)} episodes - {episodes}")
            else:
                print(f"{data_type:12}: ❌ No data file found")


def main():
    parser = argparse.ArgumentParser(description='Visualize episode dynamics within timestep ranges')
    parser.add_argument("--run_id", type=str, default="", help="Run ID to visualize")
    parser.add_argument('--min-timestep', type=int, default=0, help='Minimum timestep threshold to include in visualization')
    parser.add_argument('--max-timestep', type=int, help='Maximum timestep threshold to include in visualization (optional)')
    parser.add_argument('--timestep', type=int, help='(Deprecated) Use --min-timestep instead')
    parser.add_argument('--episode', type=int, help='(Deprecated) Episode number - use timestep range instead')
    parser.add_argument('--log_dir', type=str, default='episode_logs', help='Log directory path')
    parser.add_argument('--type', type=str, choices=['rewards', 'utilities', 'strategies', 'coefficients', 'all'], 
                       default='all', help='Type of visualization')
    parser.add_argument('--list', action='store_true', help='List available timesteps')
    parser.add_argument('--list-episodes', action='store_true', help='List available episodes (deprecated)')
    
    args = parser.parse_args()
    
    visualizer = EpisodeVisualizer(args.log_dir, args.run_id)
    
    if args.list:
        visualizer.list_available_timesteps()
        return
    
    if args.list_episodes:
        visualizer.list_available_episodes()
        return
    
    # Handle deprecated parameters
    if args.episode is not None:
        print("⚠️ Warning: --episode is deprecated. Use --min-timestep and --max-timestep instead for better cross-environment comparison.")
        min_timestep = 0
        max_timestep = None
    elif args.timestep is not None:
        print("⚠️ Warning: --timestep is deprecated. Use --min-timestep and --max-timestep instead.")
        min_timestep = args.timestep
        max_timestep = args.max_timestep
    else:
        min_timestep = args.min_timestep
        max_timestep = args.max_timestep
    
    # Validate timestep range
    if max_timestep is not None and max_timestep < min_timestep:
        print(f"❌ Error: max-timestep ({max_timestep}) cannot be less than min-timestep ({min_timestep})")
        return
    
    if args.type == 'rewards':
        visualizer.plot_episode_rewards(min_timestep, max_timestep)
    elif args.type == 'utilities':
        visualizer.plot_episode_utilities(min_timestep, max_timestep)
    elif args.type == 'strategies':
        visualizer.plot_episode_strategies(min_timestep, max_timestep)
    elif args.type == 'coefficients':
        visualizer.plot_episode_coefficients(min_timestep, max_timestep)
    elif args.type == 'all':
        visualizer.plot_all_dynamics_after_timestep(min_timestep, max_timestep)


if __name__ == "__main__":
    main()
