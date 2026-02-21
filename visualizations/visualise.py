#!/usr/bin/env python3
"""
Session Plotter - Visualization tool for negotiation session offers.

This module provides functionality to plot all offers made during a negotiation session
by analyzing the session Excel log files. Creates two-subplot visualizations showing
both our utility and opponent utility perspectives.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import json


def parse_bid_content(bid_content: str) -> dict:
    """
    Parse bid content string to dictionary.
    
    Args:
        bid_content: String representation of bid content
        
    Returns:
        Dictionary representation of the bid
    """
    if pd.isna(bid_content) or bid_content is None:
        return {}
    
    try:
        # Handle string representation of dictionary
        if isinstance(bid_content, str):
            # Replace single quotes with double quotes for valid JSON
            bid_content = bid_content.replace("'", '"')
            return json.loads(bid_content)
        else:
            return {}
    except (json.JSONDecodeError, ValueError):
        return {}


def plot_session_offers(session_file_path: str, output_dir: Optional[str] = None) -> None:
    """
    Plot all offers made during a negotiation session from Excel log file.
    
    Creates two subplots:
    1. Our utility perspective - shows both our offers and opponent offers in our utility space
    2. Opponent utility perspective - shows both our offers and opponent offers in opponent utility space
    
    Args:
        session_file_path: Path to the session Excel file
        output_dir: Directory to save the plot (optional, defaults to same directory as session file)
    """
    
    # Extract session info from filename
    filename = os.path.basename(session_file_path)
    name_parts = filename.replace('.xlsx', '').split('_')
    
    if len(name_parts) >= 3:
        agent_a = name_parts[0]
        agent_b = name_parts[1] 
        domain = name_parts[2]
    else:
        agent_a = "AgentA"
        agent_b = "AgentB"
        domain = "Unknown"
    
    try:
        # Load session data from Excel file
        print(f"📖 Loading session data from {filename}...")
        
        # Read the Session sheet
        df = pd.read_excel(session_file_path, sheet_name='Session')
        
        if df.empty:
            print(f"⚠️  No data found in session file {filename}")
            return
            
        print(f"   Found {len(df)} rows of negotiation data")
        
        # Extract data for plotting
        our_times = []
        our_utilities = []
        our_opponent_utilities = []  # Opponent's utility for our offers
        
        opponent_times = []
        opponent_utilities = []  # Our utility for opponent's offers  
        opponent_own_utilities = []  # Opponent's utility for their own offers
        
        # For Nash utility calculation
        all_times = []
        nash_utilities = []
        
        # Process each row
        for _, row in df.iterrows():
            time_progress = float(row['Time'])
            who = row['Who']
            action = row['Action']
            
            # Skip accept actions as they don't represent new offers
            if action == 'Accept':
                continue
                
            agent_a_utility = float(row['AgentAUtility'])
            agent_b_utility = float(row['AgentBUtility'])
            
            if who == 'A':  # Agent A's offer (our offers)
                our_times.append(time_progress)
                our_utilities.append(agent_a_utility)  # Our utility for our bid
                our_opponent_utilities.append(agent_b_utility)  # Opponent's utility for our bid
                
                # Calculate Nash Product
                nash_utility = agent_a_utility * agent_b_utility
                all_times.append(time_progress)
                nash_utilities.append(nash_utility)
                
            elif who == 'B':  # Agent B's offer (opponent offers)
                opponent_times.append(time_progress)
                opponent_utilities.append(agent_a_utility)  # Our utility for opponent's bid
                opponent_own_utilities.append(agent_b_utility)  # Opponent's utility for their own bid
                
                # Calculate Nash Product
                nash_utility = agent_a_utility * agent_b_utility
                all_times.append(time_progress)
                nash_utilities.append(nash_utility)
        
        # Find final agreement details
        final_time = 1.0
        final_our_utility = 0.0
        final_opp_utility = 0.0
        agreement_reached = False
        
        # Check if there was an acceptance
        accept_rows = df[df['Action'] == 'Accept']
        if not accept_rows.empty:
            final_row = accept_rows.iloc[-1]
            final_time = float(final_row['Time'])
            final_our_utility = float(final_row['AgentAUtility'])
            final_opp_utility = float(final_row['AgentBUtility'])
            agreement_reached = True
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # === LEFT SUBPLOT: Our Utility Perspective ===
        # Plot our offers
        if our_times:
            ax1.scatter(our_times, our_utilities, c='blue', s=50, alpha=0.7, 
                       label=f'{agent_a} Offers ({len(our_times)} total)', marker='o')
        
        # Plot opponent offers (from our utility perspective)
        if opponent_times:
            ax1.scatter(opponent_times, opponent_utilities, c='red', s=50, alpha=0.7, 
                       label=f'{agent_b} Offers ({len(opponent_times)} total)', marker='s')
        
        # Plot Nash utility as a line
        if all_times and nash_utilities:
            # Sort by time for proper line plotting
            sorted_data = sorted(zip(all_times, nash_utilities))
            sorted_times, sorted_nash = zip(*sorted_data)
            ax1.plot(sorted_times, sorted_nash, 'purple', linewidth=2, alpha=0.8, 
                    label=f'Nash Utility', linestyle='-')
        
        # Add final agreement marker
        agreement_status = "Agreement" if agreement_reached else "No Deal"
        marker_color = 'green' if agreement_reached else 'gray'
        ax1.scatter([final_time], [final_our_utility], c=marker_color, s=200, alpha=0.9,
                   label=f'Final {agreement_status} ({agent_a}: {final_our_utility:.3f}, {agent_b}: {final_opp_utility:.3f})', 
                   marker='*', edgecolors='darkgreen' if agreement_reached else 'darkgray', linewidth=2)
        
        # Styling for left subplot
        ax1.set_xlabel('Negotiation Time (0 = start, 1 = deadline)')
        ax1.set_ylabel(f'{agent_a} Utility')
        ax1.set_title(f'{agent_a} Utility Perspective vs {agent_b} in {domain}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 1)
        
        # === RIGHT SUBPLOT: Opponent's Utility Perspective ===
        # Plot our offers (from opponent's utility perspective)
        if our_times:
            ax2.scatter(our_times, our_opponent_utilities, c='blue', s=50, alpha=0.7, 
                       marker='o')
        
        # Plot opponent offers (from their own utility perspective)
        if opponent_times:
            ax2.scatter(opponent_times, opponent_own_utilities, c='red', s=50, alpha=0.7, 
                       marker='s')
        
        # Plot Nash utility as a line
        if all_times and nash_utilities:
            ax2.plot(sorted_times, sorted_nash, 'purple', linewidth=2, alpha=0.8, 
                    linestyle='-')
        
        # Add final agreement marker
        ax2.scatter([final_time], [final_opp_utility], c=marker_color, s=200, alpha=0.9,
                   marker='*', edgecolors='darkgreen' if agreement_reached else 'darkgray', linewidth=2)
        
        # Styling for right subplot
        ax2.set_xlabel('Negotiation Time (0 = start, 1 = deadline)')
        ax2.set_ylabel(f'{agent_b} Utility')
        ax2.set_title(f'{agent_b} Utility Perspective vs {agent_a} in {domain}')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        
        # Create a single unified legend for the entire figure
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.85, 0.85), loc='upper left')
        
        # Adjust layout to make room for legend and stats
        plt.subplots_adjust(right=0.75)
        
        # Add comprehensive statistics
        stats_text = ""
        
        # Our agent statistics
        if our_times:
            avg_our_utility = np.mean(our_utilities)
            last_offer_utility = our_utilities[-1] if our_utilities else 0
            avg_opp_utility_for_our_offers = np.mean(our_opponent_utilities) if our_opponent_utilities else 0
            
            stats_text += f"{agent_a.upper()}:\n"
            stats_text += f"• Avg utility: {avg_our_utility:.3f}\n"
            stats_text += f"• Last offer: {last_offer_utility:.3f}\n"
            stats_text += f"• Final agreement: {final_our_utility:.3f}\n"
            stats_text += f"• Avg opp utility: {avg_opp_utility_for_our_offers:.3f}\n"
            
        # Opponent statistics  
        if opponent_times:
            avg_our_utility_for_opp_offers = np.mean(opponent_utilities) if opponent_utilities else 0
            avg_opp_own_utility = np.mean(opponent_own_utilities) if opponent_own_utilities else 0
            last_opp_offer_utility = opponent_own_utilities[-1] if opponent_own_utilities else 0
            
            stats_text += f"\n{agent_b.upper()}:\n"
            stats_text += f"• Avg utility: {avg_opp_own_utility:.3f}\n"
            stats_text += f"• Last offer: {last_opp_offer_utility:.3f}\n"
            stats_text += f"• Final agreement: {final_opp_utility:.3f}\n"
            stats_text += f"• Avg our utility: {avg_our_utility_for_opp_offers:.3f}\n"
        
        # Nash statistics
        if nash_utilities:
            avg_nash = np.mean(nash_utilities)
            final_nash = final_our_utility * final_opp_utility
            stats_text += f"\nNASH:\n"
            stats_text += f"• Avg: {avg_nash:.3f}\n"
            stats_text += f"• Final: {final_nash:.3f}\n"
        
        # Negotiation summary
        stats_text += f"\nSUMMARY:\n"
        stats_text += f"• Total offers: {len(our_times) + len(opponent_times)}\n"
        stats_text += f"• Negotiation length: {len(df)} rounds\n"
        stats_text += f"• Result: {agreement_status}\n"
        
        # Add the statistics text box
        if stats_text:
            fig.text(0.85, 0.65, stats_text, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, fontfamily='monospace')
        
        # Save the negotiation plot
        if output_dir is None:
            output_dir = os.path.dirname(session_file_path)
        
        # Create negotiation plots directory
        negotiation_plots_dir = os.path.join(output_dir, "negotiation_plots")
        os.makedirs(negotiation_plots_dir, exist_ok=True)
        
        plot_filename = os.path.join(negotiation_plots_dir, f'negotiation_{agent_a}_vs_{agent_b}_{domain}.png')
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        
        print(f"📊 Negotiation plot saved as {plot_filename}")
        print(f"   {agent_a} offers: {len(our_times)}")
        print(f"   {agent_b} offers: {len(opponent_times)}")
        print(f"   Final result: {agreement_status}")
        
        if agreement_reached:
            print(f"   Final utilities: {agent_a}={final_our_utility:.3f}, {agent_b}={final_opp_utility:.3f}")
        
        # Print additional statistics
        if our_times and our_opponent_utilities:
            print(f"   {agent_a} avg offer utility: {np.mean(our_utilities):.3f}")
            print(f"   {agent_b} avg utility from {agent_a} offers: {np.mean(our_opponent_utilities):.3f}")
        
        if opponent_times and opponent_own_utilities:
            print(f"   {agent_b} avg offer utility: {np.mean(opponent_own_utilities):.3f}")
            print(f"   {agent_a} avg utility from {agent_b} offers: {np.mean(opponent_utilities):.3f}")
        
        if nash_utilities:
            print(f"   Average Nash utility: {np.mean(nash_utilities):.3f}")
        
        # Close the negotiation plot figure to free memory
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ Error plotting session offers for {filename}: {e}")
        raise


def plot_multiple_sessions(sessions_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Plot offers for all session files in a directory.
    
    Args:
        sessions_dir: Directory containing session Excel files
        output_dir: Directory to save plots (optional, defaults to sessions_dir)
    """
    
    if output_dir is None:
        output_dir = sessions_dir
    
    # Find all Excel files in the directory
    excel_files = [f for f in os.listdir(sessions_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        print(f"⚠️  No Excel files found in {sessions_dir}")
        return
    
    print(f"🔥 Found {len(excel_files)} session files to plot")
    print("=" * 60)
    
    for excel_file in excel_files:
        session_file_path = os.path.join(sessions_dir, excel_file)
        plot_session_offers(session_file_path, output_dir)
        print()  # Add spacing between plots
    
    print(f"✅ All session plots completed! Plots saved in {output_dir}")


def plot_agent_box_plots(sessions_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Create box plots for all agents showing self utility, opponent utility, and product score.

    Creates three separate plots:
    1. Self utility distribution per agent
    2. Opponent utility distribution per agent
    3. Product score (agent_a_utility × agent_b_utility) distribution per agent

    Args:
        sessions_dir: Directory containing session Excel files
        output_dir: Directory to save plots (optional, defaults to sessions_dir)
    """

    if output_dir is None:
        output_dir = sessions_dir

    # Find all Excel files in the directory
    excel_files = [f for f in os.listdir(sessions_dir) if f.endswith('.xlsx')]

    if not excel_files:
        print(f"⚠️  No Excel files found in {sessions_dir}")
        return

    print(f"📊 Creating box plots from {len(excel_files)} session files")
    print("=" * 60)

    # Data structures to collect metrics per agent
    agent_self_utilities = {}
    agent_opponent_utilities = {}
    agent_product_scores = {}
    agent_rounds = {}

    # Process each session file
    for excel_file in excel_files:
        session_file_path = os.path.join(sessions_dir, excel_file)
        filename = os.path.basename(session_file_path)
        name_parts = filename.replace('.xlsx', '').split('_')

        if len(name_parts) >= 3:
            agent_a = name_parts[0]
            agent_b = name_parts[1]
        else:
            continue

        try:
            # Read the Session sheet
            df = pd.read_excel(session_file_path, sheet_name='Session')

            if df.empty:
                continue

            # Initialize agent data structures if needed
            if agent_a not in agent_self_utilities:
                agent_self_utilities[agent_a] = []
                agent_opponent_utilities[agent_a] = []
                agent_product_scores[agent_a] = []
                agent_rounds[agent_a] = []

            if agent_b not in agent_self_utilities:
                agent_self_utilities[agent_b] = []
                agent_opponent_utilities[agent_b] = []
                agent_product_scores[agent_b] = []
                agent_rounds[agent_b] = []

            # Get number of rounds
            num_rounds = len(df)

            # Check if there was an agreement
            accept_rows = df[df['Action'] == 'Accept']
            if not accept_rows.empty:
                # Agreement reached
                final_row = accept_rows.iloc[-1]
                agent_a_utility = float(final_row['AgentAUtility'])
                agent_b_utility = float(final_row['AgentBUtility'])

                # Calculate product score
                product_score = agent_a_utility * agent_b_utility

                # Store metrics for both agents
                agent_self_utilities[agent_a].append(agent_a_utility)
                agent_opponent_utilities[agent_a].append(agent_b_utility)
                agent_product_scores[agent_a].append(product_score)
                agent_rounds[agent_a].append(num_rounds)

                agent_self_utilities[agent_b].append(agent_b_utility)
                agent_opponent_utilities[agent_b].append(agent_a_utility)
                agent_product_scores[agent_b].append(product_score)
                agent_rounds[agent_b].append(num_rounds)
            else:
                # No agreement - both agents get 0 utility
                agent_a_utility = 0.0
                agent_b_utility = 0.0
                product_score = 0.0

                # Store metrics for both agents
                agent_self_utilities[agent_a].append(agent_a_utility)
                agent_opponent_utilities[agent_a].append(agent_b_utility)
                agent_product_scores[agent_a].append(product_score)
                agent_rounds[agent_a].append(num_rounds)

                agent_self_utilities[agent_b].append(agent_b_utility)
                agent_opponent_utilities[agent_b].append(agent_a_utility)
                agent_product_scores[agent_b].append(product_score)
                agent_rounds[agent_b].append(num_rounds)

        except Exception as e:
            print(f"⚠️  Error processing {filename}: {e}")
            continue

    # Create box_plots directory
    box_plots_dir = os.path.join(output_dir, "box_plots")
    os.makedirs(box_plots_dir, exist_ok=True)

    # Get sorted agent names
    agent_names = sorted(agent_self_utilities.keys())

    if not agent_names:
        print("⚠️  No agent data found to plot")
        return

    # === PLOT 1: Self Utility ===
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    self_utility_data = [agent_self_utilities[agent] for agent in agent_names]

    bp1 = ax1.boxplot(self_utility_data, labels=agent_names, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('white')

    ax1.set_xlabel('Agent')
    ax1.set_ylabel('Self Utility')
    ax1.set_title('Self Utility Distribution by Agent')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot1_path = os.path.join(box_plots_dir, 'self_utility_boxplot.png')
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"📊 Self utility box plot saved: {plot1_path}")

    # === PLOT 2: Opponent Utility ===
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    opponent_utility_data = [agent_opponent_utilities[agent] for agent in agent_names]

    bp2 = ax2.boxplot(opponent_utility_data, labels=agent_names, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('white')

    ax2.set_xlabel('Agent')
    ax2.set_ylabel('Opponent Utility')
    ax2.set_title('Opponent Utility Distribution by Agent')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot2_path = os.path.join(box_plots_dir, 'opponent_utility_boxplot.png')
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"📊 Opponent utility box plot saved: {plot2_path}")

    # === PLOT 3: Product Score ===
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    product_score_data = [agent_product_scores[agent] for agent in agent_names]

    bp3 = ax3.boxplot(product_score_data, labels=agent_names, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('white')

    ax3.set_xlabel('Agent')
    ax3.set_ylabel('Product Score (Agent A × Agent B Utility)')
    ax3.set_title('Product Score Distribution by Agent')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot3_path = os.path.join(box_plots_dir, 'product_score_boxplot.png')
    fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"📊 Product score box plot saved: {plot3_path}")

    # === AGREEMENTS ONLY PLOTS ===
    # Filter data to only include sessions with agreements (where utilities are not both 0)
    agent_self_utilities_agreements = {}
    agent_opponent_utilities_agreements = {}
    agent_product_scores_agreements = {}
    agent_rounds_agreements = {}

    for agent in agent_names:
        self_utils = []
        opp_utils = []
        products = []
        rounds = []

        for i in range(len(agent_self_utilities[agent])):
            # Include only if there was an agreement (not both utilities = 0)
            if agent_self_utilities[agent][i] != 0.0 or agent_opponent_utilities[agent][i] != 0.0:
                self_utils.append(agent_self_utilities[agent][i])
                opp_utils.append(agent_opponent_utilities[agent][i])
                products.append(agent_product_scores[agent][i])
                rounds.append(agent_rounds[agent][i])

        agent_self_utilities_agreements[agent] = self_utils
        agent_opponent_utilities_agreements[agent] = opp_utils
        agent_product_scores_agreements[agent] = products
        agent_rounds_agreements[agent] = rounds

    # === PLOT 4: Self Utility (Agreements Only) ===
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    self_utility_data_agreements = [agent_self_utilities_agreements[agent] for agent in agent_names]

    bp4 = ax4.boxplot(self_utility_data_agreements, labels=agent_names, patch_artist=True)
    for patch in bp4['boxes']:
        patch.set_facecolor('white')

    ax4.set_xlabel('Agent')
    ax4.set_ylabel('Self Utility')
    ax4.set_title('Self Utility Distribution by Agent (Agreements Only)')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot4_path = os.path.join(box_plots_dir, 'self_utility_agreements_boxplot.png')
    fig4.savefig(plot4_path, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"📊 Self utility (agreements only) box plot saved: {plot4_path}")

    # === PLOT 5: Opponent Utility (Agreements Only) ===
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    opponent_utility_data_agreements = [agent_opponent_utilities_agreements[agent] for agent in agent_names]

    bp5 = ax5.boxplot(opponent_utility_data_agreements, labels=agent_names, patch_artist=True)
    for patch in bp5['boxes']:
        patch.set_facecolor('white')

    ax5.set_xlabel('Agent')
    ax5.set_ylabel('Opponent Utility')
    ax5.set_title('Opponent Utility Distribution by Agent (Agreements Only)')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot5_path = os.path.join(box_plots_dir, 'opponent_utility_agreements_boxplot.png')
    fig5.savefig(plot5_path, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"📊 Opponent utility (agreements only) box plot saved: {plot5_path}")

    # === PLOT 6: Product Score (Agreements Only) ===
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    product_score_data_agreements = [agent_product_scores_agreements[agent] for agent in agent_names]

    bp6 = ax6.boxplot(product_score_data_agreements, labels=agent_names, patch_artist=True)
    for patch in bp6['boxes']:
        patch.set_facecolor('white')

    ax6.set_xlabel('Agent')
    ax6.set_ylabel('Product Score (Agent A × Agent B Utility)')
    ax6.set_title('Product Score Distribution by Agent (Agreements Only)')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot6_path = os.path.join(box_plots_dir, 'product_score_agreements_boxplot.png')
    fig6.savefig(plot6_path, dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"📊 Product score (agreements only) box plot saved: {plot6_path}")

    # === PLOT 7: Rounds (All Sessions) ===
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    rounds_data = [agent_rounds[agent] for agent in agent_names]

    bp7 = ax7.boxplot(rounds_data, labels=agent_names, patch_artist=True)
    for patch in bp7['boxes']:
        patch.set_facecolor('white')

    ax7.set_xlabel('Agent')
    ax7.set_ylabel('Number of Rounds')
    ax7.set_title('Number of Rounds Distribution by Agent')
    ax7.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot7_path = os.path.join(box_plots_dir, 'rounds_boxplot.png')
    fig7.savefig(plot7_path, dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print(f"📊 Rounds box plot saved: {plot7_path}")

    # === PLOT 8: Rounds (Agreements Only) ===
    fig8, ax8 = plt.subplots(figsize=(12, 8))
    rounds_data_agreements = [agent_rounds_agreements[agent] for agent in agent_names]

    bp8 = ax8.boxplot(rounds_data_agreements, labels=agent_names, patch_artist=True)
    for patch in bp8['boxes']:
        patch.set_facecolor('white')

    ax8.set_xlabel('Agent')
    ax8.set_ylabel('Number of Rounds')
    ax8.set_title('Number of Rounds Distribution by Agent (Agreements Only)')
    ax8.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plot8_path = os.path.join(box_plots_dir, 'rounds_agreements_boxplot.png')
    fig8.savefig(plot8_path, dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print(f"📊 Rounds (agreements only) box plot saved: {plot8_path}")

    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print("=" * 60)
    for agent in agent_names:
        n_sessions = len(agent_self_utilities[agent])
        avg_self = np.mean(agent_self_utilities[agent]) if agent_self_utilities[agent] else 0
        avg_opp = np.mean(agent_opponent_utilities[agent]) if agent_opponent_utilities[agent] else 0
        avg_product = np.mean(agent_product_scores[agent]) if agent_product_scores[agent] else 0

        print(f"{agent}:")
        print(f"  Sessions: {n_sessions}")
        print(f"  Avg self utility: {avg_self:.3f}")
        print(f"  Avg opponent utility: {avg_opp:.3f}")
        print(f"  Avg product score: {avg_product:.3f}")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python session_plotter.py <sessions_directory>")
        print("Examples:")
        print("  python session_plotter.py results/test/2025-08-31_10-00-37/sessions/")
        print("  python session_plotter.py results/tournament_2024/sessions/")
        print("")
        print("This will create plots in: plots/<sessions_path>/")
        sys.exit(1)

    sessions_path = sys.argv[1]

    if not os.path.isdir(sessions_path):
        print(f"❌ Sessions directory not found: {sessions_path}")
        sys.exit(1)

    # Create output directory structure: plots/sessions_path
    # Extract the relative path from sessions_path for the plots directory
    sessions_path_normalized = os.path.normpath(sessions_path)
    output_dir = os.path.join("plots", sessions_path_normalized)

    print(f"🔥 Processing sessions from: {sessions_path}")
    print(f"📁 Plots will be saved to: {output_dir}")
    print("=" * 60)

    # Plot all sessions in the directory
    plot_multiple_sessions(sessions_path, output_dir)

    # Create box plots
    print("\n" + "=" * 60)
    print("Creating agent box plots...")
    print("=" * 60)
    plot_agent_box_plots(sessions_path, output_dir)
