#!/usr/bin/env python3
"""
Simple visualizer for Pareto front ball boundaries from ParetoFrontLogger data.
Plots boxes for each update period (deadline // 80 rounds).
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def visualize_pareto_front_logs(session_path_1: str, session_path_2: str):
    """
    Visualize Pareto front ball boundaries as boxes for each update period.

    Args:
        session_path_1: Path to first session Excel file
        session_path_2: Path to second session Excel file
    """
    print(f"\nLoading sessions:")
    print(f"  Session 1: {session_path_1}")
    print(f"  Session 2: {session_path_2}")

    # Load both sessions
    df1 = pd.read_excel(session_path_1, sheet_name='Session')
    df2 = pd.read_excel(session_path_2, sheet_name='Session')

    # Get deadline and calculate update frequency
    deadline = len(df1)  # Assuming both have same deadline
    update_frequency = deadline // 80
    print(f"\nDeadline: {deadline} rounds")
    print(f"Update frequency: every {update_frequency} rounds")
    print(f"Total periods: {80}")

    # Create figure - 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Colors for boxes
    colors = plt.cm.viridis(range(80))

    # Function to draw boxes for a session
    def draw_boxes(ax, df, agent, title):
        # For each update period
        for period_idx in range(80):
            start_round = period_idx * update_frequency
            end_round = min((period_idx + 1) * update_frequency, len(df))

            if start_round >= len(df):
                break

            # Get data for this period
            period_df = df.iloc[start_round:end_round]

            # Get ball boundaries for this period
            if agent == 'A':
                min_col = 'BallMinUtilityA'
                max_col = 'BallMaxUtilityA'
            else:
                min_col = 'BallMinUtilityB'
                max_col = 'BallMaxUtilityB'

            # Skip if columns don't exist or all NaN
            if min_col not in period_df.columns or period_df[min_col].isna().all():
                continue

            # Get min and max for the box
            min_util = period_df[min_col].min()
            max_util = period_df[max_col].max()

            if pd.isna(min_util) or pd.isna(max_util):
                continue

            # Time coordinates (normalized)
            start_time = start_round / deadline
            end_time = end_round / deadline

            # Draw box
            width = end_time - start_time
            height = max_util - min_util

            rect = Rectangle((start_time, min_util), width, height,
                           facecolor=colors[period_idx], alpha=0.4,
                           edgecolor=colors[period_idx], linewidth=2)
            ax.add_patch(rect)

        # Formatting
        ax.set_xlabel('Negotiation Time (0 = start, 1 = deadline)', fontsize=12)
        ax.set_ylabel(f'Agent {agent} Utility', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.05)

    # Draw boxes for each session and agent
    session1_name = Path(session_path_1).stem
    session2_name = Path(session_path_2).stem

    draw_boxes(ax1, df1, 'A', f'Session 1 - Agent A\n({session1_name})')
    draw_boxes(ax2, df1, 'B', f'Session 1 - Agent B')
    draw_boxes(ax3, df2, 'A', f'Session 2 - Agent A\n({session2_name})')
    draw_boxes(ax4, df2, 'B', f'Session 2 - Agent B')

    # Main title
    fig.suptitle('Pareto Front Ball Boundaries (Boxes per Update Period)',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    output_dir = Path(session_path_1).parent
    output_path = output_dir / 'pareto_boxes_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    plt.show()


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python visualize_pareto_front.py <session1.xlsx> <session2.xlsx>")
        print("\nExample:")
        print("  python visualize_pareto_front.py session1.xlsx session2.xlsx")
        sys.exit(1)

    session_path_1 = sys.argv[1]
    session_path_2 = sys.argv[2]

    # Check if files exist
    if not Path(session_path_1).exists():
        print(f"Error: File not found: {session_path_1}")
        sys.exit(1)

    if not Path(session_path_2).exists():
        print(f"Error: File not found: {session_path_2}")
        sys.exit(1)

    visualize_pareto_front_logs(session_path_1, session_path_2)


if __name__ == "__main__":
    main()