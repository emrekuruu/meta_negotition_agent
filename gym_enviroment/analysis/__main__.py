#!/usr/bin/env python3
"""
Evaluation script for trained RL artifacts.
Loads a specific model artifact and evaluates it without training.
"""

import os 
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from stable_baselines3 import PPO
from typing import Dict, List

from ..env import NegotiationEnv
from ..agent.main_strategy import MainStrategy
from ..config.config import config
from .coefficient_tracker import CoefficientTracker
from .strategy_tracker import StrategyTracker

# Configuration
DEVICE = "cuda"
os.environ["WANDB_MODE"] = "online"
ARTIFACT_NAME = "ppo_negotiator_fiery-armadillo-1:v12"

def load_artifact_model(device: str = "cpu") -> PPO:
    """Load a trained model from W&B artifact."""
    global ARTIFACT_NAME

    # Initialize wandb in online mode for artifact download
    run = wandb.init(mode="online")
    artifact = run.use_artifact(f'emre-kuru-zye-in-niversitesi/negotiation-rl/{ARTIFACT_NAME}', type='model')
    artifact_dir = artifact.download()
    
    # Debug: List files in artifact directory
    import os
    print(f"Artifact directory: {artifact_dir}")
    print(f"Files in artifact directory: {os.listdir(artifact_dir)}")
    
    # Find the zip file (it has a random name like tmpkkomq5wd.zip)
    files = os.listdir(artifact_dir)
    zip_files = [f for f in files if f.endswith('.zip')]
    
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {artifact_dir}. Available files: {files}")
    
    # Use the first (and likely only) zip file
    model_path = f"{artifact_dir}/{zip_files[0]}"
    print(f"Using model file: {model_path}")
    
    wandb.finish()
    
    # Load the PPO model
    model = PPO.load(model_path, device=device)
    print(f"✅ Model loaded successfully from {model_path}")
    
    return model


def plot_actual_offers(env: NegotiationEnv, opponent_name: str, domain: str):
    """Plot the actual offers made during the negotiation as a scatter plot."""
    try:
        
        # Get offer history from our agent
        if not hasattr(env.our_agent, 'offer_history') or not env.our_agent.offer_history:
            print(f"⚠️  No offer history available for opponent {opponent_name}")
            return
        
        offer_history = env.our_agent.offer_history
        
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
        
        for offer_point in offer_history:
            time_progress = offer_point.t
            bid = offer_point.bid
            
            if offer_point.who == 1:  # Our offer
                our_times.append(time_progress)
                our_utilities.append(bid.utility)  # Our utility for our bid
                
                # Calculate opponent's utility for our bid
                opponent_utility_for_our_bid = env.opponent_preference.get_utility(bid)
                our_opponent_utilities.append(opponent_utility_for_our_bid)
                
                # Calculate Nash Product: (our_utility * opponent_utility)
                nash_utility = (bid.utility * opponent_utility_for_our_bid) 
                all_times.append(time_progress)
                nash_utilities.append(nash_utility)
                
            elif offer_point.who == 0:  # Opponent offer
                opponent_times.append(time_progress)
                # Calculate our utility for opponent's bid
                our_utility_for_opp_bid = env.our_preference.get_utility(bid)
                opponent_utilities.append(our_utility_for_opp_bid)
                
                # Calculate opponent's utility for their own bid
                opponent_utility_for_opp_bid = env.opponent_preference.get_utility(bid)
                opponent_own_utilities.append(opponent_utility_for_opp_bid)
                
                # Calculate Nash Product: (our_utility * opponent_utility)
                nash_utility = (our_utility_for_opp_bid * opponent_utility_for_opp_bid)
                all_times.append(time_progress)
                nash_utilities.append(nash_utility)
        
        # Create subplots - two perspectives
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Find the actual final agreement time and bid
        final_time = 1.0  # Default to deadline
        final_bid = None
        if all_times:
            final_time = max(all_times)  # Last offer time
            # Get the last bid from either our offers or opponent offers
            if our_times and opponent_times:
                if our_times[-1] >= opponent_times[-1]:
                    # Our last offer was later
                    final_bid = offer_history[-1].bid if offer_history and offer_history[-1].who == 1 else None
                else:
                    # Opponent's last offer was later
                    final_bid = offer_history[-1].bid if offer_history and offer_history[-1].who == 0 else None
            elif our_times:
                final_bid = offer_history[-1].bid if offer_history and offer_history[-1].who == 1 else None
            elif opponent_times:
                final_bid = offer_history[-1].bid if offer_history and offer_history[-1].who == 0 else None
        
        # Calculate opponent's utility for final agreement (needed for legend)
        final_opp_agreement_utility = 0
        if final_bid:
            final_opp_agreement_utility = env.opponent_preference.get_utility(final_bid)
        else:
            # If no bid found, try to calculate from environment's final utility
            # This is a fallback - the opponent utility might not be exact
            final_opp_agreement_utility = 1.0 - env.final_utility if env.final_utility > 0 else 0
        
        # === LEFT SUBPLOT: Our Utility Perspective ===
        # Plot our offers
        if our_times:
            ax1.scatter(our_times, our_utilities, c='blue', s=50, alpha=0.7, 
                       label=f'Our Offers ({len(our_times)} total)', marker='o')
        
        # Plot opponent offers (from our utility perspective)
        if opponent_times:
            ax1.scatter(opponent_times, opponent_utilities, c='red', s=50, alpha=0.7, 
                       label=f'Opponent Offers ({len(opponent_times)} total)', marker='s')
        
        # Plot Nash utility as a line (same on both subplots)
        if all_times and nash_utilities:
            # Sort by time for proper line plotting
            sorted_data = sorted(zip(all_times, nash_utilities))
            sorted_times, sorted_nash = zip(*sorted_data)
            ax1.plot(sorted_times, sorted_nash, 'purple', linewidth=2, alpha=0.8, 
                    label=f'Nash Utility', linestyle='-')
        
        # Add final agreement marker (big green dot)
        ax1.scatter([final_time], [env.final_utility], c='green', s=200, alpha=0.9,
                   label=f'Final Agreement (Ours: {env.final_utility:.3f}, Opponents: {final_opp_agreement_utility:.3f})', marker='*', 
                   edgecolors='darkgreen', linewidth=2)
        
        # Styling for left subplot
        ax1.set_xlabel('Negotiation Time (0 = start, 1 = deadline)')
        ax1.set_ylabel('Our Utility')
        ax1.set_title(f'Our Utility Perspective vs {opponent_name}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 1)
        # No legend on left subplot
        
        # === RIGHT SUBPLOT: Opponent's Utility Perspective ===
        # Plot our offers (from opponent's utility perspective) - no label to avoid duplicate
        if our_times:
            ax2.scatter(our_times, our_opponent_utilities, c='blue', s=50, alpha=0.7, 
                       marker='o')
        
        # Plot opponent offers (from their own utility perspective) - no label to avoid duplicate
        if opponent_times:
            ax2.scatter(opponent_times, opponent_own_utilities, c='red', s=50, alpha=0.7, 
                       marker='s')
        
        # Plot Nash utility as a line (same as left subplot) - no label to avoid duplicate
        if all_times and nash_utilities:
            ax2.plot(sorted_times, sorted_nash, 'purple', linewidth=2, alpha=0.8, 
                    linestyle='-')
        
        # Add final agreement marker (big green dot) - no label to avoid duplicate
        ax2.scatter([final_time], [final_opp_agreement_utility], c='green', s=200, alpha=0.9,
                   marker='*', edgecolors='darkgreen', linewidth=2)
        
        # Styling for right subplot
        ax2.set_xlabel('Negotiation Time (0 = start, 1 = deadline)')
        ax2.set_ylabel('Opponent Utility')
        ax2.set_title(f'Opponent Utility Perspective vs {opponent_name}')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        # No legend on right subplot
        
        # Create a single unified legend for the entire figure
        # Get handles and labels from the left subplot (which has all the labels)
        handles, labels = ax1.get_legend_handles_labels()
        
        # Create figure-level legend positioned to the right of plots
        fig.legend(handles, labels, bbox_to_anchor=(0.85, 0.85), loc='upper left')
        
        # Adjust layout to make room for legend and stats on the right
        plt.subplots_adjust(right=0.75)
        
        # Add comprehensive statistics as text on the right side
        stats_text = ""
        
        # Our statistics
        if our_times:
            avg_our_utility = np.mean(our_utilities)
            last_offer_utility = our_utilities[-1] if our_utilities else 0
            avg_opp_utility_for_our_offers = np.mean(our_opponent_utilities) if our_opponent_utilities else 0
            final_opp_utility_for_our_offers = our_opponent_utilities[-1] if our_opponent_utilities else 0
            
            stats_text += f"OUR AGENT:\n"
            stats_text += f"• Avg utility: {avg_our_utility:.3f}\n"
            stats_text += f"• Last offer: {last_offer_utility:.3f}\n"
            stats_text += f"• Final agreement: {env.final_utility:.3f}\n"
            stats_text += f"• Avg opp utility: {avg_opp_utility_for_our_offers:.3f}\n"
            
        # Opponent statistics  
        if opponent_times:
            avg_our_utility_for_opp_offers = np.mean(opponent_utilities) if opponent_utilities else 0
            last_opp_offer_utility_for_us = opponent_utilities[-1] if opponent_utilities else 0
            avg_opp_own_utility = np.mean(opponent_own_utilities) if opponent_own_utilities else 0
            last_opp_offer_utility = opponent_own_utilities[-1] if opponent_own_utilities else 0
            
            # Use the already calculated final_opp_agreement_utility
            
            stats_text += f"\nOPPONENT:\n"
            stats_text += f"• Avg utility: {avg_opp_own_utility:.3f}\n"
            stats_text += f"• Last offer: {last_opp_offer_utility:.3f}\n"
            stats_text += f"• Final agreement: {final_opp_agreement_utility:.3f}\n"
            stats_text += f"• Avg our utility: {avg_our_utility_for_opp_offers:.3f}\n"
        
        # Nash statistics
        if nash_utilities:
            avg_nash = np.mean(nash_utilities)
            final_nash = nash_utilities[-1] if nash_utilities else 0
            stats_text += f"\nNASH:\n"
            stats_text += f"• Avg: {avg_nash:.3f}\n"
            stats_text += f"• Final: {final_nash:.3f}\n"
        
        # Add the statistics text box right under the legend
        if stats_text:
            fig.text(0.85, 0.65, stats_text, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, fontfamily='monospace')
        
        # Save the plot with domain and opponent name
        domain_dir = f'plots/domain{domain}'
        os.makedirs(domain_dir, exist_ok=True)
        plot_filename = f'{domain_dir}/negotiation_vs_{opponent_name}.png'
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Negotiation plot saved as {plot_filename}")
        print(f"   Total our offers: {len(our_times)}")
        print(f"   Total opponent offers: {len(opponent_times)}")
        print(f"   Final agreement utility (ours): {env.final_utility:.3f}")
        
        # Print additional statistics
        if our_times and our_opponent_utilities:
            print(f"   Our avg offer utility: {np.mean(our_utilities):.3f}")
            print(f"   Opponent avg utility from our offers: {np.mean(our_opponent_utilities):.3f}")
        
        if opponent_times and opponent_own_utilities:
            print(f"   Opponent avg offer utility (theirs): {np.mean(opponent_own_utilities):.3f}")
            print(f"   Our avg utility from opponent offers: {np.mean(opponent_utilities):.3f}")
        
        if nash_utilities:
            print(f"   Average Nash utility: {np.mean(nash_utilities):.3f}")
        
    except Exception as e:
        print(f"❌ Error plotting actual offers for opponent {opponent_name}: {e}")


def _run_single_negotiation(model: PPO, opponent_name: str, domain: str, render: bool = False) -> Dict:
    """Synchronous function to run a single negotiation."""
    
    domains = [domain]  

    opponents = [opponent_name]
    
    env = NegotiationEnv(
        our_agent_class=MainStrategy,
        domains=domains,
        deadline_round=1000,
        opponent_names=opponents,
        mode="oracle"
    )
    
    print(f"\n🤖 Starting negotiation vs {opponent_name}...")
    print("=" * 50)
    
    # Initialize trackers
    coeff_tracker = CoefficientTracker()
    strategy_tracker = StrategyTracker()
    
    # Run single negotiation
    obs, info = env.reset()
    done = False
    step_count = 0
    
    while not done:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Extract strategy index from action
        strategy_index = int(action[0]) if len(action) > 0 else 0
        
        # Record coefficients (only after we have enough history for RL)
        if step_count >= config.core['sequence_length']:  # After 96 steps
            # Calculate time as negotiation progress
            time_progress = step_count / config.environment["deadline_round"]
            coeff_tracker.record_coefficients(time_progress, action, strategy_index)
        
        # Record strategy selection at every step
        time_progress = step_count / config.environment["deadline_round"]
        strategy_tracker.record_strategy(time_progress, strategy_index)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        if render:
            env.render()
    
    # Collect results
    final_utility = env.final_utility
    is_agreement = final_utility > env.our_preference.reservation_value
    
    result = {
        "opponent": opponent_name,
        "final_utility": final_utility,
        "is_agreement": is_agreement,
        "step_count": step_count,
        "domain": info.get("domain", "unknown"),
        "reservation_value": env.our_preference.reservation_value,
        "env": env,  # Pass environment for plotting
        "coeff_tracker": coeff_tracker,  # Pass coefficient tracker
        "strategy_tracker": strategy_tracker  # Pass strategy tracker
    }
    
    return result


async def evaluate_single_opponent(model: PPO, opponent_name: str, domain: str, render: bool = False) -> Dict:
    """Evaluate the model against a single opponent using thread executor."""
    loop = asyncio.get_event_loop()
    
    # Run the blocking negotiation in a thread executor
    result = await loop.run_in_executor(
        None, _run_single_negotiation, model, opponent_name, domain, render
    )
    
    # Extract environment and trackers for plotting (done in main thread to avoid threading issues with matplotlib)
    env = result.pop("env")
    coeff_tracker = result.pop("coeff_tracker")
    strategy_tracker = result.pop("strategy_tracker")
    
    # Plot actual offers made during the negotiation
    print(f"\n📈 Plotting actual offers made during negotiation vs {opponent_name} in domain {domain}...")
    plot_actual_offers(env, opponent_name, domain)
    
    # Plot coefficient evolution
    print(f"\n📊 Plotting coefficient evolution vs {opponent_name} in domain {domain}...")
    coeff_tracker.plot_coefficients(opponent_name, result["domain"])
    
    # Plot strategy selection
    print(f"\n🎯 Plotting strategy selection vs {opponent_name} in domain {domain}...")
    strategy_tracker.plot_strategy_selection(opponent_name, result["domain"])
    
    # Print negotiation summary
    agreement_status = "✅ AGREEMENT" if result["is_agreement"] else "❌ NO DEAL"
    print(f"Result vs {opponent_name}: {agreement_status} | "
          f"Utility: {result['final_utility']:.3f} | "
          f"Length: {result['step_count']:2d} steps")
    
    env.close()
    return result


async def evaluate_model_parallel(model: PPO, opponents: List[str], domains: List[str], render: bool = False) -> Dict:
    """Evaluate the model against multiple opponents across multiple domains in parallel."""
    
    # Create all domain-opponent combinations
    combinations = [(domain, opponent) for domain in domains for opponent in opponents]
    
    print(f"\n🔥 Starting parallel evaluation against {len(combinations)} domain-opponent combinations...")
    print(f"   Domains: {len(domains)} | Opponents: {len(opponents)} | Total: {len(combinations)}")
    print("=" * 60)
    
    # Create tasks for parallel execution
    tasks = []
    for domain, opponent in combinations:
        task = evaluate_single_opponent(model, opponent, domain, render)
        tasks.append(task)
    
    # Run all evaluations in parallel
    results = await asyncio.gather(*tasks)
    
    # Aggregate results
    total_agreements = sum(1 for r in results if r["is_agreement"])
    total_combinations = len(results)
    utilities = [r["final_utility"] for r in results]
    step_counts = [r["step_count"] for r in results]
    
    summary = {
        "domains": domains,
        "opponents": opponents,
        "combinations": combinations,
        "results": results,
        "total_combinations": total_combinations,
        "agreements": total_agreements,
        "agreement_rate": total_agreements / total_combinations if total_combinations > 0 else 0,
        "avg_utility": np.mean(utilities) if utilities else 0,
        "avg_step_count": np.mean(step_counts) if step_counts else 0,
        "utilities": utilities,
        "step_counts": step_counts
    }
    
    return summary


def print_evaluation_summary(results: Dict):
    """Print a comprehensive evaluation summary."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"Total Domains:      {len(results['domains'])}")
    print(f"Total Opponents:    {len(results['opponents'])}")
    print(f"Total Combinations: {results['total_combinations']}")
    print(f"Agreements:         {results['agreements']}")
    print(f"Agreement Rate:     {results['agreement_rate']:.1%}")
    print(f"Average Utility:    {results['avg_utility']:.3f}")
    print(f"Average Length:     {results['avg_step_count']:.1f} steps")
    
    # Utility distribution
    utilities = results['utilities']
    if utilities:
        print(f"\nUtility Distribution:")
        print(f"  Min:     {np.min(utilities):.3f}")
        print(f"  Max:     {np.max(utilities):.3f}")
        print(f"  Std:     {np.std(utilities):.3f}")
    
    # Results organized by domain
    print(f"\nResults by Domain:")
    domain_results = {}
    for result in results['results']:
        domain = result['domain']
        if domain not in domain_results:
            domain_results[domain] = []
        domain_results[domain].append(result)
    
    for domain in sorted(domain_results.keys()):
        domain_data = domain_results[domain]
        domain_agreements = sum(1 for r in domain_data if r["is_agreement"])
        domain_avg_utility = np.mean([r["final_utility"] for r in domain_data])
        
        print(f"\n  Domain {domain}: {domain_agreements}/{len(domain_data)} agreements, avg utility: {domain_avg_utility:.3f}")
        for result in domain_data:
            opp = result['opponent']
            agreement_status = "✅" if result['is_agreement'] else "❌"
            print(f"    {opp:20s}: {agreement_status} | "
                  f"Utility: {result['final_utility']:.3f} | "
                  f"Steps: {result['step_count']:2d}")


async def main(opponents: List[str] = None, domains: List[str] = None):
    """Main evaluation function that takes a list of opponents and domains."""
    if opponents is None:
        opponents = ["SAGAAgent", "MICROAgent", "ConcederAgent", "BoulwareAgent","HybridAgent"]  # Default opponents
    
    if domains is None:
        domains = ["5", "10", "15", "30", "46"]  # Default domains - you can modify this list
    
    print("🤖 RL Negotiation Model Evaluator")
    print("=" * 60)
    print(f"Running parallel negotiations:")
    print(f"  Domains ({len(domains)}): {domains}")
    print(f"  Opponents ({len(opponents)}): {opponents}")
    print(f"  Total combinations: {len(domains) * len(opponents)}")
    print(f"Device: {DEVICE}")
    
    try:
        # Load model
        model = load_artifact_model(DEVICE)
        
        # Evaluate model in parallel
        results = await evaluate_model_parallel(
            model=model,
            opponents=opponents,
            domains=domains,
            render=True  # Enable rendering for detailed output
        )
        
        # Print summary
        print_evaluation_summary(results)
        
        print(f"\n✅ Parallel evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


def run_evaluation(opponents: List[str] = None, domains: List[str] = None):
    """Synchronous wrapper to run the async main function.""" 
    return asyncio.run(main(opponents, domains))


if __name__ == "__main__":
    run_evaluation()
