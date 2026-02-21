"""
Strategy Data Tracking Callback for RL Negotiation Training
Tracks strategy selection and saves to CSV for later analysis.
Async I/O for zero training disruption.
"""

import os
import csv
import numpy as np
import asyncio
import threading
import queue
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback


class AsyncStrategyWriter:
    """Async strategy data writer."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.write_queue = queue.Queue()
        self.running = True
        self.loop = None
        self.thread = None
        
        os.makedirs(log_dir, exist_ok=True)
        self._start_background_thread()
    
    def _start_background_thread(self):
        def run_async_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._writer_loop())
        
        self.thread = threading.Thread(target=run_async_loop, daemon=True)
        self.thread.start()
    
    async def _writer_loop(self):
        while self.running:
            try:
                await asyncio.sleep(0.1)
                while not self.write_queue.empty():
                    try:
                        write_data = self.write_queue.get_nowait()
                        if write_data is None:
                            return
                        await self._write_data(write_data)
                        self.write_queue.task_done()
                    except queue.Empty:
                        break
            except Exception as e:
                print(f"❌ AsyncStrategyWriter error: {e}")
    
    async def _write_data(self, write_data):
        file_path, rows, headers = write_data
        try:
            file_exists = os.path.exists(file_path)
            await asyncio.get_event_loop().run_in_executor(
                None, self._sync_write, file_path, rows, headers, file_exists
            )
        except Exception as e:
            print(f"❌ Error writing to {file_path}: {e}")
    
    def _sync_write(self, file_path, rows, headers, file_exists):
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists and headers:
                writer.writerow(headers)
            writer.writerows(rows)
    
    def queue_write(self, file_path: str, rows: list, headers: list = None):
        if self.running:
            try:
                self.write_queue.put((file_path, rows, headers), block=False)
            except queue.Full:
                print(f"⚠️ Write queue full, dropping data for {file_path}")
    
    def shutdown(self):
        """Shutdown the async writer and flush remaining data."""
        print(f"🔄 Shutting down AsyncStrategyWriter, processing {self.write_queue.qsize()} remaining items...")
        
        # Process any remaining queued items synchronously
        while not self.write_queue.empty():
            try:
                write_data = self.write_queue.get_nowait()
                if write_data is not None:
                    file_path, rows, headers = write_data
                    file_exists = os.path.exists(file_path)
                    self._sync_write(file_path, rows, headers, file_exists)
                    print(f"✅ Flushed {len(rows)} rows to {file_path}")
            except:
                break
        
        # Stop the async loop
        self.running = False
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        print("✅ AsyncStrategyWriter shutdown complete")


class StrategyTrackingCallback(BaseCallback):
    """
    Tracks strategy selection during episodes with async I/O.
    """
    
    def __init__(self, 
                 episode_freq: int = 50,  # More frequent since I/O is async
                 log_dir: str = "episode_logs/strategy_tracking",
                 verbose: int = 0):
        super().__init__(verbose)
        self.episode_freq = episode_freq
        self.log_dir = log_dir
        
        # Async writer
        self.csv_writer = AsyncStrategyWriter(log_dir)
        
        # Episode counters
        self.episode_counters = defaultdict(int)
        
        # Current episode tracking
        self.current_episodes = defaultdict(lambda: {
            'strategies': [],
            'opponent_name': 'unknown',
            'domain': 'unknown',
            'step_count': 0
        })
        
        # Strategy names
        self.strategy_names = ['SAGA', 'Hybrid', 'Conceder', 'Boulware', 'MICRO', 'KAWAII']
        
        # CSV headers - new order: domain, opponent, episode, round, ...
        self.headers = ["domain", "opponent", "episode", "round", "time_progress", "strategy_index", "strategy_name", "timestep"]
    
    def _queue_episode_data(self, ep_data: dict, episode_num: int):
        """Queue episode data for async writing."""
        try:
            opponent = ep_data['opponent_name'].replace("/", "_").replace(" ", "_")
            
            # Prepare data rows with new column order: domain, opponent, episode, round, ...
            rows = []
            domain = ep_data.get('domain', 'unknown').replace("/", "_").replace(" ", "_")
            for round_no, time_progress, strategy_index, strategy_name in ep_data['strategies']:
                row = [domain, opponent, episode_num, round_no, time_progress, strategy_index, strategy_name, self.num_timesteps]
                rows.append(row)
            
            if rows:
                # Queue for global CSV
                global_path = os.path.join(self.log_dir, "global_strategy_dynamics.csv")
                self.csv_writer.queue_write(global_path, rows, self.headers)
                
                # Queue for opponent CSV
                opponent_path = os.path.join(self.log_dir, f"{opponent}_strategy_dynamics.csv")
                self.csv_writer.queue_write(opponent_path, rows, self.headers)
                
                if self.verbose > 0:
                    print(f"📊 Queued {len(rows)} strategy rows for {opponent}")
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ Error queuing strategy data: {e}")
    
    def _on_step(self) -> bool:
        """Track strategy selections each step - completely non-blocking version."""
        
        try:
            actions = self.locals.get('actions', None)
            if actions is None:
                return True
                
            infos = self.locals.get('infos', [])
            dones = self.locals.get('dones', [])
            
            # OPTIMIZATION: No vectorized environment calls
            
            for env_idx, (action, info, done) in enumerate(zip(actions, infos, dones)):
                if not isinstance(info, dict):
                    continue
                    
                if len(action) >= 1:  # Ensure we have strategy index
                    # Initialize episode data if needed
                    if env_idx not in self.current_episodes:
                        self.current_episodes[env_idx] = {
                            'strategies': [],
                            'opponent_name': 'unknown',
                            'domain': 'unknown',
                            'step_count': 0
                        }
                    
                    ep_data = self.current_episodes[env_idx]
                    
                    # Update episode info
                    ep_data['opponent_name'] = info.get('opponent', 'unknown')
                    ep_data['domain'] = info.get('domain', 'unknown')
                    
                    # Get round info from info dict (no vectorized calls)
                    round_no = info.get('round', ep_data['step_count'])
                    deadline_round = info.get('deadline_round', 1000)
                    
                    # Extract strategy selection
                    strategy_index = int(np.clip(np.round(action[0]), 0, len(self.strategy_names) - 1))
                    strategy_name = self.strategy_names[strategy_index] if strategy_index < len(self.strategy_names) else 'Unknown'
                    
                    # Calculate time progress
                    time_progress = round_no / deadline_round if deadline_round > 0 else 0.0
                    
                    # Record strategy
                    ep_data['strategies'].append((round_no, time_progress, strategy_index, strategy_name))
                    ep_data['step_count'] += 1
                    
                    # Episode ended - save data if needed
                    if done:
                        self.episode_counters[env_idx] += 1
                        episode_num = self.episode_counters[env_idx]
                        
                        # Save every Nth episode  
                        if episode_num % self.episode_freq == 0 and ep_data['strategies']:
                            if self.verbose > 0:
                                print(f"🎯 Queuing strategy data for episode {episode_num} vs {ep_data['opponent_name']}")
                            
                            self._queue_episode_data(ep_data, episode_num)
                        
                        # Reset for next episode
                        self.current_episodes[env_idx] = {
                            'strategies': [],
                            'opponent_name': 'unknown',
                            'domain': 'unknown',
                            'step_count': 0
                        }
                        
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ StrategyTrackingCallback error: {e}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log summary at rollout end."""
        if self.verbose > 0:
            total_episodes = sum(self.episode_counters.values())
            active_episodes = len([ep for ep in self.current_episodes.values() if ep['step_count'] > 0])
            print(f"🎯 Strategy Callback Summary: {total_episodes} episodes completed, {active_episodes} active")
    
    def _on_training_end(self) -> None:
        """Clean shutdown of async writer."""
        if hasattr(self, 'csv_writer'):
            self.csv_writer.shutdown()