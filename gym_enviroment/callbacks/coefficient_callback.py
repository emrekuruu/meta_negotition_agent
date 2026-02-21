"""
Coefficient Data Tracking Callback for RL Negotiation Training
Tracks Nash equation coefficients and saves to CSV for later analysis.
Async background processing for zero training disruption.
"""

import os
import csv
import numpy as np
import asyncio
import threading
import queue
from collections import defaultdict, deque
from stable_baselines3.common.callbacks import BaseCallback


class AsyncCSVWriter:
    """Async CSV writer that processes data in background."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.write_queue = queue.Queue()
        self.running = True
        self.loop = None
        self.thread = None
        
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Start background thread
        self._start_background_thread()
    
    def _start_background_thread(self):
        """Start the background asyncio event loop."""
        def run_async_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._writer_loop())
        
        self.thread = threading.Thread(target=run_async_loop, daemon=True)
        self.thread.start()
    
    async def _writer_loop(self):
        """Main async writer loop."""
        while self.running:
            try:
                # Check for work every 100ms to avoid blocking
                await asyncio.sleep(0.1)
                
                # Process all queued writes
                while not self.write_queue.empty():
                    try:
                        write_data = self.write_queue.get_nowait()
                        if write_data is None:  # Shutdown signal
                            return
                        
                        await self._write_data(write_data)
                        self.write_queue.task_done()
                        
                    except queue.Empty:
                        break
                        
            except Exception as e:
                print(f"❌ AsyncCSVWriter error: {e}")
    
    async def _write_data(self, write_data):
        """Write data to CSV file asynchronously."""
        file_path, rows, headers = write_data
        
        try:
            # Check if file exists
            file_exists = os.path.exists(file_path)
            
            # Use asyncio to run file I/O in thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, self._sync_write, file_path, rows, headers, file_exists
            )
            
        except Exception as e:
            print(f"❌ Error writing to {file_path}: {e}")
    
    def _sync_write(self, file_path, rows, headers, file_exists):
        """Synchronous write operation."""
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers if new file
            if not file_exists and headers:
                writer.writerow(headers)
            
            # Write data
            writer.writerows(rows)
    
    def queue_write(self, file_path: str, rows: list, headers: list = None):
        """Queue data for async writing."""
        if self.running:
            try:
                self.write_queue.put((file_path, rows, headers), block=False)
            except queue.Full:
                print(f"⚠️ Write queue full, dropping data for {file_path}")
    
    def shutdown(self):
        """Shutdown the async writer and flush remaining data."""
        print(f"🔄 Shutting down AsyncCSVWriter, processing {self.write_queue.qsize()} remaining items...")
        
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
        
        print("✅ AsyncCSVWriter shutdown complete")


class CoefficientTrackingCallback(BaseCallback):
    """
    Tracks coefficient evolution during episodes and saves to CSV files.
    Async background I/O for zero training disruption.
    """
    
    def __init__(self, 
                 episode_freq: int = 100,  # Back to more frequent logging since I/O is async
                 min_steps_for_tracking: int = 96,
                 log_dir: str = "episode_logs/coefficient_tracking", 
                 verbose: int = 0):
        super().__init__(verbose)
        self.episode_freq = episode_freq
        self.min_steps_for_tracking = min_steps_for_tracking
        self.log_dir = log_dir
        
        # Async CSV writer
        self.csv_writer = AsyncCSVWriter(log_dir)
        
        # Episode counters
        self.episode_counters = defaultdict(int)
        
        # Current episode tracking
        self.current_episodes = defaultdict(lambda: {
            'coefficients': [],
            'opponent_name': 'unknown',
            'domain': 'unknown',
            'step_count': 0
        })
        
        # CSV headers - new order: domain, opponent, episode, round, ...
        self.headers = ["domain", "opponent", "episode", "round", "time_progress", "time_coefficient"] + \
                      [f"nash_coeff_{i+1}" for i in range(96)] + \
                      ["timestep"]
    
    def _queue_episode_data(self, ep_data: dict, episode_num: int):
        """Queue episode data for async writing."""
        try:
            opponent = ep_data['opponent_name'].replace("/", "_").replace(" ", "_")
            
            # Prepare data rows with new column order: domain, opponent, episode, round, ...
            rows = []
            domain = ep_data.get('domain', 'unknown').replace("/", "_").replace(" ", "_")
            for round_no, time_progress, nash_coeffs, time_coeff in ep_data['coefficients']:
                row = [domain, opponent, episode_num, round_no, time_progress, time_coeff] + \
                      list(nash_coeffs) + [self.num_timesteps]
                rows.append(row)
            
            if rows:
                # Queue for global CSV
                global_path = os.path.join(self.log_dir, "global_coefficient_dynamics.csv")
                self.csv_writer.queue_write(global_path, rows, self.headers)
                
                # Queue for opponent CSV
                opponent_path = os.path.join(self.log_dir, f"{opponent}_coefficient_dynamics.csv")
                self.csv_writer.queue_write(opponent_path, rows, self.headers)
                
                if self.verbose > 0:
                    print(f"📊 Queued {len(rows)} coefficient rows for {opponent}")
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ Error queuing coefficient data: {e}")
    
    def _on_step(self) -> bool:
        """Track coefficient values each step - completely non-blocking version."""
        
        try:
            actions = self.locals.get('actions', None)
            infos = self.locals.get('infos', [])
            dones = self.locals.get('dones', [])
            
            # OPTIMIZATION: Skip entirely if no actions
            if actions is None:
                return True
            
            # OPTIMIZATION: No vectorized environment calls - extract round info from infos
            for env_idx, (action, info, done) in enumerate(zip(actions, infos, dones)):
                if not isinstance(info, dict):
                    continue
            
                if len(action) >= 98:  # Ensure full action vector
                    # Initialize episode data if needed
                    if env_idx not in self.current_episodes:
                        self.current_episodes[env_idx] = {
                            'coefficients': [],
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
                    deadline_round = info.get('deadline_round', 1000)  # Default fallback
                    
                    # Only track coefficients after we have enough history
                    if ep_data['step_count'] >= self.min_steps_for_tracking:
                        # OPTIMIZATION: Minimal processing
                        nash_coeffs = action[1:97]
                        time_coeff = action[97]
                        time_progress = round_no / deadline_round if deadline_round > 0 else 0.0
                        
                        # Record coefficients (copy only when storing)
                        ep_data['coefficients'].append((round_no, time_progress, nash_coeffs.copy(), time_coeff))
                    
                    ep_data['step_count'] += 1
                    
                    # Episode ended - save data if needed
                    if done:
                        self.episode_counters[env_idx] += 1
                        episode_num = self.episode_counters[env_idx]
                        
                        # Save every Nth episode
                        if episode_num % self.episode_freq == 0 and ep_data['coefficients']:
                            # Queue data for background processing
                            if self.verbose > 0:
                                print(f"📊 Queuing coefficient data for episode {episode_num} vs {ep_data['opponent_name']}")
                            
                            self._queue_episode_data(ep_data, episode_num)
                        
                        # Reset for next episode
                        self.current_episodes[env_idx] = {
                            'coefficients': [],
                            'opponent_name': 'unknown',
                            'domain': 'unknown',
                            'step_count': 0
                        }
                        
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ CoefficientTrackingCallback error: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Clean shutdown of async writer."""
        if hasattr(self, 'csv_writer'):
            self.csv_writer.shutdown()
    
    def _on_rollout_end(self) -> None:
        """Log summary at rollout end."""
        if self.verbose > 0:
            total_episodes = sum(self.episode_counters.values())
            active_episodes = len([ep for ep in self.current_episodes.values() if ep['step_count'] > 0])
            print(f"📊 Coefficient Callback Summary: {total_episodes} episodes completed, {active_episodes} active")