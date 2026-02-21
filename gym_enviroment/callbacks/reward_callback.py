# Simple episode dynamics reward tracking with async I/O

import os
import csv
import asyncio
import threading
import queue
from stable_baselines3.common.callbacks import BaseCallback


class AsyncRewardWriter:
    """Async reward data writer."""
    
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
                print(f"❌ AsyncRewardWriter error: {e}")
    
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
        print(f"🔄 Shutting down AsyncRewardWriter, processing {self.write_queue.qsize()} remaining items...")
        
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
        
        print("✅ AsyncRewardWriter shutdown complete")


class RewardTrackingCallback(BaseCallback):
    """
    Simple episode dynamics tracking with async I/O.
    """

    def __init__(self, max_round: int = 50, episode_freq: int = 50, log_dir: str = "episode_logs", verbose: int = 0):
        super().__init__(verbose)
        self.max_round = max_round
        self.episode_freq = episode_freq
        self.log_dir = log_dir
        
        # Async writer
        self.csv_writer = AsyncRewardWriter(log_dir)
        
        # Episode tracking
        self.current_episodes = {}
        self.episode_counters = {}
        
        # CSV headers - new order: domain, opponent, episode, round, ...
        self.headers = ["domain", "opponent", "episode", "round", "nash_reward", "terminal_reward", "strategy_fit_reward", "timestep"]

    def _clean(self, name: str) -> str:
        return name.replace("/", "_").replace(" ", "_").replace(".", "_")

    def _on_step(self) -> bool:
        try:
            # OPTIMIZATION: No vectorized environment calls - get data from locals only
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [])
            dones = self.locals.get("dones", [])

            for idx, (info, total_rew, done) in enumerate(zip(infos, rewards, dones)):
                if not isinstance(info, dict):
                    continue
                    
                # Extract data from info dict (no vectorized calls)
                nash = info.get("nash_dense", 0.0)
                round_no = info.get("round", 0)
                info_term = info.get("terminal_reward", None)
                strategy_fit = info.get("strategy_fit", 0.0)
                
                if self.verbose > 1:  # Extra verbose debugging
                    print(f"RewardCallback: round={round_no}, nash={nash}, terminal={info_term}, strategy_fit={strategy_fit}, done={done}")
                # Initialize episode if needed
                if idx not in self.current_episodes or not self.current_episodes[idx]:
                    # Initialize episode counter for this env if needed
                    if idx not in self.episode_counters:
                        self.episode_counters[idx] = 0
                        
                    opponent_name = info.get("opponent", "unknown")
                    domain_name = info.get("domain", "unknown")
                    self.current_episodes[idx] = {
                        'nash_by_round': {},
                        'strategy_fit_by_round': {},
                        'opponent': opponent_name,
                        'domain': domain_name
                    }

                ep_data = self.current_episodes[idx]

                # Collect Nash reward and strategy fit reward during episode
                if not done and round_no < self.max_round:
                    if nash is not None:
                        ep_data['nash_by_round'][round_no] = float(nash)
                    if strategy_fit is not None:
                        ep_data['strategy_fit_by_round'][round_no] = float(strategy_fit)

                # Episode ended - check if we should log this episode
                if done:
                    opponent = self._clean(ep_data['opponent'])
                    domain = self._clean(ep_data['domain'])
                    self.episode_counters[idx] += 1
                    episode_num = self.episode_counters[idx]
                    
                    # Only log every Nth episode
                    if episode_num % self.episode_freq == 0:
                        if self.verbose > 0:
                            print(f"Logging episode {episode_num}")
                        
                        # Build episode data with new column order: domain, opponent, episode, round, nash_reward, terminal_reward, strategy_fit_reward, timestep
                        episode_data = []
                        
                        # Collect all rounds that have either Nash or strategy fit data
                        all_rounds = set(ep_data['nash_by_round'].keys()) | set(ep_data['strategy_fit_by_round'].keys())
                        
                        # Add rewards by round
                        for round_num in sorted(all_rounds):
                            if round_num < self.max_round:
                                nash_reward = ep_data['nash_by_round'].get(round_num, None)
                                strategy_fit_reward = ep_data['strategy_fit_by_round'].get(round_num, None)
                                # [domain, opponent, episode, round, nash_reward, terminal_reward, strategy_fit_reward, timestep]
                                episode_data.append([domain, opponent, episode_num, round_num, nash_reward, None, strategy_fit_reward, None])

                        # Add terminal reward if it exists
                        if info_term is not None and info_term != 0.0:
                            episode_data.append([domain, opponent, episode_num, round_no, None, float(info_term), None, None])
                        elif total_rew != 0.0:
                            episode_data.append([domain, opponent, episode_num, round_no, None, float(total_rew), None, None])

                        # Queue if we have data
                        if episode_data:
                            self._queue_episode_data(episode_data)

                    # Reset episode
                    self.current_episodes[idx] = None

        except Exception as e:
            if self.verbose > 0:
                print(f"RewardTrackingCallback error: {e}")
                import traceback
                traceback.print_exc()

        return True

    def _queue_episode_data(self, episode_data: list):
        """Queue episode data for async writing."""
        try:
            # Add timestep to each row (data already contains domain, opponent, episode, round, ...)
            enriched_data = []
            for row in episode_data:
                # Update the timestep field (last column)
                enriched_row = row[:-1] + [self.num_timesteps]
                enriched_data.append(enriched_row)
            
            if enriched_data:
                # Extract opponent from first row for file naming
                opponent = enriched_data[0][1] if len(enriched_data) > 0 and len(enriched_data[0]) > 1 else "unknown"
                
                # Queue for global CSV
                global_path = os.path.join(self.log_dir, "global_episode_dynamics.csv")
                self.csv_writer.queue_write(global_path, enriched_data, self.headers)
                
                # Queue for opponent CSV
                opponent_path = os.path.join(self.log_dir, f"{opponent}_episode_dynamics.csv")
                self.csv_writer.queue_write(opponent_path, enriched_data, self.headers)
                
                if self.verbose > 0:
                    print(f"📊 Queued {len(episode_data)} reward rows for {opponent}")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ Error queuing reward data: {e}")

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        """Clean shutdown of async writer."""
        if hasattr(self, 'csv_writer'):
            self.csv_writer.shutdown()
