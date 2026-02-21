import os
import random
import shutil
import time
import warnings
import traceback
import itertools
from typing import Union, Set, List, Tuple, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from nenv.Agent import AgentClass
from nenv.logger import AbstractLogger, LoggerClass
from nenv.OpponentModel import OpponentModelClass
from nenv.SessionManager import SessionManager
from nenv.utils import ExcelLog, TournamentProcessMonitor

_original_random_class = random.Random

_gpu_queue = None

def _init_worker_gpu(gpu_queue):
    """Initialize worker process with GPU queue for conditional assignment."""
    global _gpu_queue
    _gpu_queue = gpu_queue

def _configure_process_seed(seed: Optional[int]):
    """
    Configure deterministic randomness inside a worker process.

    Seeds Python's random, NumPy, and redefines random.Random()'s default
    constructor so that seedless instantiations (common in agents) become
    reproducible across runs.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    counter = itertools.count()
    random.Random = _original_random_class

    class SeededRandom(_original_random_class):
        def __init__(self, a=None):
            if a is None:
                a = seed + next(counter)
            super().__init__(a)

    random.Random = SeededRandom


def run_single_session(session_data):
    """
    Run a single negotiation session in a separate process.
    Only uses session-level loggers to avoid conflicts.
    """
    try:
        (session_id, agent_class_1_name, agent_class_2_name, domain_name,
         deadline_time, deadline_round, estimators, result_dir, session_seed) = session_data

        _configure_process_seed(session_seed)

        # Assign GPU if either agent is NegoformerAgent
        global _gpu_queue
        if _gpu_queue is not None and ('NegoformerAgent' in [agent_class_1_name, agent_class_2_name]):
            gpu_id = _gpu_queue.get()
            os.environ['WORKER_GPU_ID'] = str(gpu_id)

        # Import agent classes dynamically
        import importlib
        
        agent_mapping = {
            'BoulwareAgent': 'agents.boulware.Boulware.BoulwareAgent',
            'MICROAgent': 'agents.MICRO.MICRO.MICROAgent', 
            'HybridAgent': 'agents.HybridAgent.HybridAgent.HybridAgent',
            'SAGAAgent': 'agents.SAGA.SAGAAgent.SAGAAgent',
            'ConcederAgent': 'agents.conceder.Conceder.ConcederAgent',
            'ParetoWalkerAgent': 'agents.ParetoWalkerAgent.ParetoWalkerAgent.ParetoWalkerAgent',
            'NegoformerAgent': 'agents.NegoformerAgent.NegoformerAgent.NegoformerAgent',
            'CUHKAgent': 'agents.CUHKAgent.CUHKAgent.CUHKAgent',
            "PonPokoAgent": "agents.PonPoko.PonPoko.PonPokoAgent",
            "NiceTitForTat": "agents.NiceTitForTat.NiceTitForTat.NiceTitForTat",
            "IAMhaggler": "agents.IAMhaggler.IAMhaggler.IAMhaggler",
            "HardHeaded": "agents.HardHeaded.KLH.HardHeaded",
            "FakeoutAgent": "agents.FakeoutAgent.FakeoutAgent.FakeoutAgent",
        }
        
        
        def get_agent_class(agent_name):
            if agent_name in agent_mapping:
                module_path, class_name = agent_mapping[agent_name].rsplit('.', 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
        
        agent_class_1 = get_agent_class(agent_class_1_name)
        agent_class_2 = get_agent_class(agent_class_2_name)
        
        # Filter to session-level loggers only (exclude tournament-level ones)
        session_loggers = [
            'BidSpaceLogger',
            'MoveAnalyzeLogger',
            'UtilityDistributionLogger',
            'EstimatorWeightLogger',
        ]
        
        # Import and create logger instances (session-level only)
        logger_instances = []
        for logger_name in session_loggers:
            try:
                module = importlib.import_module('nenv.logger')
                logger_class = getattr(module, logger_name)
                logger_instances.append(logger_class(result_dir))
            except Exception as e:
                print(f"Warning: Could not create logger {logger_name}: {e}")
        
        # Create session with session-level loggers only
        session_runner = SessionManager(
            agent_class_1, agent_class_2, domain_name, 
            deadline_time, deadline_round, estimators, logger_instances
        )
        
        if os.getenv("OPPONENT_MODEL"):
            session_path = session_path = f"{agent_class_1_name}_{agent_class_2_name}_Domain{domain_name}_{os.getenv('OPPONENT_MODEL')}.xlsx"
        else:
            session_path = session_path = f"{agent_class_1_name}_{agent_class_2_name}_Domain{domain_name}_Oracle.xlsx"

        full_session_path = os.path.join(result_dir, "sessions", session_path)
        
        # Run the session
        session_start_time = time.time()
        result = session_runner.run(full_session_path)
        session_elapsed_time = time.time() - session_start_time
        
        # Return full session result including TournamentResults and estimator data
        return {
            'session_id': session_id,
            'agent_a': agent_class_1_name,
            'agent_b': agent_class_2_name,
            'domain': domain_name,
            'time': session_elapsed_time,
            'session_file': full_session_path,
            'result_data': result if isinstance(result, dict) else {}
        }
        
    except Exception as e:
        print(f"Process {session_id}: Error - {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        return {'error': str(e), 'session_id': session_id}


class Tournament:
    """
        This class conducts a tournament based on given settings.
    """
    agent_classes: Set[AgentClass]                 #: List of Agent classes
    loggers: List[AbstractLogger]                  #: List of Logger classes
    domains: List[str]                             #: List of domains
    estimators: Set[OpponentModelClass]            #: List of opponent models
    deadline_time: Optional[int]                    #: Time-based deadline in terms of seconds
    deadline_round: Optional[int]                   #: Round-based deadline in terms of number of rounds
    result_dir: str                                #: The directory where the result logs will be extracted
    seed: Optional[int]                             #: Random seed for whole tournament
    shuffle: bool                                  #: Whether the combinations will be shuffled, or not
    repeat: int                                    #: Number of repetition for each combination
    self_negotiation: bool                         #: Whether the agents negotiate with itself, or not
    tournament_process: TournamentProcessMonitor   #: Process monitor
    killed: bool                                   #: Whether the tournament process is killed, or not

    def __init__(self, agent_classes: Union[List[AgentClass], Set[AgentClass]],
                 domains: List[str],
                 logger_classes: Union[List[LoggerClass], Set[LoggerClass]],
                 estimator_classes: Union[List[OpponentModelClass], Set[OpponentModelClass]],
                 deadline_time: Optional[int],
                 deadline_round: Optional[int],
                 self_negotiation: bool = False,
                 repeat: int = 1,
                 result_dir: str = "results/",
                 seed: Optional[int] = None,
                 shuffle: bool = False
                 ):
        """
            This class conducts a negotiation tournament.

            :param agent_classes: List of agent classes (i.e., subclass of AbstractAgent class)
            :param domains: List of domains
            :param logger_classes: List of loggers classes (i.e., subclass of AbstractLogger class)
            :param estimator_classes: List of estimator classes (i.e, subclass of AbstractOpponentModel class)
            :param deadline_time: Time-based deadline in terms of seconds
            :param deadline_round: Round-based deadline in terms of number of rounds
            :param self_negotiation: Whether the agents negotiate with itself. *Default false*.
            :param repeat: Number of repetition for each combination. *Default 1*
            :param result_dir: The result directory that the tournament logs will be created. *Default 'results/'*
            :param seed: Setting seed for whole tournament. *Default None*.
            :param shuffle: Whether shuffle negotiation combinations. *Default False*
        """

        assert deadline_time is not None or deadline_round is not None, "No deadline type is specified."
        assert deadline_time is None or deadline_time > 0, "Deadline must be positive."
        assert deadline_round is None or deadline_round > 0, "Deadline must be positive."

        if repeat <= 0:
            warnings.warn("repeat is set to 1.")
            repeat = 1

        assert len(agent_classes) > 0, "Empty list of agent classes."
        assert len(domains) > 0, "Empty list of domains."

        self.agent_classes = agent_classes
        self.domains = domains
        self.estimators = estimator_classes
        self.deadline_time = deadline_time
        self.deadline_round = deadline_round
        self.loggers = [logger_class(result_dir) for logger_class in set(logger_classes)]
        self.result_dir = result_dir + os.getenv("NEGOFORMER_MODE") + "/" + os.getenv("DEADLINE_ROUND") + "/" + os.getenv("DOMAIN_NAME") + "/"
        self.seed = seed
        self.repeat = repeat
        self.self_negotiation = self_negotiation
        self.shuffle = shuffle
        self.tournament_process = TournamentProcessMonitor()
        self.killed = False

    def run(self):
        """
            This method starts the tournament

            :return: Nothing
        """
        # Set seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            os.environ['PYTHONHASHSEED'] = str(self.seed)

        # Create result directory
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(os.path.join(self.result_dir, "sessions/")), exist_ok=True)

        # Set killed flag
        self.killed = False

        # Extract domain information into the result directory
        self.extract_domains()

        # Get all combinations
        negotiations = self.generate_combinations()

        # Names for logger
        agent_names = []
        estimator_names = []

        # Tournament log file
        tournament_logs = ExcelLog(["TournamentResults"])

        tournament_logs.save(os.path.join(self.result_dir, "results.xlsx"))

        self.tournament_process.initiate(len(negotiations))

        max_workers =  min(180,multiprocessing.cpu_count())
        
        # Prepare session data for parallel execution
        session_data_list = []
        
        for i, (agent_class_1, agent_class_2, domain_name) in enumerate(negotiations):

            if "NegoformerAgent" in [agent_class_1.__name__, agent_class_2.__name__]:
                for file in os.listdir(os.path.join(self.result_dir, "sessions/")):
                    file = file.split("Process")[0]
                    if file == f"{agent_class_1.__name__}_{agent_class_2.__name__}_Domain{domain_name}_":
                        break
                else:
                    # Only add session if no matching file was found
                    session_seed = None if self.seed is None else self.seed + i
                    session_data_list.append((i, agent_class_1.__name__, agent_class_2.__name__, domain_name,
                        self.deadline_time, self.deadline_round, self.estimators, self.result_dir, session_seed))

        # Count sessions with NegoformerAgent and detect GPUs
        negoformer_sessions = sum(
            1 for (_, a1, a2, _, _, _, _, _, _) in session_data_list
            if 'NegoformerAgent' in [a1, a2]
        )

        gpu_queue = None
        if negoformer_sessions > 0:
            try:
                import torch
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if num_gpus > 0:
                    from multiprocessing import Manager
                    manager = Manager()
                    gpu_queue = manager.Queue()
                    for i in range(negoformer_sessions):
                        gpu_queue.put(i % num_gpus)
            except ImportError:
                pass

        # Run sessions in parallel
        completed_sessions = []
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker_gpu if gpu_queue is not None else None,
            initargs=(gpu_queue,) if gpu_queue is not None else ()
        ) as executor:
            future_to_session = {
                executor.submit(run_single_session, session_data): session_data
                for session_data in session_data_list
            }
            
            for future in as_completed(future_to_session):
                if self.killed:
                    executor.shutdown(wait=False)
                    return
                
                try:
                    result = future.result()
                    
                    if 'error' in result:
                        print(f"❌ Session {result['session_id']} failed: {result['error']}")
                        continue
                    
                    completed_sessions.append(result)

                    # Collect agent names from actual session results (not class names)
                    tournament_result = result['result_data'].get('TournamentResults', {})
                    actual_agent_a = tournament_result.get('AgentA', result['agent_a'])
                    actual_agent_b = tournament_result.get('AgentB', result['agent_b'])
                    
                    if actual_agent_a not in agent_names:
                        agent_names.append(actual_agent_a)
                    if actual_agent_b not in agent_names:
                        agent_names.append(actual_agent_b)
                    
                except Exception as e:
                    print(f"❌ Error processing session result: {e}")
                    continue
        
        print(f"Completed {len(completed_sessions)} sessions.")

        # Rebuild tournament logs from in-memory result data
        self.rebuild_tournament_logs_from_files(tournament_logs, completed_sessions)

        # Extract estimator names from the rebuilt tournament logs
        standard_sheets = {"TournamentResults"}
        estimator_names = [sheet for sheet in tournament_logs.sheet_names if sheet not in standard_sheets]

        # Extract domain names
        domain_names = list(set([session['domain'] for session in completed_sessions]))

        # Call on_tournament_end for all loggers
        for logger in self.loggers:
            logger.on_tournament_end(tournament_logs, agent_names, domain_names, estimator_names)

        # Save final tournament results
        tournament_logs.save(os.path.join(self.result_dir, "results.xlsx"))

        print(f"Tournament completed. Results saved to {os.path.join(self.result_dir, 'results.xlsx')}")

    def rebuild_tournament_logs_from_files(self, tournament_logs: ExcelLog, completed_sessions: List[dict]):
        """
        Rebuild complete tournament logs from in-memory session results.
        The estimator data is only available in result_data (not in session files)
        because session files are saved before on_accept/on_fail are called.
        """
        # Clear existing logs and rebuild
        tournament_logs.log_rows = {"TournamentResults": []}
        tournament_logs.sheet_names = {"TournamentResults"}

        for session_result in completed_sessions:
            try:
                result_data = session_result.get('result_data', {})

                # Extract and append TournamentResults
                if "TournamentResults" in result_data:
                    tournament_result = result_data["TournamentResults"].copy()
                    tournament_result["SessionRealTime"] = session_result['time']
                    tournament_logs.log_rows["TournamentResults"].append(tournament_result)

                # Extract estimator sheets from result_data
                # Estimator data is returned by loggers' on_accept/on_fail methods
                standard_sheets = {"TournamentResults"}
                for sheet_name in result_data.keys():
                    if sheet_name not in standard_sheets:
                        # This is an estimator sheet
                        tournament_logs.sheet_names.add(sheet_name)

                        # Initialize the sheet in tournament_logs if it doesn't exist
                        if sheet_name not in tournament_logs.log_rows:
                            tournament_logs.log_rows[sheet_name] = []

                        # Append the estimator data for this session
                        tournament_logs.log_rows[sheet_name].append(result_data[sheet_name])

            except Exception as e:
                print(f"Warning: Error processing session result: {e}")
                continue
        

    def generate_combinations(self) -> List[Tuple[AgentClass, AgentClass, str]]:
        """
            This method generates all combinations of negotiations.

            :return: Nothing
        """
        combinations = []

        for domain in self.domains:
            for agent_class_1 in self.agent_classes:
                for agent_class_2 in self.agent_classes:
                    for i in range(self.repeat):
                        combinations.append((agent_class_1, agent_class_2, domain))

        if self.shuffle:
            random.shuffle(combinations)

        return combinations

    def extract_domains(self):
        """
            This method extracts the domain information into the result directory.

            :return: Nothing
        """
        full_domains = pd.read_excel("domains/domains.xlsx", sheet_name="domains")

        domains = pd.DataFrame(columns=full_domains.columns[1:])

        domain_counter = 0

        for i, row in full_domains.iterrows():
            if str(row["DomainName"]) in self.domains:
                domains.loc[domain_counter] = row

                domain_counter += 1

        domains.to_excel(os.path.join(self.result_dir, "domains.xlsx"), sheet_name="domains", index=False)
