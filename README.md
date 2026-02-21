# Meta Negotiation Agent

A Gymnasium-based reinforcement learning extension built on top of [**NegoLog**](https://github.com/aniltrue/NegoLog) (IJCAI 2024) — an integrated Python framework for automated negotiation research.

NegoLog provides the negotiation engine (`nenv`), the tournament infrastructure, the pre-built agent pool, and the domain tooling. This project adds a fully pluggable RL training layer on top: define your observation space, reward function, and policy, and train directly against the existing opponent pool.

> Doğru, A., Keskin, M. O., Jonker, C. M., Baarslag, T., & Aydoğan, R. (2024). NegoLog: An Integrated Python-based Automated Negotiation Framework with Enhanced Assessment Components. *IJCAI 2024*. https://doi.org/10.24963/ijcai.2024/998

---

## Overview

The project has two distinct modes of use:

1. **Tournament** — pit any set of agents against each other across domains and measure performance with configurable loggers. This is the base NegoLog workflow.
2. **RL Training** — define your own observation space, reward function, and policy, then train an agent via reinforcement learning against the existing opponent pool.

---

## Project Structure

```
meta_negotition_agent/
│
├── nenv/                        # Core negotiation framework (NegoLog)
│   ├── Agent.py                 # AbstractAgent base class (BOA architecture)
│   ├── Bid.py                   # Bid representation
│   ├── Preference.py            # Additive utility function
│   ├── Action.py                # Offer / Accept actions
│   ├── Session.py               # Single negotiation session (alternating offers protocol)
│   ├── Tournament.py            # Runs all agent × domain combinations
│   ├── OpponentModel/           # Opponent preference estimators
│   └── logger/                  # Session and tournament loggers
│
├── agents/                      # Pre-built negotiation agents 
│   ├── boulware/                # Time-based Boulware concession
│   ├── conceder/                # Time-based Conceder
│   ├── SAGA/                    # Genetic algorithm-based (ANAC 2019)
│   ├── HybridAgent/
│   ├── CUHKAgent/
│   └── ...
│
├── domains/                     # Negotiation domain files (preference profiles)
│
├── gym_enviroment/              # RL training framework
│   ├── env.py                   # NegotiationEnv (Gymnasium-compatible)
│   ├── rl_agent.py              # AbstractRLAgent — extend this for your agent
│   ├── reward.py                # AbstractRewardFunction — extend this for your reward
│   ├── train.py                 # Training entry point (PPO via Stable-Baselines3)
│   ├── config/
│   │   ├── config.py            # Config loader
│   │   └── default.yaml         # Training hyperparameters, domains, opponents
│   └── agent/                   # Your implementation lives here
│       └── my_agent.py          # Fill in MyRLAgent + MyRewardFunction
│
├── domain_generator/            # Tool to generate new negotiation domains
├── run.py                       # Tournament entry point
└── config.yaml                  # Tournament configuration
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running a Tournament

Tournaments evaluate agents head-to-head across negotiation domains using the stacked alternating offers protocol.

### 1. Configure `config.yaml`

```yaml
deadline_round: 1000

agents:
  - "agents.SAGAAgent"
  - "agents.ConcederAgent"
  - "agents.HybridAgent"

domains: ["5", "6", "7"]

loggers:
  - "BidSpaceLogger"

estimators: []          # optional opponent models for agents that use them

self_negotiation: false
repeat: 1
result_dir: "results/"
seed: 42
shuffle: false
drawing_format: plotly
```

### 2. Run

```bash
python run.py config.yaml
```

Results are written to `results/`.

---

## Training an RL Agent

The RL framework is designed to be fully pluggable. You define three things — **what the policy sees**, **how it is scored**, and **how it decides to bid** — and the framework handles the rest.

### Step 1 — Implement your agent and reward in `gym_enviroment/agent/my_agent.py`

```python
class MyRLAgent(AbstractRLAgent):

    @classmethod
    def get_observation_space(cls) -> gym.Space:
        # Define what the policy sees each step
        return spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    @classmethod
    def get_action_space(cls) -> gym.Space:
        # Define what the policy outputs each step
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def build_observation(self) -> np.ndarray:
        # Convert current negotiation state to an observation vector
        t = self.session_time  # rounds remaining
        last_received = self.last_received_bids[-1].utility if self.last_received_bids else 0.0
        ...
        return np.array([...], dtype=np.float32)

    def set_action(self, action: np.ndarray) -> None:
        self._action = action   # store — used inside act()

    def act(self, t: float) -> Action:
        target_utility = float(np.clip(self._action[0], 0.0, 1.0))
        bid = self.preference.get_bid_at(target_utility)
        if self.can_accept() and bid <= self.last_received_bids[-1]:
            return self.accept_action
        return Offer(bid)

    def receive_offer(self, bid: Bid, t: float) -> None:
        pass  # update opponent model here if needed

    def initiate(self, opponent_name):
        self._action = None


class MyRewardFunction(AbstractRewardFunction):

    def dense_reward(self, env) -> float:
        # Reward at every non-terminal step
        if env.last_our_bid is None:
            return 0.0
        return env.our_preference.get_utility(env.last_our_bid) - 0.5

    def terminal_reward(self, env) -> float:
        # Reward at episode end
        if not env.agreement_reached:
            return -1.0
        return env.final_utility
```

### Step 2 — Configure `gym_enviroment/config/default.yaml`

Set your opponent pool, domains, deadline, and PPO hyperparameters. All agents in `opponents` must be importable subclasses of `AbstractAgent`.

### Step 3 — Train

```bash
python -m gym_enviroment.train
```

Training uses PPO (Stable-Baselines3) across parallel environments, one opponent per worker. Metrics are logged to W&B and TensorBoard.

---

## Key Abstractions

### `AbstractAgent` (`nenv/Agent.py`)

Base class for all negotiation agents. The BOA architecture:

| Method | Purpose |
|---|---|
| `initiate(opponent_name)` | Called once before the session starts |
| `receive_offer(bid, t)` | Called when the opponent makes an offer |
| `act(t) → Action` | Return `Offer(bid)` or `self.accept_action` |
| `terminate(is_accept, opponent_name, t)` | Called at session end |

Available in `act()`: `self.preference`, `self.last_received_bids`, `self.can_accept()`, `self.accept_action`.

### `AbstractRLAgent` (`gym_enviroment/rl_agent.py`)

Extends `AbstractAgent` for RL training. Additional interface:

| Method | Purpose |
|---|---|
| `get_observation_space()` | Class method — declares the observation space |
| `get_action_space()` | Class method — declares the action space |
| `build_observation()` | Called each step — returns the current observation |
| `set_action(action)` | Called before `act()` — stores the policy output |

### `AbstractRewardFunction` (`gym_enviroment/reward.py`)

| Method | Purpose |
|---|---|
| `on_reset(env)` | Called at episode start — reset per-episode state |
| `dense_reward(env) → float` | Called every non-terminal step |
| `terminal_reward(env) → float` | Called on the final step (agreement or deadline) |

Available on `env`: `our_preference`, `opponent_preference`, `last_our_bid`, `last_opponent_bid`, `current_round`, `deadline_round`, `agreement_reached`, `final_utility`, `last_agreed_bid`, `our_agent`.

### `Preference` (`nenv/Preference.py`)

Additive utility: `U(bid) = Σ weight_i × value_utility_i`

| Method | Purpose |
|---|---|
| `get_utility(bid) → float` | Utility of a bid in [0, 1] |
| `get_bid_at(utility) → Bid` | Closest bid to a target utility (binary search) |
| `reservation_value` | Minimum acceptable utility |

---

## Adding a New Agent

Create a new directory under `agents/` and implement `AbstractAgent`:

```python
# agents/MyAgent/MyAgent.py
from nenv import AbstractAgent, Offer
import nenv

class MyAgent(AbstractAgent):
    @property
    def name(self) -> str:
        return "MyAgent"

    def initiate(self, opponent_name):
        pass

    def receive_offer(self, bid, t):
        pass

    def act(self, t):
        target = max(self.preference.reservation_value, 1.0 - t)
        bid = self.preference.get_bid_at(target)
        if self.can_accept() and bid <= self.last_received_bids[-1]:
            return self.accept_action
        return Offer(bid)
```

Then reference it in `config.yaml` as `"agents.MyAgent"` or by full path.

---

## Citation

If you use the negotiation framework (`nenv`) in your research, please cite the NegoLog paper:

```bibtex
@inproceedings{ijcai2024p998,
  title     = {NegoLog: An Integrated Python-based Automated Negotiation Framework with Enhanced Assessment Components},
  author    = {Doğru, Anıl and Keskin, Mehmet Onur and Jonker, Catholijn M. and Baarslag, Tim and Aydoğan, Reyhan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {8640--8643},
  year      = {2024},
  doi       = {10.24963/ijcai.2024/998},
  url       = {https://doi.org/10.24963/ijcai.2024/998},
}
```
