from abc import abstractmethod, ABC
from typing import List

import numpy as np
import gymnasium as gym

from nenv.Agent import AbstractAgent
from nenv.OpponentModel import AbstractOpponentModel
from nenv.Preference import Preference


class AbstractRLAgent(AbstractAgent, ABC):
    """
    Base class for RL-controlled negotiation agents.

    Extend this class to define your own negotiation agent for RL training.
    You must implement:
        - get_observation_space: class method declaring what build_observation() returns
        - get_action_space: class method declaring what set_action() receives
        - build_observation: converts current negotiation state to an RL observation
        - set_action: stores the action from the RL policy for use in act()
        - act: the negotiation decision (Offer or Accept) using the stored action
        - receive_offer: called when the opponent makes an offer
        - initiate: called before the negotiation session starts
        - name: unique string identifier for the agent
    """

    def __init__(self, preference: Preference, session_time: int, estimators: List[AbstractOpponentModel]):
        super().__init__(preference, session_time, estimators)

    @classmethod
    @abstractmethod
    def get_observation_space(cls) -> gym.Space:
        """
        Declare the observation space for the RL policy.

        The space must match exactly what build_observation() returns.
        Called by NegotiationEnv before any agent is instantiated.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_action_space(cls) -> gym.Space:
        """
        Declare the action space for the RL policy.

        The space must match exactly what set_action() receives.
        Called by NegotiationEnv before any agent is instantiated.
        """
        raise NotImplementedError

    @abstractmethod
    def build_observation(self) -> np.ndarray:
        """
        Build the current RL observation from negotiation state.

        Called by NegotiationEnv each step before the policy is queried.
        Must return a value conforming to get_observation_space().

        Available state from AbstractAgent:
            self.preference          -- own utility function
            self.last_received_bids  -- list of bids received from opponent
            self.session_time        -- deadline (rounds)
        """
        raise NotImplementedError

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """
        Receive and store the action chosen by the RL policy.

        Called by NegotiationEnv immediately before act(t) each step.
        Store the action here and use it inside act(t) to decide what to offer.
        """
        raise NotImplementedError

    def get_extra_info(self) -> dict:
        """
        Return agent-specific metrics to include in the step info dict.

        Override this to expose custom per-step data (e.g. internal targets,
        model outputs, gap metrics) to W&B callbacks running in the main process.

        The returned dict is merged into the info dict that NegotiationEnv
        returns from step() and reset(). Keys must be serialisable across
        subprocess boundaries (primitives only — no numpy arrays or objects).

        Example:
            def get_extra_info(self) -> dict:
                return {"my_target": self._target, "gap": self._gap}
        """
        return {}
