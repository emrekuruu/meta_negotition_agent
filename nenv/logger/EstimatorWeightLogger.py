from nenv.logger.AbstractLogger import AbstractLogger, Bid, SessionLogs, Session, LogRow
from typing import Union, List
from nenv.Agent import AbstractAgent


class EstimatorWeightLogger(AbstractLogger):
    """
    Logger that tracks opponent model weight estimates at each round.

    Stores issue weights and value weights for all active estimators, enabling proper
    forecaster ablation that matches NegoformerAgent's actual behavior. This logger
    extracts the weight tracking functionality from EstimatorMetricLoggerWithVisualizations
    into a standalone logger.

    The logger creates sheets named `Weights` containing:
    - Issue weights for both agents
    - Value weights for both agents
    - Tracks how Agent A estimates Agent B's preferences and vice versa

    This allows forecaster ablation to recalculate historical bid utilities with any
    round's opponent model, replicating how NegoformerAgent updates all historical
    bids with the current opponent model before forecasting.
    """

    def before_session_start(self, session: Union[Session, SessionLogs]) -> List[str]:
        """
        Register sheet names for all active estimators before the session starts.

        Args:
            session: The session that is about to start

        Returns:
            List of sheet names to create in the Excel log
        """
        sheet_names = []

        # Get all active estimators and create sheet names for them
        if len(session.agentA.estimators) > 0:
            for estimator in session.agentA.estimators:
                sheet_names.append(f"{estimator.name} Weights")

        return sheet_names

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        """Log weights when an offer is made."""
        return self.get_preference_tracking(session.agentA, session.agentB)

    def on_accept(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        """Log weights when an offer is accepted."""
        return self.get_preference_tracking(session.agentA, session.agentB)

    def on_fail(self, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        """Log weights when negotiation fails."""
        return self.get_preference_tracking(session.agentA, session.agentB)

    def get_preference_tracking(self, agent_a: AbstractAgent, agent_b: AbstractAgent) -> LogRow:
        """
        Track estimated preference weights at each round.

        For each active estimator, logs:
        - Agent A's estimation of Agent B's issue weights
        - Agent A's estimation of Agent B's value weights
        - Agent B's estimation of Agent A's issue weights
        - Agent B's estimation of Agent A's value weights

        Args:
            agent_a: First agent in the session
            agent_b: Second agent in the session

        Returns:
            LogRow with format: {"{estimator_name} Weights": {weight_dict}}
        """
        row = {}

        for estimator_id in range(len(agent_a.estimators)):
            estimator_name = agent_a.estimators[estimator_id].name

            # Get estimated preferences
            pref_a_estimates_b = agent_a.estimators[estimator_id].preference
            pref_b_estimates_a = agent_b.estimators[estimator_id].preference

            # Create tracking log
            tracking_log = {}

            # Track Agent A's estimation of Agent B's preferences
            for issue, weight in pref_a_estimates_b.issue_weights.items():
                tracking_log[f"A_estimates_B_{issue.name}_issue_weight"] = weight

            for issue, value_weights in pref_a_estimates_b.value_weights.items():
                for value_name, weight in value_weights.items():
                    tracking_log[f"A_estimates_B_{issue.name}_{value_name}_weight"] = weight

            # Track Agent B's estimation of Agent A's preferences
            for issue, weight in pref_b_estimates_a.issue_weights.items():
                tracking_log[f"B_estimates_A_{issue.name}_issue_weight"] = weight

            for issue, value_weights in pref_b_estimates_a.value_weights.items():
                for value_name, weight in value_weights.items():
                    tracking_log[f"B_estimates_A_{issue.name}_{value_name}_weight"] = weight

            row[f"{estimator_name} Weights"] = tracking_log

        return row
