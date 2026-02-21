from typing import Any, Dict, List, Optional
from aci import ACI

class ACITracker:
    """Thin wrapper around the external ACI library for agent-side tracking."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.05,
        lookback: int = 500,
    ):
        self.target_alpha = float(alpha)
        self.gamma = float(gamma)
        self.lookback = int(lookback)
        self.reset()

    def issue(self, prediction: float, current_round: int) -> bool:
        """
        Issue one ACI interval from a point prediction (median).

        Args:
            prediction: Point prediction (float).
            current_round: Current negotiation round.

        Returns False if ACI already has an active issued prediction.
        """
        if self.aci.has_pending_prediction:
            return False

        interval = self.aci.issue(prediction)
        self.issue_history.append(
            {
                "round": int(current_round),
                "predicted": prediction,
                "interval": interval,
                "alpha_used": float(self.aci.alpha_t),
            }
        )
        return True

    def observe(self, actual_value: float, current_round: int) -> Optional[Dict[str, Any]]:
        """
        Observe the realized value and update ACI for the active issued prediction.

        Returns the raw `aci.observe(...)` output when an update occurs, else None.
        """
        if not self.aci.has_pending_prediction:
            return None

        actual = float(actual_value)
        issued = self.issue_history[-1]
        predicted = float(issued["predicted"])
        now = int(current_round)
        issued_interval = issued.get("interval", (predicted, predicted))
        if isinstance(issued_interval, (tuple, list)) and len(issued_interval) == 2:
            issued_upper = float(issued_interval[1])
        else:
            issued_upper = predicted
        issued_upper_interval_width = max(0.0, issued_upper - predicted)

        # Ignore all points below median:
        # no ACI observe/update, treated as hit for plotting only.
        if actual < predicted:
            interval = issued_interval
            self._clear_pending_without_update()
            result = {
                "round": now,
                "predicted": predicted,
                "actual": actual,
                "interval": interval,
                "upper_interval_width": issued_upper_interval_width,
                "residual": 0.0,
                "hit": True,
                "ignored": True,
                "alpha_used": float(self.aci.alpha_t),
                "alpha_next": float(self.aci.alpha_t),
            }
            self.observation_history.append(result)
            return result

        # Upper side (actual >= median): standard vanilla ACI update.
        out = self.aci.observe(actual)
        interval = out["prediction_set"]
        if isinstance(interval, (tuple, list)) and len(interval) == 2:
            interval_upper = float(interval[1])
        else:
            interval_upper = predicted
        upper_interval_width = max(0.0, interval_upper - predicted)

        self.total_predictions += 1
        self.total_errors += int(out["err_t"])

        self.alpha_history.append(
            {
                "round": now,
                "alpha_used": float(out["alpha_used"]),
                "alpha_next": float(out["alpha_next"]),
            }
        )

        result = {
            "round": now,
            "predicted": predicted,
            "actual": actual,
            "interval": interval,
            "upper_interval_width": upper_interval_width,
            "residual": float(out["score_t"]),
            "hit": bool(out["hit"]),
            "ignored": False,
            "alpha_used": float(out["alpha_used"]),
            "alpha_next": float(out["alpha_next"]),
        }
        self.observation_history.append(result)
        return result

    def get_observation_history(self) -> List[Dict[str, Any]]:
        return self.observation_history

    def get_alpha_history(self) -> List[Dict[str, Any]]:
        return self.alpha_history

    def get_empirical_error_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_errors / self.total_predictions

    def _clear_pending_without_update(self) -> None:
        """
        Drop currently issued ACI prediction without updating alpha/history.
        Needed for intentionally ignored observations.
        """
        if hasattr(self.aci, "_issued"):
            self.aci._issued = None

    def reset(self) -> None:
        self.aci = ACI(alpha=self.target_alpha, gamma=self.gamma, lookback=self.lookback)
        self.total_predictions = 0
        self.total_errors = 0
        self.issue_history: List[Dict[str, Any]] = []
        self.observation_history: List[Dict[str, Any]] = []
        self.alpha_history: List[Dict[str, Any]] = []
