from nenv.OpponentModel.EstimatedPreference import EstimatedPreference
from nenv.Preference import Preference

class UniformEstimatedPreference(EstimatedPreference):
    """
        Uniform initialization for opponent model preferences.
        All issue weights are initialized uniformly (1/N for N issues).
        All value weights within each issue are initialized uniformly (1/M for M values).
    """
    def __init__(self, reference: Preference):
        """
            Constructor
        :param reference: Reference Preference to get domain information.
        """
        super().__init__(reference)

    def initialize_weights(self, reference: Preference):
        """
            Initialize all weights uniformly.

        :param reference: Reference Preference to get domain information.
        """
        num_issues = len(self._issue_weights)

        for issue in self._issue_weights.keys():
            self._issue_weights[issue] = 1.0 / num_issues

            num_values = len(issue.values)

            for value in issue.values:
                self._value_weights[issue][value] = 1.0 / num_values

        self.normalize()