import math
from numba import njit
from typing import List, Union

@njit
def populate_time_list(t: float, number_of_sample: int, estimated_round_time: float) -> List[float]:
    """
        Faster implementation to populate time steps.
    :param t: Current negotiation time
    :param number_of_sample: Number of required samples.
    :param estimated_round_time: Estimated round time
    :return: List of time-steps
    """
    time_samples = []

    for i in range(number_of_sample):
        time_samples.append(t)

        t += estimated_round_time

        if t > 1.0:
            break

    return time_samples

class TimeEstimator:
    """
        This component estimates the number of remaining rounds.
    """
    last_receive_time: float
    total_differences: float
    total_counting: int
    number_of_rounds: int

    def __init__(self):
        self.last_receive_time = 0.
        self.total_differences = 0.
        self.total_counting = 0
        self.number_of_rounds = 0

    def update(self, t: float):
        """
            This method updates the parameter when a bid is received.
        :param t: Current negotiation time
        :return: Nothing
        """
        diff = t - self.last_receive_time

        self.last_receive_time = t

        self.number_of_rounds += 1
        self.total_differences += diff * self.number_of_rounds
        self.total_counting += self.number_of_rounds

    def estimated_round_time(self) -> Union[float, None]:
        """
            This method estimates the required time for a round. It employs weighted averaging approach.
        :return: Required normalized negotiation time for a round.
        """
        if self.number_of_rounds < 2:
            return None

        return self.total_differences / self.total_counting

    def get_remaining_round(self, t: float) -> Union[int, None]:
        """
            This method estimates the number of remaining rounds.
        :param t: Current negotiation time
        :return: Number of remaining rounds
        """
        remaining_time = 1. - t

        return math.floor(remaining_time / self.estimated_round_time())

    def populate(self, t: float, number_of_sample: int) -> List[float]:
        """
            This method is required to NegoFormer component. It provides next time-steps.
        :param t: Current negotiation time
        :param number_of_sample: Number of required sample
        :return: List of time-steps
        """
        estimated_round_time = self.estimated_round_time()

        return populate_time_list(t, number_of_sample, estimated_round_time)