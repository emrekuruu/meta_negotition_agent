from typing import Optional
import nenv
from nenv import Bid, Action, Offer


class FakeoutAgent(nenv.AbstractAgent):
    """
    Test agent with repeating sawtooth pattern:
    - Concede: 1.0 → 0.7 over 0.13 time
    - Ascend:  0.7 → 1.0 over 0.04 time
    - Repeat

    This creates a zigzag pattern that repeatedly tricks the forecaster.
    """

    # Cycle parameters
    CONCEDE_DURATION = 0.13  # Time to go from 1.0 to 0.7
    ASCEND_DURATION = 0.17   # Time to go from 0.7 to 1.0
    HIGH = 1.0
    LOW = 0.7

    @property
    def name(self) -> str:
        return "Fakeout"

    def initiate(self, opponent_name: Optional[str]):
        pass

    def receive_offer(self, bid: Bid, t: float):
        pass

    def act(self, t: float) -> Action:
        cycle_length = self.CONCEDE_DURATION + self.ASCEND_DURATION  # 0.17

        # Where are we in the current cycle?
        cycle_position = t % cycle_length

        if cycle_position <= self.CONCEDE_DURATION:
            # Conceding phase: 1.0 → 0.7
            progress = cycle_position / self.CONCEDE_DURATION
            target_utility = self.HIGH - progress * (self.HIGH - self.LOW)
        else:
            # Ascending phase: 0.7 → 1.0
            ascend_position = cycle_position - self.CONCEDE_DURATION
            progress = ascend_position / self.ASCEND_DURATION
            target_utility = self.LOW + progress * (self.HIGH - self.LOW)

        # Target utility cannot be lower than reservation value
        if target_utility < self.preference.reservation_value:
            target_utility = self.preference.reservation_value

        # AC_Next
        if self.can_accept() and target_utility <= self.last_received_bids[-1]:
            return self.accept_action

        # Find closest bid to target utility
        bid = self.preference.get_bid_at(target_utility)

        return Offer(bid)
