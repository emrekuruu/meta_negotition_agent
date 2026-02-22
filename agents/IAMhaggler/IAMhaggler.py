"""
Complete rewrite of IAMhaggler to exactly match Java implementation.
Based on IAMhaggler2011 from ANAC 2012.
"""
import math
import warnings
import numpy as np
from typing import List, Optional, Tuple
from scipy.special import erf as scipy_erf
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.exceptions import ConvergenceWarning
import nenv


class IAMhaggler(nenv.AbstractAgent):
    """
    IAMhaggler agent by Colin R. Williams.
    ANAC 2012 Nash category winner.

    This is a complete rewrite to exactly match the Java implementation.
    """

    # Constants
    RISK_PARAMETER: float = 1.0
    MAXIMUM_ASPIRATION: float = 0.9
    acceptMultiplier: float = 1.02

    # Java Matrix equivalents (using numpy)
    utilitySamples: np.ndarray  # Column vector (m, 1)
    timeSamples: np.ndarray     # Row vector (1, n+1)
    utility: np.ndarray          # Pre-computed utility surface (m, n+1)
    means: Optional[np.ndarray]  # Column vector (n+1, 1)
    variances: Optional[np.ndarray]  # Column vector (n+1, 1)
    matrixTimeSamplesAdjust: Optional[np.ndarray]  # Column vector (n+1, 1)

    # GP state
    gp: Optional[GaussianProcessRegressor]
    opponentTimes: List[float]
    opponentUtilities: List[float]

    # Agent state
    lastRegressionTime: float
    lastRegressionUtility: float
    maxUtilityInTimeSlot: float
    lastTimeSlot: int
    maxUtility: float
    bestReceivedBid: Optional[nenv.Bid]
    previousTargetUtility: float
    intercept: float
    maxOfferedUtility: float
    minOfferedUtility: float
    discounting_factor: float

    def __init__(self, preference: nenv.Preference, session_time: int,
                 estimators: List[nenv.OpponentModel.AbstractOpponentModel]):
        super().__init__(preference, session_time, estimators)
        self.session_time = session_time

    @property
    def name(self) -> str:
        return "IAMhaggler"

    def initiate(self, opponent_name: Optional[str]):
        """Initialize agent - matches Java init() method."""
        # Get discounting factor (Java lines 79-91)
        self.discounting_factor = 1.0  # Default for nenv (no discounting)

        # Create utility samples (Java lines 136-145)
        # m-by-1 column vector
        m = 100
        utility_samples_array = np.array([1.0 - (i + 0.5) / (m + 1.0) for i in range(m)])
        self.utilitySamples = utility_samples_array.reshape(m, 1)  # Column vector

        # Create time samples (Java lines 153-161)
        # 1-by-(n+1) row vector
        n = 100
        time_samples_array = np.array([i / n for i in range(n + 1)])
        self.timeSamples = time_samples_array.reshape(1, n + 1)  # Row vector

        # Pre-compute utility surface (Java lines 95-97)
        discounting = self.generateDiscountingFunction(self.discounting_factor)
        risk = self.generateRiskFunction(self.RISK_PARAMETER)
        self.utility = risk * discounting  # Element-wise multiplication

        # Initialize GP (Java lines 104-112)
        # Note: Java uses GaussianProcessRegressionBMC with priors
        # We use scikit-learn GP as close approximation
        # The data can be nearly noiseless; allow smaller noise floor to avoid
        # hitting the default lower bound too early.
        kernel = kernels.Matern(nu=1.5) + kernels.WhiteKernel(
            noise_level=1e-6,
            noise_level_bounds=(1e-10, 1e1),
        )
        self.gp = GaussianProcessRegressor(kernel, alpha=1e-6, normalize_y=False)
        self._gp_X = []  # Accumulated training data
        self._gp_y = []

        # Initialize state (Java lines 116-117, 37-60)
        self.maxUtility = 0.0
        self.previousTargetUtility = 1.0
        self.lastRegressionTime = 0.0
        self.lastRegressionUtility = 1.0
        self.opponentTimes = []
        self.opponentUtilities = []
        self.maxUtilityInTimeSlot = 0.0
        self.lastTimeSlot = -1
        self.means = None
        self.variances = None
        self.bestReceivedBid = None
        self.intercept = 0.5  # Will be set when first slot closes
        self.matrixTimeSamplesAdjust = None
        self.maxOfferedUtility = float('-inf')
        self.minOfferedUtility = float('inf')

    def receive_offer(self, bid: nenv.Bid, t: float):
        """Record opponent bid - called before act()."""
        self.last_received_bids.append(bid)

    def act(self, t: float) -> nenv.Action:
        """
        Choose action - matches Java chooseAction() flow.
        Combines proposeInitialBid, proposeNextBid, and handleOffer logic.
        """
        # First action: return max utility bid (Java proposeInitialBid, line 170)
        if not self.can_accept():
            max_bid = self.preference.bids[0]  # Assumes sorted descending by utility
            self.previousTargetUtility = max_bid.utility
            return nenv.Offer(max_bid)

        # Get opponent's last bid
        opponent_bid = self.last_received_bids[-1]
        opponent_utility = self.preference.get_utility(opponent_bid)

        # Update best received (Java lines 182-186)
        if opponent_utility > self.maxUtility:
            self.bestReceivedBid = opponent_bid
            self.maxUtility = opponent_utility

        # Get target utility (Java line 190)
        target_utility = self.getTarget(opponent_utility, t)

        # Special acceptance: crossing threshold (Java lines 194-195)
        if target_utility <= self.maxUtility and self.previousTargetUtility > self.maxUtility:
            return nenv.Offer(self.bestReceivedBid)

        self.previousTargetUtility = target_utility

        # Plan next bid in range [target - 0.025, target + 0.025] (Java lines 200-202)
        planned_bid = self.preference.get_random_bid(
            target_utility - 0.025, target_utility + 0.025)

        # Acceptance logic (Java handleOffer in SouthamptonAgent)
        # Check 1: opponent utility * multiplier >= my last bid utility
        if opponent_utility * self.acceptMultiplier >= self.previousTargetUtility:
            return self.accept_action

        # Check 2: opponent utility * multiplier >= MAXIMUM_ASPIRATION
        if opponent_utility * self.acceptMultiplier >= self.MAXIMUM_ASPIRATION:
            return self.accept_action

        # Check 3: opponent utility * multiplier >= planned bid utility
        if opponent_utility * self.acceptMultiplier >= planned_bid.utility:
            return self.accept_action

        # Otherwise, offer the planned bid
        return nenv.Offer(planned_bid)

    def getTarget(self, opponent_utility: float, time: float) -> float:
        """
        Calculate target utility based on opponent behavior.
        Exact match to Java getTarget() method (lines 214-363).
        """
        # Update concession limiter bounds (Java lines 219-220)
        self.maxOfferedUtility = max(self.maxOfferedUtility, opponent_utility)
        self.minOfferedUtility = min(self.minOfferedUtility, opponent_utility)

        # Calculate current time slot (Java line 223)
        time_slot = int(math.floor(time * 36))

        # Check if regression update required (Java lines 225-228)
        regression_update_required = False
        if self.lastTimeSlot == -1:
            regression_update_required = True

        # Handle time slot changes (Java lines 231-254)
        if time_slot != self.lastTimeSlot:
            if self.lastTimeSlot != -1:  # NOT on first slot change!
                # Store the data from the closed time slot (Java line 234)
                slot_time = (self.lastTimeSlot + 0.5) / 36.0
                self.opponentTimes.append(slot_time)

                # Set intercept on FIRST CLOSED SLOT (Java lines 235-245)
                if len(self.opponentUtilities) == 0:
                    self.intercept = max(0.5, self.maxUtilityInTimeSlot)
                    gradient = 0.9 - self.intercept
                    # Create matrixTimeSamplesAdjust ONCE
                    time_adjust = np.array([self.intercept + gradient * t
                                           for t in self.timeSamples.flatten()])
                    self.matrixTimeSamplesAdjust = time_adjust.reshape(-1, 1)  # Column vector

                self.opponentUtilities.append(self.maxUtilityInTimeSlot)
                regression_update_required = True

            # Update the time slot (Java line 251)
            self.lastTimeSlot = time_slot
            # Reset max utility for new slot (Java line 253)
            self.maxUtilityInTimeSlot = 0.0

        # Update max utility in current slot (Java line 259)
        self.maxUtilityInTimeSlot = max(self.maxUtilityInTimeSlot, opponent_utility)

        # Early return for slot 0 (Java lines 261-263)
        if time_slot == 0:
            return 1.0 - time / 2.0

        # Perform regression update if required (Java lines 265-328)
        if regression_update_required:
            gradient = 0.9 - self.intercept

            # Handle empty regression vs incremental update (Java lines 297-317)
            if self.lastTimeSlot == -1:
                # Empty regression - no training data yet
                # Just use prior (matrixTimeSamplesAdjust)
                self.means = self.matrixTimeSamplesAdjust.copy()
                self.variances = np.zeros_like(self.means)
            else:
                # Get last added point for incremental update (Java lines 306-307)
                x = self.opponentTimes[-1]
                y = self.opponentUtilities[-1]

                # Adjust y (Java line 316)
                y_adjusted = y - self.intercept - (gradient * x)

                # Accumulate training data
                self._gp_X.append([x])
                self._gp_y.append(y_adjusted)

                # Train GP on all accumulated data
                X_train = np.array(self._gp_X)
                y_train = np.array(self._gp_y).ravel()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=ConvergenceWarning,
                        message=r".*k2__noise_level is close to the specified lower bound.*",
                    )
                    self.gp.fit(X_train, y_train)

                # Predict on time samples (Java lines 319-320)
                time_samples_col = self.timeSamples.T  # Convert to column for prediction
                mu, sigma = self.gp.predict(time_samples_col, return_std=True)

                # Add back adjustment (Java line 323)
                self.means = (mu.reshape(-1, 1) + self.matrixTimeSamplesAdjust)
                self.variances = (sigma ** 2).reshape(-1, 1)

        # Generate probability acceptance matrices (Java lines 330-333)
        prob_accept, cum_accept = self.generateProbabilityAccept(
            self.means, self.variances, time)

        # Multiply by utility surface (Java lines 335-336)
        prob_expected_utility = prob_accept * self.utility
        cum_expected_utility = cum_accept * self.utility

        # Get best agreement (Java lines 345-348)
        best_time, best_utility = self.getExpectedBestAgreement(
            prob_expected_utility, cum_expected_utility, time)

        # Interpolate to current time (Java lines 350-352)
        target_utility = self.lastRegressionUtility + \
            ((time - self.lastRegressionTime) *
             (best_utility - self.lastRegressionUtility) /
             (best_time - self.lastRegressionTime))

        # Store for next iteration (Java lines 357-358)
        self.lastRegressionUtility = target_utility
        self.lastRegressionTime = time

        # Apply concession limiter (Java line 362)
        return self.limitConcession(target_utility)

    def limitConcession(self, target_utility: float) -> float:
        """Limit concession based on opponent range (Java lines 365-373)."""
        limit = 1.0 - ((self.maxOfferedUtility - self.minOfferedUtility) + 0.1)
        if limit > target_utility:
            return limit
        return target_utility

    def generateDiscountingFunction(self, discounting_factor: float) -> np.ndarray:
        """
        Generate m-by-(n+1) discounting matrix (Java lines 385-395).
        Each row i has [df^t0, df^t1, ..., df^tn] where tj are time samples.
        """
        time_samples_1d = self.timeSamples.flatten()
        m = self.utilitySamples.shape[0]
        n_plus_1 = self.timeSamples.shape[1]

        # Create matrix where each row is [df^t0, df^t1, ...]
        discounting = np.zeros((m, n_plus_1))
        for i in range(m):
            for j in range(n_plus_1):
                discounting[i, j] = math.pow(discounting_factor, time_samples_1d[j])

        return discounting

    def generateRiskFunction(self, risk_parameter: float) -> np.ndarray:
        """
        Generate m-by-(n+1) risk-adjusted utility matrix (Java lines 488-509).
        Each row i has constant value: normalized(utility_i ^ risk_parameter).
        """
        # Calculate min and max for normalization (Java lines 489-491)
        r_min = self._generateRiskFunction_single(risk_parameter, 0.0)
        r_max = self._generateRiskFunction_single(risk_parameter, 1.0)
        r_range = r_max - r_min

        utility_samples_1d = self.utilitySamples.flatten()
        m = self.utilitySamples.shape[0]
        n_plus_1 = self.timeSamples.shape[1]

        # Create matrix where each row has constant normalized risk value
        risk = np.zeros((m, n_plus_1))
        for i in range(m):
            if r_range == 0:
                val = utility_samples_1d[i]
            else:
                val = (self._generateRiskFunction_single(risk_parameter, utility_samples_1d[i]) - r_min) / r_range

            for j in range(n_plus_1):
                risk[i, j] = val

        return risk

    def _generateRiskFunction_single(self, risk_parameter: float, utility: float) -> float:
        """Helper for single utility risk calculation (Java lines 520-522)."""
        return math.pow(utility, risk_parameter)

    def generateProbabilityAccept(self, mean: np.ndarray, variance: np.ndarray,
                                   time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate probability and cumulative acceptance matrices (Java lines 411-448).
        Returns (probabilityAccept, cumulativeAccept), both m-by-(n+1) matrices.
        """
        # Find first future time index (Java lines 413-417)
        i = 0
        time_samples_1d = self.timeSamples.flatten()
        for i in range(len(time_samples_1d)):
            if time_samples_1d[i] > time:
                break

        m = self.utilitySamples.shape[0]
        n_plus_1 = self.timeSamples.shape[1]

        # Initialize matrices (Java lines 418-421)
        cumulative_accept = np.zeros((m, n_plus_1))
        probability_accept = np.zeros((m, n_plus_1))

        # Interval for bin edges (Java line 423)
        interval = 1.0 / m

        # Process each future time column (Java lines 425-446)
        utility_samples_1d = self.utilitySamples.flatten()
        for col in range(i, n_plus_1):
            s = math.sqrt(2 * variance[col, 0])
            mu = mean[col, 0]

            # Calculate normalization range (Java lines 429-432)
            minp = 1.0 - 0.5 * (1 + self._erf(
                (utility_samples_1d[0] + interval/2.0 - mu) / s))
            maxp = 1.0 - 0.5 * (1 + self._erf(
                (utility_samples_1d[m-1] - interval/2.0 - mu) / s))

            # Calculate probabilities for each utility bin (Java lines 434-445)
            for row in range(m):
                util = utility_samples_1d[row]

                # Center probability (Java lines 436-437)
                p = 1.0 - 0.5 * (1 + self._erf((util - mu) / s))

                # Edge probabilities (Java lines 438-441)
                p1 = 1.0 - 0.5 * (1 + self._erf((util - interval/2.0 - mu) / s))
                p2 = 1.0 - 0.5 * (1 + self._erf((util + interval/2.0 - mu) / s))

                # Normalized probabilities (Java lines 443-444)
                cumulative_accept[row, col] = (p - minp) / (maxp - minp)
                probability_accept[row, col] = (p1 - p2) / (maxp - minp)

        return probability_accept, cumulative_accept

    def _erf(self, x: float) -> float:
        """Error function with clamping (Java lines 456-477)."""
        if x > 6:
            return 1.0
        if x < -6:
            return -1.0

        result = scipy_erf(x)

        # Clamp to [-1, 1]
        if result > 1:
            return 1.0
        if result < -1:
            return -1.0

        return result

    def getExpectedBestAgreement(self, prob_expected_values: np.ndarray,
                                  cum_expected_values: np.ndarray,
                                  time: float) -> Tuple[float, float]:
        """
        Find best time and utility combination (Java lines 537-588).
        Returns (best_time, best_utility).
        """
        # Get future-only slices (Java lines 541-542)
        prob_future = self.getFutureExpectedValues(prob_expected_values, time)
        cum_future = self.getFutureExpectedValues(cum_expected_values, time)

        # Find best column (time) by summing probabilityExpectedUtility (Java lines 552-566)
        col_sums = np.sum(prob_future, axis=0)
        best_col = 0
        best_col_sum = 0.0

        for x in range(col_sums.shape[0]):
            if col_sums[x] >= best_col_sum:  # >= prefers later time on tie
                best_col_sum = col_sums[x]
                best_col = x

        # Find best row (utility) in that column using cumulativeExpectedUtility (Java lines 570-579)
        best_row = 0
        best_row_value = 0.0

        for y in range(cum_future.shape[0]):
            expected_value = cum_future[y, best_col]
            if expected_value > best_row_value:
                best_row_value = expected_value
                best_row = y

        # Map back to original indices (Java lines 581-584)
        original_col_index = best_col + prob_expected_values.shape[1] - prob_future.shape[1]
        best_time = self.timeSamples[0, original_col_index]
        best_utility = self.utilitySamples[best_row, 0]

        return best_time, best_utility

    def getFutureExpectedValues(self, expected_values: np.ndarray, time: float) -> np.ndarray:
        """
        Get future-only slice of matrix (Java lines 602-611).
        Returns submatrix for times > current time.
        """
        # Find first future time index (Java lines 603-607)
        i = 0
        time_samples_1d = self.timeSamples.flatten()
        for i in range(len(time_samples_1d)):
            if time_samples_1d[i] > time:
                break

        # Return submatrix (Java lines 608-610)
        return expected_values[:, i:]
