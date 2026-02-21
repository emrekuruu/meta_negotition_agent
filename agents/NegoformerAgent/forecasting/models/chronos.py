import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
from ..forecaster import BaseForecaster

class ChronosForecaster(BaseForecaster):
    """Chronos-2 time series forecaster returning point predictions only."""

    def _initialize_model(self):
        """
        Load Chronos-2 model with constraint validation.

        Raises:
            ValueError: If max_context or prediction_length exceed Chronos-2 limits
        """
        if self.max_context > 8192:
            raise ValueError(
                f"Chronos-2 max_context ({self.max_context}) exceeds limit of 8192"
            )
        if self.prediction_length > 1024:
            raise ValueError(
                f"Chronos-2 prediction_length ({self.prediction_length}) exceeds limit of 1024"
            )

        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=self.device
        )

    def _prepare_data(self, data: list[tuple[float, float]]) -> pd.DataFrame:
        """
        Prepare data for Chronos-2 model using DataFrame format.

        Args:
            data: List of (utility, timestamp) tuples

        Returns:
            DataFrame with regular timestamps and target values
        """
        utilities = [utility for utility, _ in data]

        if len(utilities) > self.max_context:
            utilities = utilities[-self.max_context:]

        timestamps = pd.date_range(
            start='2024-01-01',
            periods=len(utilities),
            freq='1s'
        )

        context_df = pd.DataFrame({
            'id': ['opponent_utility'] * len(utilities),
            'timestamp': timestamps,
            'target': utilities
        })

        return context_df

    def _forecast(self, prepared_data: pd.DataFrame) -> np.ndarray:
        """
        Generate point forecast using Chronos-2.

        Args:
            prepared_data: DataFrame with historical data

        Returns:
            1D numpy array of predicted utilities (median), shape: (prediction_length,)
        """
        pred_df = self.pipeline.predict_df(
            prepared_data,
            prediction_length=self.prediction_length,
            quantile_levels=[0.5],  # Only median
            id_column='id',
            timestamp_column='timestamp',
            target='target'
        )

        # Find median column
        quantile_cols = [col for col in pred_df.columns if '0.5' in str(col) or 'median' in str(col).lower()]

        if not quantile_cols:
            raise ValueError(
                f"Could not find median quantile column. "
                f"Available columns: {list(pred_df.columns)}"
            )

        median_forecast = pred_df[quantile_cols[0]].values

        # Clip to valid utility range
        return np.clip(median_forecast, 0.0, 1.0)
