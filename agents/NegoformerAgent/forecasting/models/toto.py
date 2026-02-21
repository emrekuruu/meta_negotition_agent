import torch
import numpy as np
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster as TotoModel
from toto.data.util.dataset import MaskedTimeseries
from ..forecaster import BaseForecaster


class TotoForecaster(BaseForecaster):
    """Toto-based time series forecaster returning point predictions only."""

    def _initialize_model(self):
        """Load Toto model."""
        self.toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(self.device)
        self.model = TotoModel(self.toto.model)

    def _prepare_data(self, data: list[tuple[float, float]]) -> MaskedTimeseries:
        """
        Prepare negotiation data for TOTO forecasting.

        Args:
            data: List of (utility, timestamp) tuples

        Returns:
            MaskedTimeseries object ready for forecasting
        """
        utilities = [utility for utility, _ in data]
        timestamps = [t for _, t in data]

        if len(utilities) > self.max_context:
            utilities = utilities[-self.max_context:]
            timestamps = timestamps[-self.max_context:]

        utilities_tensor = torch.tensor(utilities, dtype=torch.float32).unsqueeze(0).to(self.device)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32).unsqueeze(0).to(self.device)
        time_interval_seconds = torch.tensor([1.0/self.deadline], dtype=torch.float32).to(self.device)

        padding_mask = torch.ones_like(utilities_tensor, dtype=torch.bool).to(self.device)
        id_mask = torch.zeros_like(utilities_tensor, dtype=torch.long).to(self.device)

        inputs = MaskedTimeseries(
            series=utilities_tensor,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps_tensor,
            time_interval_seconds=time_interval_seconds,
        )

        return inputs

    def _forecast(self, prepared_data: MaskedTimeseries) -> np.ndarray:
        """
        Generate point forecast using Toto model.

        Args:
            prepared_data: MaskedTimeseries object

        Returns:
            1D numpy array of predicted utilities (median), shape: (prediction_length,)
        """
        forecast_result = self.model.forecast(
            prepared_data,
            prediction_length=self.prediction_length,
            num_samples=256,  
            samples_per_batch=256,
        )

        # Return median forecast, clipped to valid utility range
        median_forecast = forecast_result.median.cpu().numpy().squeeze()
        return np.clip(median_forecast, 0.0, 1.0)
