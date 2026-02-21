import torch
import os
import numpy as np
from abc import ABC, abstractmethod


class BaseForecaster(ABC):
    """
    Abstract base class for time series forecasters.

    Subclasses must implement:
    - _initialize_model(): Load the forecasting model
    - _prepare_data(data): Convert input data to model-specific format
    - _forecast(prepared_data): Generate point forecast from prepared data
    """

    def __init__(self, max_context=512, prediction_length=256, deadline=1000):
        self.max_context = max_context
        self.prediction_length = prediction_length
        self.deadline = deadline
        self.device = self._get_device()
        self._initialize_model()

    def _get_device(self) -> str:
        """Get assigned GPU device for this worker process."""
        gpu_id = int(os.environ.get('WORKER_GPU_ID', 0))
        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{gpu_id}"

    @abstractmethod
    def _initialize_model(self):
        """Initialize the forecasting model."""
        pass

    @abstractmethod
    def _prepare_data(self, data: list[tuple[float, float]]):
        """
        Prepare data for the forecasting model.

        Args:
            data: List of (utility, timestamp) tuples

        Returns:
            Model-specific data format
        """
        pass

    @abstractmethod
    def _forecast(self, prepared_data) -> np.ndarray:
        """
        Generate point forecast from prepared data.

        Args:
            prepared_data: Data in model-specific format

        Returns:
            1D numpy array of predicted utilities, shape: (prediction_length,)
        """
        pass

    def __call__(self, inputs: list[tuple[float, float]]) -> np.ndarray:
        """
        Forecast opponent utility trajectory.

        Args:
            inputs: List of (utility, timestamp) tuples from opponent's offers

        Returns:
            numpy array of predicted utilities, shape: (prediction_length,)
        """
        prepared_data = self._prepare_data(inputs)
        forecast = self._forecast(prepared_data)
        return forecast


def Forecaster(max_context=512, prediction_length=256, deadline=1000):
    """
    Factory function to create appropriate forecaster.

    Environment Variable:
        FORECASTER_MODEL: 'toto' (default) or 'chronos'
    """
    from .models.toto import TotoForecaster
    from .models.chronos import ChronosForecaster

    model_type = os.environ.get('FORECASTER_MODEL', 'toto').lower()

    if model_type == 'toto':
        return TotoForecaster(max_context, prediction_length, deadline)
    elif model_type == 'chronos':
        return ChronosForecaster(max_context, prediction_length, deadline)
    else:
        raise ValueError(
            f"Invalid FORECASTER_MODEL: '{model_type}'. "
            f"Must be 'toto' or 'chronos'."
        )
