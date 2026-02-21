from typing import Dict, List
import numpy as np
import nenv
from gym_enviroment.config.config import config
from gym_enviroment.agent.common.offer_point import OfferPoint, process_offer_history

class ObservationBuilder:
    """
    Stateless observation building for RL integration.
    
    Uses the shared process_offer_history function for consistency with forecaster.
    """
    
    @staticmethod
    def build_observation(candidate_forecasts: List[np.ndarray],
                         offer_points: List[OfferPoint],
                         estimated_preference: nenv.OpponentModel.EstimatedPreference) -> Dict[str, np.ndarray]:
        """
        Build observation dictionary for RL agent from OfferPoint history.
        
        Args:
            candidate_forecasts: List of forecast sequences for candidates (passed through)
            offer_points: List of OfferPoint objects representing negotiation history
            estimated_preference: Estimated opponent preference for feature extraction
            
        Returns:
            Dictionary with 'negotiation_history' and 'candidate_forecasts' keys
        """
        # Build negotiation history using shared function
        negotiation_history = process_offer_history(
            offer_points, estimated_preference, config.core['sequence_length']
        )
        
        # Convert candidate forecasts to proper array format
        candidate_array = np.zeros((config.core['max_candidates'], 
                                  config.core['forecast_length']), 
                                 dtype=np.float32)
        
        # Fill with actual forecasts (guaranteed to be correct size when present)
        if candidate_forecasts:
            for i, forecast in enumerate(candidate_forecasts[:config.core['max_candidates']]):
                candidate_array[i] = forecast
        

        obs =  {
            'negotiation_history': negotiation_history,
            'candidate_forecasts': candidate_array,
            'statistical_features': ObservationBuilder.create_statistical_features(negotiation_history)
        }

        return obs
    
    @staticmethod
    def is_positive_move(offer_point: np.ndarray) -> int:
        return offer_point[5] + offer_point[7] + offer_point[10]

    @staticmethod
    def is_negative_move(offer_point: np.ndarray) -> int:
        return offer_point[6] + offer_point[8] + offer_point[9]
    
    @staticmethod
    def awareness_correlation(processed_offer_points: List[np.ndarray]) -> float:
        cause = []
        effect = [] 
        
        for i in range(len(processed_offer_points) -1):
            current = processed_offer_points[i]
            next_ = processed_offer_points[i + 1]

            if current[1] == 1 and next_[1] == 0:
                our_signal = ObservationBuilder.is_positive_move(current) - ObservationBuilder.is_negative_move(current)
                opponent_signal = ObservationBuilder.is_positive_move(next_) - ObservationBuilder.is_negative_move(next_)

                cause.append(our_signal)
                effect.append(opponent_signal)

        correlation = np.corrcoef(cause, effect)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return correlation

    @staticmethod
    def create_statistical_features(processed_offer_points: List[np.ndarray]) -> np.ndarray:

        awareness_corr = ObservationBuilder.awareness_correlation(processed_offer_points)

        mean_agent_util = np.mean([p[2] for p in processed_offer_points])
        mean_opponent_util = np.mean([p[-1] for p in processed_offer_points])

        std_agent_util = np.std([p[2] for p in processed_offer_points]) 
        std_opponent_util = np.std([p[-1] for p in processed_offer_points])

        min_agent_util = np.min([p[2] for p in processed_offer_points])
        min_opponent_util = np.min([p[-1] for p in processed_offer_points])

        max_agent_util = np.max([p[2] for p in processed_offer_points])
        max_opponent_util = np.max([p[-1] for p in processed_offer_points])

        return np.array([awareness_corr, mean_agent_util, mean_opponent_util, std_agent_util, std_opponent_util, min_agent_util, min_opponent_util, max_agent_util, max_opponent_util])
