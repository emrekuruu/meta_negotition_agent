from typing import Union, List
import numpy as np
import nenv
from gym_enviroment.config.config import config

class OfferPoint:
    """
        Offer point contains required features in a time-step for NegoFormer model
    """
    who: int
    bid: nenv.Bid
    t: float

    def __init__(self, who: int, bid: nenv.Bid, t: float):
        self.who = who
        self.bid = bid
        self.t = t


def process(prev_offer_point: Union[OfferPoint, None], offer_point: OfferPoint, t: float, estimated_preference: nenv.OpponentModel.EstimatedPreference) -> np.ndarray:
    """
    This method converts Offer Point object into NumPy vector
    :param prev_offer_point: Previous offer point
    :param offer_point: Current offer point
    :param t: Current negotiation time
    :param estimated_preference: Estimated opponent preferences
    :return: NumPy vector for Autoformer model.
    """
    move = ''
    if prev_offer_point is not None:
    
        if offer_point.who == 1:
            move = nenv.utils.get_move(prev_offer_point.bid.utility,
                                       offer_point.bid.utility,
                                       estimated_preference.get_utility(prev_offer_point.bid),
                                       estimated_preference.get_utility(offer_point.bid))
        else:
            move = nenv.utils.get_move(estimated_preference.get_utility(prev_offer_point.bid),
                                       estimated_preference.get_utility(offer_point.bid),
                                       prev_offer_point.bid.utility,
                                       offer_point.bid.utility)
    
    return np.array([
        t,
        offer_point.who,
        offer_point.bid.utility,
        offer_point.bid.utility * estimated_preference.get_utility(offer_point.bid),
        offer_point.bid.utility + estimated_preference.get_utility(offer_point.bid),
        int(move == 'Concession'),
        int(move == 'Selfish'),
        int(move == 'Fortunate'),
        int(move == 'Unfortunate'),
        int(move == 'Silent'),
        int(move == 'Nice'),
        estimated_preference.get_utility(offer_point.bid),
    ])


def process_offer_history(offer_points: List[OfferPoint], 
                         estimated_preference: nenv.OpponentModel.EstimatedPreference,
                         sequence_length: int) -> np.ndarray:
    """
    Convert a list of OfferPoints into a feature array for ML models.
    
    Used by both forecaster and observation builder for consistency.
    
    Args:
        offer_points: List of OfferPoint objects representing negotiation history
        estimated_preference: Estimated opponent preference for feature extraction
        sequence_length: Length of output sequence (e.g., 96)
        
    Returns:
        Array of shape (sequence_length, config.core['negotiation_features']) with processed features
    """
    feature_count = config.core['negotiation_features']  # From process() function output
    history = np.zeros((sequence_length, feature_count), dtype=np.float32)
    
    if not offer_points:
        return history
    
    # Use the most recent sequence_length entries
    recent_offers = offer_points[-sequence_length:] if len(offer_points) > sequence_length else offer_points
    
    # Process each offer point into features
    for i, offer_point in enumerate(recent_offers):
        # Calculate actual index in output array
        output_idx = sequence_length - len(recent_offers) + i
        
        # Get previous offer point for move calculation
        prev_offer = recent_offers[i-2] if i > 1 else None
        
        # Convert OfferPoint to feature vector using existing process function
        features = process(prev_offer, offer_point, offer_point.t, estimated_preference)
        
        history[output_idx] = features
    
    return history