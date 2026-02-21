import os
import torch
import numpy as np
from gym_enviroment.agent.forecasting.timexer.TimeXer import Model
from gym_enviroment.agent.common.offer_point import OfferPoint, process_offer_history
from types import SimpleNamespace
from typing import List, Tuple
import nenv
from gym_enviroment.config.config import config

class Forecaster:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_model("timexer/checkpoint.pth")
        
    def get_model_config(self):
        """Return a namespace object with all the parameters needed by the model"""
        model_params = {
            # Data parameters
            'features': 'MS',  # forecasting task, options:[M, S, MS]
            'seq_len': 96,     # input sequence length
            'label_len': 48,   # start token length
            'pred_len': 336,   # prediction sequence length
            'enc_in': 11,      # encoder input size (features excluding time)
            'dec_in': 11,      # decoder input size
            'c_out': 1,        # output size
            'use_norm': True,  # whether to use normalization

            # Model parameters
            'd_model': 512,    # dimension of model
            'n_heads': 8,      # number of heads
            'e_layers': 1,     # number of encoder layers
            'd_ff': 1024,      # dimension of fcn
            'factor': 3,       # attn factor
            'dropout': 0.1,    # dropout rate
            'activation': 'gelu',  # activation function

            # Patching parameters
            'patch_len': 16,   # patch length for patching

            # Embedding parameters
            'embed': 'timeF',  # time features encoding, options:[timeF, fixed, learned]
            'freq': 'm',       # frequency for time features encoding
        }
        
        return SimpleNamespace(**model_params)

    def set_model(self, checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_config = self.get_model_config()
        model = Model(self.model_config)
        model.load_state_dict(checkpoint)
        
        # Move model to configured device
        model = model.to(self.device)
        self.model = model

    def predict(self, bx, bxm, bym):
        """Original ML prediction method"""

        # Ensure all inputs are tensors and on the correct device
        if not isinstance(bx, torch.Tensor):
            bx = torch.FloatTensor(bx)
        if not isinstance(bxm, torch.Tensor):
            bxm = torch.FloatTensor(bxm)
        if not isinstance(bym, torch.Tensor):
            bym = torch.FloatTensor(bym)
            
        bx = bx.to(self.device)
        bxm = bxm.to(self.device)
        bym = bym.to(self.device)

        dec = torch.zeros_like(bym[:, -self.model_config.pred_len:, :])
        dec = torch.cat([bx[:, :self.model_config.label_len, -1:], dec], dim=1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(bx, bxm, dec, bym)

        return output[0]

    def predict_from_history(self, 
                           offer_points: List[OfferPoint],
                           estimated_preference: nenv.OpponentModel.EstimatedPreference,
                           target_times: List[float]) -> np.ndarray:
        """
        Predict negotiation outcomes from OfferPoint history.
        
        MainStrategy should pass offer_points that already include any candidate bids.
        
        Args:
            offer_points: List of OfferPoint objects (may include candidate bids from MainStrategy)
            estimated_preference: Estimated opponent preference for feature extraction
            target_times: List of target times for prediction
            
        Returns:
            Array of predicted acceptance probabilities (length = len(target_times))
        """

        sequence_length = self.model_config.seq_len
        
        # Process offer history into features using shared function
        history_features = process_offer_history(offer_points, estimated_preference, sequence_length)
        
        # Split the (sequence_length, 12) array directly:
        # Feature 0 = time -> T
        # Features 1-11 = other features -> X
        T = history_features[:, 0:1].reshape(1, sequence_length, 1)  # Time
        X = history_features[:, 1:12].reshape(1, sequence_length, 11)  # Other features
        
        # Prepare target times
        T_Target = np.zeros((1, self.model_config.pred_len, 1), dtype=np.float32)
        
        # Fill with actual target times, then repeat last time for model compatibility
        for i in range(self.model_config.pred_len):
            if i < len(target_times):
                T_Target[0, i, 0] = target_times[i] 
            else:
                T_Target[0, i, 0] = target_times[-1] if target_times else 0.0
        
        # Run ML prediction - predict method will handle device migration
        prediction = self.predict(X, T, T_Target).detach().cpu().numpy()
        y_pred = np.reshape(np.clip(prediction[:self.model_config.pred_len, -1], 0., 1.), (self.model_config.pred_len,))
        return y_pred
