import math
from typing import List, Union, Dict
import numpy as np
import nenv
from .NegoFormerAgent import NegoFormerAgent
from nenv import Session, SessionEstimator, Bid, AbstractAgent
from nenv.utils import LogRow, ExcelLog


class PredictionLogger(nenv.logger.AbstractLogger):
    """
        This logger evaluate the prediction performance of Autoformer model.
    """
    agent_pos: str = ""
    is_estimator_session: bool

    in_session_total_mape: List[float]
    in_session_total_mse: List[float]
    in_session_total_rmse: List[float]
    in_session_estimated_utility: Dict[float, float]
    in_session_real_utility: Dict[float, float]

    def before_session_start(self, session: Union[Session, SessionEstimator]) -> List[str]:
        """
            This method checks whether NegoFormerAgent will negotiate in this session.
        """
        if isinstance(session, SessionEstimator):
            self.is_estimator_session = True
        else:
            self.is_estimator_session = False

        self.in_session_total_mape = []
        self.in_session_total_mse = []
        self.in_session_total_rmse = []
        self.in_session_real_utility = {}
        self.in_session_estimated_utility = {}

        if session.agentA.name == "NegoFormerAgent":
            self.agent_pos = "A"

            return ["NegoFormer_Prediction"]
        elif session.agentB.name == "NegoFormerAgent":
            self.agent_pos = "B"

            return ["NegoFormer_Prediction"]
        else:
            self.agent_pos = ""

        return []

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        if self.agent_pos == "" or self.is_estimator_session:
            return {}

        agent: NegoFormerAgent = session.agentA if self.agent_pos == "A" else session.agentB

        if self.agent_pos == agent:
            return {"NegoFormer_Prediction": {"Time": "-", "PredictedUtility": "-", "EstimatedUtility": agent.opponent_model.preference.get_utility(offer)}}

        agent: NegoFormerAgent = session.agentA if self.agent_pos == "A" else session.agentB
        other: AbstractAgent = session.agentB if self.agent_pos == "A" else session.agentA

        self.in_session_real_utility[time] = other.preference.get_utility(offer)
        self.in_session_estimated_utility[time] = agent.opponent_model.preference.get_utility(offer)

        return {"NegoFormer_Prediction": {"Time": time, "PredictedUtility": "-", "EstimatedUtility": agent.opponent_model.preference.get_utility(offer)}}

    def on_session_end(self, final_row: LogRow, session: Union[Session, SessionEstimator]) -> LogRow:
        if self.agent_pos == "":
            return {"NegoFormer_Prediction": {"MSE_Real": "-", "RMSE_Real": "-", "MAPE_Real": "-", "MSE_Est": "-", "RMSE_Est": "-", "MAPE_Est": "-"}}

        if self.is_estimator_session:
            return self.on_estimator_session_end(session)

        agent: NegoFormerAgent = session.agentA if self.agent_pos == "A" else session.agentB

        real_mape, real_mse = [], []
        est_mape, est_mse = [], []
        predictions = {}

        for t, preds in agent.negoformer.predictions.items():
            if t not in self.in_session_real_utility or t not in self.in_session_estimated_utility:
                continue

            pred = np.mean(preds)
            predictions[t] = pred

            real_mse.append(math.pow(pred - self.in_session_real_utility[t], 2.))
            est_mse.append(math.pow(pred - self.in_session_estimated_utility[t], 2.))

            if self.in_session_real_utility[t] != 0.:
                real_mape.append(abs(pred - self.in_session_real_utility[t]) / self.in_session_real_utility[t])

            if self.in_session_estimated_utility[t] != 0.:
                est_mape.append(abs(pred - self.in_session_estimated_utility[t]) / self.in_session_estimated_utility[t])

        if len(real_mape) == 0 or len(est_mape) == 0:
            return {"NegoFormer_Prediction": {"MSE_Real": "-", "RMSE_Real": "-", "MAPE_Real": "-",
                                                "MSE_Est": "-", "RMSE_Est": "-", "MAPE_Est": "-"}}

        real_mse = np.mean(real_mse)
        real_mape = np.mean(real_mape)
        real_rmse = math.sqrt(real_mse)

        est_mse = np.mean(est_mse)
        est_mape = np.mean(est_mape)
        est_rmse = math.sqrt(est_mse)

        for i, row in enumerate(session.session_log.log_rows["NegoFormer_Prediction"]):
            if "Time" not in row or row["Time"] == "-" or row["Time"] not in predictions:
                continue

            if session.session_log.log_rows["Session"][i]["Who"] == self.agent_pos or session.session_log.log_rows["Session"][i]['Action'] != 'Offer':
                continue

            row["PredictedUtility"] = predictions[row["Time"]]

        session.session_log.save(session.log_path)

        return {"NegoFormer_Prediction": {"MSE_Real": real_mse, "RMSE_Real": real_rmse, "MAPE_Real": real_mape,
                                            "MSE_Est": est_mse, "RMSE_Est": est_rmse, "MAPE_Est": est_mape}}

    def on_estimator_session_end(self, session: SessionEstimator) -> LogRow:
        real_mape, real_mse = [], []
        est_mape, est_mse = [], []

        for i, row in enumerate(session.session_log.log_rows["NegoFormer_Prediction"]):
            if "PredictedUtility" not in row or row["PredictedUtility"] == "-":
                continue

            opp_pos = 'B' if self.agent_pos == 'A' else 'A'

            if session.session_log.log_rows["Session"][i]["Who"] != opp_pos or session.session_log.log_rows["Session"][i]["Action"] != 'Offer':
                continue

            pred = row["PredictedUtility"]
            real_utility = session.session_log.log_rows["Session"][i][f"Agent{opp_pos}Utility"]
            est_utility = row["EstimatedUtility"]

            real_mse.append(math.pow(pred - real_utility, 2.))
            est_mse.append(math.pow(pred - est_utility, 2.))

            if real_utility != 0.:
                real_mape.append(abs(pred - real_utility) / real_utility)

            if est_utility != 0.:
                est_mape.append(abs(pred - est_utility) / est_utility)

        if len(real_mape) == 0 or len(est_mape) == 0:
            return {"NegoFormer_Prediction": {"MSE_Real": "-", "RMSE_Real": "-", "MAPE_Real": "-",
                                                "MSE_Est": "-", "RMSE_Est": "-", "MAPE_Est": "-"}}

        real_mse = np.mean(real_mse)
        real_mape = np.mean(real_mape)
        real_rmse = math.sqrt(real_mse)

        est_mse = np.mean(est_mse)
        est_mape = np.mean(est_mape)
        est_rmse = math.sqrt(est_mse)

        return {"NegoFormer_Prediction": {"MSE_Real": real_mse, "RMSE_Real": real_rmse, "MAPE_Real": real_mape,
                                            "MSE_Est": est_mse, "RMSE_Est": est_rmse, "MAPE_Est": est_mape}}

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        total_real_mse = 0.
        total_real_rmse = 0.
        total_real_mape = 0.

        total_est_mse = 0.
        total_est_rmse = 0.
        total_est_mape = 0.

        counter = 0.

        for row in tournament_logs.log_rows["NegoFormer_Prediction"]:
            if row["MSE_Real"] == "-":
                continue

            total_real_mse += float(row["MSE_Real"])
            total_real_mape += float(row["MAPE_Real"])
            total_real_rmse += float(row["RMSE_Real"])

            total_est_mse += float(row["MSE_Est"])
            total_est_mape += float(row["MAPE_Est"])
            total_est_rmse += float(row["RMSE_Est"])

            counter += 1

        if counter == 0:
            return

        total_real_mse /= counter
        total_real_rmse /= counter
        total_real_mape /= counter

        total_est_mse /= counter
        total_est_rmse /= counter
        total_est_mape /= counter

        with open(self.get_path("NegoFormer_Prediction_logs.csv"), "w") as f:
            f.write("MSE_Real;RMSE_Real;MAPE_Real;MSE_Est;RMSE_Est;MAPE_Est;\n")

            f.write(f'{total_real_mse};{total_real_rmse};{total_real_mape};{total_est_mse};{total_est_rmse};{total_est_mape};\n')
