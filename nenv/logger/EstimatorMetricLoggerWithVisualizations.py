from nenv.logger.AbstractLogger import AbstractLogger, Bid, SessionLogs, Session, LogRow, ExcelLog
from typing import Union
import os
from nenv.Agent import AbstractAgent
from nenv.Preference import Preference
from nenv.utils.tournament_graphs import draw_line
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class EstimatorMetricLoggerWithVisualizations(AbstractLogger):
    """
        EstimatorMetricLogger logs the performance analysis of each Estimator round by round. RMSE, Spearman,
        Kendal-Tau and Pearson metrics which are commonly used for the evaluation of an Opponent Model are applied
        [Baarslag2013]_ [Keskin2023]_

        At the end of tournament, it generates overall results containing these metric results. It also draws the
        necessary plots.

        **Note**: This logger increases the computational time due to the expensive calculation of the metrics. If you
        have strict time for the tournament run, you can look *EstimatorOnlyFinalMetricLogger* which is a cheaper
        version of this logger.

        .. [Baarslag2013] Tim Baarslag, Mark J.C. Hendrikx, Koen V. Hindriks, and Catholijn M. Jonker. Predicting the performance of opponent models in automated negotiation. In International Joint Conferences on Web Intelligence (WI) and Intelligent Agent Technologies (IAT), 2013 IEEE/WIC/ACM, volume 2, pages 59–66, 2013.
        .. [Keskin2023] Mehmet Onur Keskin, Berk Buzcu, and Reyhan Aydoğan. Conflict-based negotiation strategy for human-agent negotiation. Applied Intelligence, 53(24):29741–29757, dec 2023.

    """

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        metrics = self.get_metrics(session.agentA, session.agentB, time)
        round_num = getattr(session, 'round', -1)
        preferences = self.get_preference_tracking(session.agentA, session.agentB, round_num)
        return {**metrics, **preferences}

    def on_accept(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        metrics = self.get_metrics(session.agentA, session.agentB)
        round_num = getattr(session, 'round', -1)
        preferences = self.get_preference_tracking(session.agentA, session.agentB, round_num)
        self.draw_preference_evolution(session)
        return {**metrics, **preferences}

    def on_fail(self, time: float, session: Union[Session, SessionLogs]) -> LogRow:
        metrics = self.get_metrics(session.agentA, session.agentB)
        round_num = getattr(session, 'round', -1)
        preferences = self.get_preference_tracking(session.agentA, session.agentB, round_num)
        self.draw_preference_evolution(session)
        return {**metrics, **preferences}

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        if len(estimator_names) == 0:
            return

        if not os.path.exists(self.get_path("opponent model/")):
            os.makedirs(self.get_path("opponent model/"))

        self.extract_estimator_summary(tournament_logs, estimator_names)
        rmse, kendall, spearman, pearson = self.get_estimator_results(tournament_logs, estimator_names)

        self.draw(rmse, kendall, spearman, pearson)

    def get_metrics(self, agent_a: AbstractAgent, agent_b: AbstractAgent, time: float = 0.0) -> LogRow:
        row = {}

        for estimator_id in range(len(agent_a.estimators)):
            rmseA, spearmanA, kendallA, pearsonA = agent_a.estimators[estimator_id].calculate_error(agent_b.preference)
            rmseB, spearmanB, kendallB, pearsonB = agent_b.estimators[estimator_id].calculate_error(agent_a.preference)

            log = {
                "RMSE_A": rmseA,
                "RMSE_B": rmseB,
                "SpearmanA": spearmanA,
                "SpearmanB": spearmanB,
                "KendallTauA": kendallA,
                "KendallTauB": kendallB,
                "PearsonA": pearsonA,
                "PearsonB": pearsonB,
                "RMSE": (rmseA + rmseB) / 2.,
                "Spearman": (spearmanA + spearmanB) / 2.,
                "KendallTau": (kendallA + kendallB) / 2.,
                "Pearson": (pearsonA + pearsonB) / 2.
            }

            row[agent_a.estimators[estimator_id].name] = log

        return row

    def get_preference_tracking(self, agent_a: AbstractAgent, agent_b: AbstractAgent, round_num: int) -> LogRow:
        """Track estimated preference weights every 20 rounds."""
        row = {}

        for estimator_id in range(len(agent_a.estimators)):
            estimator_name = agent_a.estimators[estimator_id].name

            # Get estimated preferences
            pref_a_estimates_b = agent_a.estimators[estimator_id].preference
            pref_b_estimates_a = agent_b.estimators[estimator_id].preference

            # Create tracking log
            tracking_log = {}

            # Track Agent A's estimation of Agent B's preferences
            for issue, weight in pref_a_estimates_b.issue_weights.items():
                tracking_log[f"A_estimates_B_{issue.name}_issue_weight"] = weight

            for issue, value_weights in pref_a_estimates_b.value_weights.items():
                for value_name, weight in value_weights.items():
                    tracking_log[f"A_estimates_B_{issue.name}_{value_name}_weight"] = weight

            # Track Agent B's estimation of Agent A's preferences
            for issue, weight in pref_b_estimates_a.issue_weights.items():
                tracking_log[f"B_estimates_A_{issue.name}_issue_weight"] = weight

            for issue, value_weights in pref_b_estimates_a.value_weights.items():
                for value_name, weight in value_weights.items():
                    tracking_log[f"B_estimates_A_{issue.name}_{value_name}_weight"] = weight

            row[f"{estimator_name}_PreferenceTracking"] = tracking_log

        return row

    def extract_estimator_summary(self, tournament_logs: ExcelLog, estimator_names: List[str]):
        summary = pd.DataFrame(
            columns=["EstimatorName", "Avg.RMSE", "Std.RMSE", "Avg.Spearman", "Std.Spearman", "Avg.KendallTau",
                     "Std.KendallTau", "Avg.Pearson", "Std.Pearson"]
        )

        for i in range(len(estimator_names)):
            results = tournament_logs.to_data_frame(estimator_names[i])

            RMSE, spearman, kendall, pearson = [], [], [], []

            RMSE.extend(results["RMSE_A"].to_list())
            RMSE.extend(results["RMSE_B"].to_list())

            spearman.extend(results["SpearmanA"].to_list())
            spearman.extend(results["SpearmanB"].to_list())

            kendall.extend(results["KendallTauA"].to_list())
            kendall.extend(results["KendallTauB"].to_list())

            pearson.extend(results["PearsonA"].to_list())
            pearson.extend(results["PearsonB"].to_list())

            summary.loc[i] = {
                "EstimatorName": estimator_names[i],
                "Avg.RMSE": np.mean(RMSE),
                "Std.RMSE": np.std(RMSE),
                "Avg.Spearman": np.mean(spearman),
                "Std.Spearman": np.std(spearman),
                "Avg.KendallTau": np.mean(kendall),
                "Std.KendallTau": np.std(kendall),
                "Avg.Pearson": np.mean(pearson),
                "Std.Pearson": np.std(pearson)
            }

        summary.sort_values(by="Avg.RMSE", inplace=True, ascending=True)

        summary.to_excel(self.get_path("opponent model/estimator_summary.xlsx"), sheet_name="EstimatorSummary")

    def get_estimator_results(self, tournament_logs: ExcelLog, estimator_names: list) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]], Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
        tournament_results = tournament_logs.to_data_frame()

        max_round = max(tournament_results["TournamentResults"]["Round"].to_list())

        rmse = {name: [[] for _ in range(max_round + 1)] for name in estimator_names}
        spearman = {name: [[] for _ in range(max_round + 1)] for name in estimator_names}
        kendall = {name: [[] for _ in range(max_round + 1)] for name in estimator_names}
        pearson = {name: [[] for _ in range(max_round + 1)] for name in estimator_names}

        for _, row in tournament_results["TournamentResults"].to_dict('index').items():
            agent_a = row["AgentA"]
            agent_b = row["AgentB"]
            domain_name = "Domain%d" % int(row["DomainName"])

            session_path = self.get_path(f"sessions/{agent_a}_{agent_b}_{domain_name}.xlsx")

            for i in range(len(estimator_names)):
                session_log = ExcelLog(file_path=session_path)

                for row_index, estimator_row in enumerate(session_log.log_rows[estimator_names[i]]):
                    if session_log.log_rows["Session"][row_index]["Action"] == "Accept":
                        break

                    _round = session_log.log_rows["Session"][row_index]["Round"]

                    rmse[estimator_names[0]][_round].append(estimator_row["RMSE_A"])
                    spearman[estimator_names[0]][_round].append(estimator_row["SpearmanA"])
                    kendall[estimator_names[0]][_round].append(estimator_row["KendallTauA"])
                    pearson[estimator_names[0]][_round].append(estimator_row["PearsonA"])
                    rmse[estimator_names[0]][_round].append(estimator_row["RMSE_B"])
                    spearman[estimator_names[0]][_round].append(estimator_row["SpearmanB"])
                    kendall[estimator_names[0]][_round].append(estimator_row["KendallTauB"])
                    pearson[estimator_names[0]][_round].append(estimator_row["PearsonB"])

        return rmse, spearman, kendall, pearson

    def draw(self, rmse: dict, spearman: dict, kendall: dict, pearson: dict):
        rmse_mean, _ = self.get_mean_std(rmse)
        spearman_mean, _ = self.get_mean_std(spearman)
        kendall_mean, _ = self.get_mean_std(kendall)
        pearson_mean, _ = self.get_mean_std(pearson)

        draw_line(rmse_mean, self.get_path("opponent model/estimator_rmse"), "Rounds", "RMSE")
        draw_line(spearman_mean, self.get_path("opponent model/estimator_spearman"), "Rounds", "Spearman")
        draw_line(kendall_mean, self.get_path("opponent model/estimator_kendall_tau"), "Rounds", "KendallTau")
        draw_line(pearson_mean, self.get_path("opponent model/estimator_pearson"), "Rounds", "Pearson")

        # After median round, these metrics may mislead since the number of session dramatically decreases.
        median_round = self.get_median_round(rmse)

        for estimator_name in rmse:
            rmse[estimator_name] = rmse[estimator_name][:median_round]
            spearman[estimator_name] = spearman[estimator_name][:median_round]
            kendall[estimator_name] = kendall[estimator_name][:median_round]
            pearson[estimator_name] = pearson[estimator_name][:median_round]

        draw_line(rmse_mean, self.get_path("opponent model/estimator_rmse_until_median_round"), "Rounds", "RMSE")
        draw_line(spearman_mean, self.get_path("opponent model/estimator_spearman_until_median_round"), "Rounds",
                  "Spearman")
        draw_line(kendall_mean, self.get_path("opponent model/estimator_kendall_tau_until_median_round"), "Rounds",
                  "KendallTau")
        draw_line(pearson_mean, self.get_path("opponent model/estimator_pearson_until_median_round"), "Rounds",
                  "Pearson")

    @staticmethod
    def get_median_round(results: dict) -> int:
        counts = []

        for estimator_name, rounds in results.items():
            for i, results in enumerate(rounds):
                for j in range(len(results)):
                    counts.append(i)

            break

        return round(float(np.median(counts)))

    @staticmethod
    def get_mean_std(results: dict) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        means, std = {}, {}

        for estimator_name, rounds in results.items():
            means[estimator_name] = []
            std[estimator_name] = []

            for result in rounds:
                means[estimator_name].append(float(np.mean(result)))
                std[estimator_name].append(float(np.std(result)))

        return means, std

    def extract_preference_tracking_data(self, session: Union[Session, SessionLogs]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[str]]:
        """
        Extract preference tracking data from session logs.

        Returns:
            - agent_a_data: Dict mapping weight names to lists of values over rounds
            - agent_b_data: Dict mapping weight names to lists of values over rounds
            - issue_names: List of issue names in the domain
        """
        session_log = session.session_log if hasattr(session, 'session_log') else session

        if len(session.agentA.estimators) == 0:
            return {}, {}, []

        estimator_name = session.agentA.estimators[0].name
        sheet_name = f"{estimator_name}_PreferenceTracking"

        # Check if tracking data exists
        if sheet_name not in session_log.log_rows or len(session_log.log_rows[sheet_name]) == 0:
            return {}, {}, []

        agent_a_data = {}
        agent_b_data = {}
        issue_names = []

        # Extract all data points
        for row in session_log.log_rows[sheet_name]:
            for key, value in row.items():
                if key.startswith("A_estimates_B_"):
                    if key not in agent_a_data:
                        agent_a_data[key] = []
                    agent_a_data[key].append(value)
                elif key.startswith("B_estimates_A_"):
                    if key not in agent_b_data:
                        agent_b_data[key] = []
                    agent_b_data[key].append(value)

        # Extract issue names from keys
        issue_set = set()
        for key in agent_a_data.keys():
            if "_issue_weight" in key:
                # Extract issue name from "A_estimates_B_IssueX_issue_weight"
                parts = key.split("_")
                issue_name = parts[3]  # IssueX
                issue_set.add(issue_name)

        issue_names = sorted(list(issue_set))

        return agent_a_data, agent_b_data, issue_names

    def draw_preference_evolution(self, session: Union[Session, SessionLogs]):
        """Generate line plots showing preference weight evolution over rounds."""
        agent_a_data, agent_b_data, issue_names = self.extract_preference_tracking_data(session)

        if not agent_a_data or not agent_b_data:
            return

        # Create directory for preference plots
        agent_a_name = session.agentA.name
        agent_b_name = session.agentB.name

        # Extract domain name from preference path
        # Path format: .../domains/{domain_name}/profileA.json
        profile_path = session.agentA.preference.profile_json_path
        domain_name = os.path.basename(os.path.dirname(profile_path))

        pref_path = self.get_path(f"opponent model/preferences/{agent_a_name}_{agent_b_name}_{domain_name}/")
        if not os.path.exists(pref_path):
            os.makedirs(pref_path)

        # Get true preferences
        true_pref_a = session.agentA.preference
        true_pref_b = session.agentB.preference

        # Draw Agent A's preference estimates (A estimates B's preferences)
        self._draw_agent_preference_plots(agent_a_data, issue_names, pref_path, "agentA_estimates_B", "A_estimates_B", true_pref_b)

        # Draw Agent B's preference estimates (B estimates A's preferences)
        self._draw_agent_preference_plots(agent_b_data, issue_names, pref_path, "agentB_estimates_A", "B_estimates_A", true_pref_a)

    def _draw_agent_preference_plots(self, data: Dict[str, List[float]], issue_names: List[str],
                                     save_path: str, file_prefix: str, data_prefix: str, true_preference: Preference):
        """Helper method to draw preference plots for one agent."""

        # 1. Draw issue weights plot
        issue_weights_data = {}
        for issue_name in issue_names:
            key = f"{data_prefix}_{issue_name}_issue_weight"
            if key in data:
                issue_weights_data[issue_name] = data[key]

        if issue_weights_data:
            plt.figure(figsize=(10, 6), dpi=150)

            # Plot estimated and true values with matching colors
            for issue_name, values in issue_weights_data.items():
                line = plt.plot(range(len(values)), values, marker='o', label=f"{issue_name} (estimated)", linewidth=2)
                color = line[0].get_color()

                # Plot true value with same color
                for issue, true_weight in true_preference.issue_weights.items():
                    if issue.name == issue_name:
                        plt.axhline(y=true_weight, linestyle='--', linewidth=2, alpha=0.7, color=color, label=f"{issue_name} (true)")
                        break

            plt.xlabel("Round (every 20)", fontsize=12)
            plt.ylabel("Weight", fontsize=12)
            plt.title(f"{file_prefix.replace('_', ' ').title()} - Issue Weights", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"{file_prefix}_issue_weights.png"))
            plt.close()

        # 2. Draw value weights plots (one per issue)
        for issue_name in issue_names:
            value_weights_data = {}

            # Find all value weight keys for this issue
            for key in data.keys():
                if key.startswith(f"{data_prefix}_{issue_name}_") and not key.endswith("_issue_weight"):
                    # Extract value name from "A_estimates_B_IssueX_ValueY_weight"
                    parts = key.split("_")
                    value_name = parts[4]  # ValueY
                    value_weights_data[value_name] = data[key]

            if value_weights_data:
                plt.figure(figsize=(10, 6), dpi=150)

                # Get true value weights for this issue
                true_value_weights = {}
                for issue, value_weights in true_preference.value_weights.items():
                    if issue.name == issue_name:
                        true_value_weights = value_weights
                        break

                # Plot estimated and true values with matching colors
                for value_name, values in value_weights_data.items():
                    line = plt.plot(range(len(values)), values, marker='o', label=f"{value_name} (estimated)", linewidth=2)
                    color = line[0].get_color()

                    # Plot true value with same color
                    if value_name in true_value_weights:
                        plt.axhline(y=true_value_weights[value_name], linestyle='--', linewidth=2, alpha=0.7, color=color, label=f"{value_name} (true)")

                plt.xlabel("Sampled Rounds", fontsize=12)
                plt.ylabel("Weight", fontsize=12)
                plt.title(f"{file_prefix.replace('_', ' ').title()} - {issue_name} Value Weights", fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"{file_prefix}_{issue_name}_value_weights.png"))
                plt.close()
