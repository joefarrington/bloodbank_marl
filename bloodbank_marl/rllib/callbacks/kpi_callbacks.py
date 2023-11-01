import numpy as np
from typing import Dict, Tuple
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy


class DeMoorKpiCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # Specify the elements of info to be logger
        self._policy_ids = ["replenishment", "issuing"]
        self._log_from_info = ["holding", "wastage", "shortage", "demand", "order"]
        # Fixed for now
        self._gamma = 0.99

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Create lists to store KPIs in
        episode.user_data["custom_metrics"] = {}
        episode.user_data["custom_metrics"]["holding"] = []
        episode.user_data["custom_metrics"]["wastage"] = []
        episode.user_data["custom_metrics"]["shortage"] = []
        episode.user_data["custom_metrics"]["demand"] = []
        episode.user_data["custom_metrics"]["order"] = []

        # Use step count to work out which policy took
        # a given step
        self._cb_steps = {}

        for p in self._policy_ids:
            self._cb_steps[p] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        info = episode.last_info_for("replenishment")
        if (
            (info is not None)
            and ("step" in info.keys())
            and (info["step"] not in self._cb_steps["replenishment"])
        ):
            self._log_info_from_step("replenishment", info, episode)
            self._cb_steps["replenishment"].append(info["step"])

    def _log_info_from_step(self, policy_id, info_to_log, episode):
        if type(info_to_log) == dict:
            for k, v in info_to_log.items():
                if k in self._log_from_info:
                    # At the moment, summing the values because storing lists. Might want to revisit.
                    episode.user_data["custom_metrics"][k].append(v)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        episode.custom_metrics["service_level_pc"] = (
            (
                np.sum(episode.user_data["custom_metrics"]["demand"])
                - np.sum(episode.user_data["custom_metrics"]["shortage"])
            )
            * 100
            / np.sum(episode.user_data["custom_metrics"]["demand"])
        )
        episode.custom_metrics["wastage_pc"] = (
            (np.sum(episode.user_data["custom_metrics"]["wastage"]))
            * 100
            / (np.sum(episode.user_data["custom_metrics"]["order"]))
        )
        episode.custom_metrics["holding_units"] = np.mean(
            episode.user_data["custom_metrics"]["holding"]
        )
        episode.custom_metrics["order_q"] = np.mean(
            episode.user_data["custom_metrics"]["order"]
        )
        episode.custom_metrics["rep_return"] = np.sum(
            [
                x * self._gamma**i
                for i, x in enumerate(episode._agent_reward_history["replenishment"])
            ]
        )
