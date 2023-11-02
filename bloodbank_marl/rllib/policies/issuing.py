import numpy as np
from typing import Union
from ray.rllib.policy.sample_batch import SampleBatch
import gymnasium
from gymnasium.spaces.utils import unflatten
from bloodbank_marl.rllib.policies.common import DictObsSpacePolicy

# TODO: Priority match policy for Meneses


# TODO: For now this is designed for a single product
# We can tweak later to handle multiple products if that is
# easiest way to handle that
class OufoPolicy(DictObsSpacePolicy):
    """Issue oldest unit first"""

    def _compute_single_action(self, obs):
        # This relies on the dict obs space we're currently using
        # We don't use the action mask here, we assume an action output by
        # this policy is valid
        if np.sum(obs["stock"]) == 0:
            return 0
        else:
            return len(obs["stock"]) - (obs["stock"] > 0)[::-1].argmax()


# We can tweak later to handle multiple products if that is
# easiest way to handle that
class YufoPolicy(DictObsSpacePolicy):
    """Issue oldest unit first"""

    def _compute_single_action(self, obs):
        # This relies on the dict obs space we're currently using
        # We don't use the action mask here, we assume an action output by
        # this policy is valid

        if np.sum(obs["stock"]) == 0:
            return 0
        else:
            return (
                obs["stock"] > 0
            ).argmax() + 1  # Add one because action 0 is issue nothing


class ExactMatchPolicy(DictObsSpacePolicy):
    """For multiple product situation: issue exact match if possible"""

    def _compute_single_action(self, obs):
        # This relies on the dict obs space we're currently using
        # We don't use the action mask here, we assume an action output by
        # this policy is valid
        total_stock_by_product = obs["stock"].sum(axis=1)
        if total_stock_by_product[obs["request_type"]] > 0:
            return obs["request_type"] + 1  # Because 0 is issuing nothing
        else:
            return 0


class PriorityMatchPolicy(DictObsSpacePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priorities = self.config["policy_args"]["priorities"]

    def _compute_single_action(self, obs):
        total_stock_by_product = obs["stock"].sum(axis=-1)
        rt = obs["request_type"]
        in_stock_and_compatible = np.where(
            total_stock_by_product[self.priorities[rt]] > 0, 1, 0
        ) * np.where(self.priorities[rt] >= 0, 1, 0)
        if np.any(in_stock_and_compatible):
            return self.priorities[rt][in_stock_and_compatible.argmax()] + 1
        else:
            return 0
