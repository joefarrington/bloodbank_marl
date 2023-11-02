import numpy as np
from typing import Union
from ray.rllib.policy.sample_batch import SampleBatch
import gymnasium
from gymnasium.spaces.utils import unflatten
from bloodbank_marl.rllib.policies.common import DictObsSpacePolicy


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
