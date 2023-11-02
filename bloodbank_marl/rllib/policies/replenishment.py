import numpy as np
from ray.rllib.policy.policy import Policy
from typing import Union
from ray.rllib.policy.sample_batch import SampleBatch
import gymnasium
from gymnasium.spaces.utils import unflatten
from bloodbank_marl.rllib.policies.common import DictObsSpacePolicy


# TODO: For now this is designed for a single product
# We can tweak later to handle multiple products if that is
# easiest way to handle that
class SPolicy(DictObsSpacePolicy):
    """Order up to S"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S = self.config["policy_args"]["S"]

    def _compute_single_action(self, obs):
        # This relies on the dict obs space we're currently using
        # We don't use the action mask here, we assume an action output by
        # this policy is valid
        stock_on_hand_and_in_transit = obs["stock"].sum() + obs["in_transit"].sum()
        return np.clip(self.S - stock_on_hand_and_in_transit, 0, None)
