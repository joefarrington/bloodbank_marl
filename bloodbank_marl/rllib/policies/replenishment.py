import numpy as np
from ray.rllib.policy.policy import Policy


# TODO: For now this is designed for a single product
# We can tweak later to handle multiple products if that is
# easiest way to handle that
class SPolicy(Policy):
    """Order up to S"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S = self.config["policy_args"]["S"]

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        return [self._compute_single_action(x) for x in obs_batch], [], {}

    def _compute_single_action(self, obs):
        # This relies on the dict obs space we're currently using
        # We don't use the action mask here, we assume an action output by
        # this policy is valid
        stock_on_hand_and_in_transit = obs["stock"].sum() + obs["in_transit"].sum()
        return np.clip(self.S - stock_on_hand_and_in_transit, 0, None)

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
