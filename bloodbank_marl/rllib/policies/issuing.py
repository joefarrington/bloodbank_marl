import numpy as np
from ray.rllib.policy.policy import Policy


# TODO: For now this is designed for a single product
# We can tweak later to handle multiple products if that is
# easiest way to handle that
class OufoPolicy(Policy):
    """Issue oldest unit first"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        if np.sum(obs["stock"]) == 0:
            return 0
        else:
            return len(obs["stock"]) - (obs["stock"] > 0)[::-1].argmax()

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    # TODO: For now this is designed for a single product


# We can tweak later to handle multiple products if that is
# easiest way to handle that
class YufoPolicy(Policy):
    """Issue oldest unit first"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        if np.sum(obs["stock"]) == 0:
            return 0
        else:
            return (
                obs["stock"] > 0
            ).argmax() + 1  # Add one because action 0 is issue nothing

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
