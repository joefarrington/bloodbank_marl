import numpy as np
from ray.rllib.policy.policy import Policy
from typing import Union
from ray.rllib.policy.sample_batch import SampleBatch
import gymnasium
from gymnasium.spaces.utils import unflatten


class DictObsSpacePolicy(Policy):
    """Policy that handles a Dict observation space, and wants to use
    specific elements in that to make computing actions simpler"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get original obs space before preproc
        orig_space = getattr(
            self.observation_space, "original_space", self.observation_space
        )
        assert (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        self.observation_space = orig_space

    def compute_actions_from_input_dict(
        self, input_dict, explore=None, timestep=None, episodes=None, **kwargs
    ):
        # Default implementation just passes obs, prev-a/r, and states on to
        # `self.compute_actions()`.
        state_batches = [s for k, s in input_dict.items() if k.startswith("state_in")]
        # NOTE: For now we're undoing the preprocessing
        return self.compute_actions(
            [unflatten(self.observation_space, x) for x in input_dict[SampleBatch.OBS]],
            state_batches,
            prev_action_batch=input_dict.get(SampleBatch.PREV_ACTIONS),
            prev_reward_batch=input_dict.get(SampleBatch.PREV_REWARDS),
            info_batch=input_dict.get(SampleBatch.INFOS),
            explore=explore,
            timestep=timestep,
            episodes=episodes,
            **kwargs,
        )

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
        # We use this custom method to compute the action for a single observation
        raise NotImplementedError

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
