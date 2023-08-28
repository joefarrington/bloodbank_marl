from functools import partial


class FixedPolicy:
    def __init__(self, policy_function, policy_params, env_kwargs):
        self.policy_function = partial(
            policy_function, policy_params=policy_params, env_kwargs=env_kwargs
        )

    def apply(self, policy_params, obs, rng):
        return self.policy_function(obs=obs, rng=rng)
