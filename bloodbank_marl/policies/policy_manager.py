import jax


class PolicyManager:
    def __init__(self, policies: list):
        self.policies = policies

    def apply(self, policy_params, obs, rng):
        return jax.lax.switch(obs.agent_id, self.policies, policy_params, obs.obs, rng)
