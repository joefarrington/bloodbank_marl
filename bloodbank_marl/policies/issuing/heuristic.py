from bloodbank_marl.policies.common import HeuristicPolicy
import jax
import jax.numpy as jnp
import flax.linen as nn

# NOTE: Unlike for replenishment, we don't fit the parameters for the heuristic
# issuing policies using SimOpt so we don't need any methods for naming parameters etc


class OufoIssuingPolicy(HeuristicPolicy):
    # Policy paramters are not required
    def _apply(self, policy_params, obs, rng):
        # Apply should get you an action
        tr_action = jax.lax.cond(
            jnp.sum(obs.stock) == 0,
            lambda _: jnp.array(0),
            lambda _: self.env_kwargs["max_useful_life"]
            - jnp.flip(jnp.where(obs.stock > 0, 1, 0)).argmax(),
            None,
        )
        return tr_action

    def _postprocess_action(self, obs, tr_action):
        return tr_action.astype(jnp.int32)


class ExactMatchIssuingPolicy(HeuristicPolicy):
    # Policy parameters are not required
    def _apply(self, policy_params, obs, rng):
        # Apply should get you an action
        total_stock_by_product = obs.stock.sum(axis=-1)
        tr_action = jnp.zeros_like(total_stock_by_product, dtype=jnp.float32)
        tr_action = jax.lax.select(
            total_stock_by_product[obs.request_type] > 0,
            tr_action.at[obs.request_type].set(1.0),
            tr_action,
        )
        return tr_action

    def _postprocess_action(self, obs, tr_action):
        return tr_action.astype(jnp.int32)


class PriorityMatchIssuingPolicy(HeuristicPolicy):
    # Policy parameters are the priorities for each request type
    def _apply(self, policy_params, obs, rng):
        # Apply should get you an action
        total_stock_by_product = obs.stock.sum(axis=-1)
        tr_action = jnp.zeros_like(total_stock_by_product, dtype=jnp.float32)
        rt = obs.request_type
        priorities = jax.lax.dynamic_index_in_dim(policy_params, rt, 0)
        in_stock_and_compatible = jnp.where(
            total_stock_by_product[priorities] > 0, 1, 0
        ) * jnp.where(priorities >= 0, 1, 0)
        tr_action = jax.lax.select(
            jnp.any(in_stock_and_compatible),
            tr_action.at[priorities[..., in_stock_and_compatible.argmax()]].set(1.0),
            tr_action,
        )
        return tr_action

    def _postprocess_action(self, obs, tr_action):
        return tr_action.astype(jnp.int32)


class OldestCompatibleIssuingPolicy(HeuristicPolicy):
    # Issue the oldest compatible unit; if there are multiple compatible units with the same age, issue the one with the highest priority
    # Policy parameters are the priorities for each request type
    def _apply(self, policy_params, obs, rng):
        # 0 if there are units about the expire, 1 if units has 1 day of remaining useful life etc; inf if no units in stock
        age_priority = jnp.where(
            obs.stock.sum(axis=-1) > 0,
            jnp.flip(obs.stock, axis=-1).argmax(axis=-1),
            jnp.inf,
        )
        # Compatibility priority for the requested type
        compatibility_priority = jax.lax.dynamic_index_in_dim(
            policy_params, obs.request_type, 0
        )
        # Array of the age prioritity in order of compatibility priority, so first element is age priority of most compatible unit
        age_priority_in_compatibility_order = jnp.where(
            compatibility_priority >= 0, age_priority[compatibility_priority], jnp.inf
        )
        min_compatible_age_priority = jnp.min(age_priority_in_compatibility_order)

        tr_action = jnp.zeros_like(obs.stock.sum(axis=-1), dtype=jnp.float32)
        # If there are compatible units in stock, select the most compatible unit with the highest age priority
        tr_action = jax.lax.select(
            min_compatible_age_priority < jnp.inf,
            tr_action.at[
                compatibility_priority[
                    0,
                    (
                        age_priority_in_compatibility_order
                        == min_compatible_age_priority
                    ).argmax(),
                ]
            ].set(1.0),
            tr_action,
        )
        return tr_action

    def _postprocess_action(self, obs, tr_action):
        return tr_action.astype(jnp.int32)
