import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Optional
from bloodbank_marl.utils.gymnax_fitness import make

# TODO: Should probably unify actions for DeMoor and Meneses environments
# At the moment, the DemOOr issue action 0 is issue nothing,
# Whereas in Meneses, all zeros means 0.
# NOTE: Actually, I think we'll have to keep them separate.


class FlaxIssuePolicy:
    def __init__(
        self,
        model_class,
        model_kwargs,
        policy_id,
        env_name,
        env_kwargs={},
        env_params={},
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.replace(**env_params)
        self.model = model_class(
            n_actions=env.action_space(self.env_params, policy_id).n, **model_kwargs
        )

    def get_params(self, rng):
        env, _ = make(self.env_name, **self.env_kwargs)
        rng, _rng = jax.random.split(rng)
        obs, _ = env.reset(_rng, self.env_params)
        return self.model.init(rng, obs)

    def apply(self, policy_params, obs, rng):
        return self.model.apply(policy_params[self.policy_id], obs, rng)


class IssueDiscreteMLP(nn.Module):
    n_hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.stock
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        x = jnp.argmax(x, axis=-1)
        return x


class IssueMultiProductMLP(nn.Module):
    # NOTE: Here we expect to be selecting between products, and then we'll automatically
    # use an OUFO policy to choose between units of that product.

    # NOTE: This is designed for the Meneses environment, and for use with NeuroEvo.
    # Action spaces need to be the same size, so this has one action dim per product type
    # And will be all zeros if no products are issued.

    n_hidden: int
    n_actions: int  # This will be the number of products

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.stock
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        x = jnp.argmax(x, axis=-1)
        a = jnp.zeros_like(x)
        a = a.at[x].set(1)
        a = (
            a * ob.action_mask
        )  # Catch the case where no stock and so first element is argmax by default
        return a


def issue_fifo(policy_params, obs, rng, env_kwargs):
    """Action is idx of oldest unit, plus one.
    Zero if no units are present"""
    obs = obs.obs
    stock = obs[
        env_kwargs["lead_time"]
        - 1 : env_kwargs["max_useful_life"]
        + env_kwargs["lead_time"]
        - 1
    ]
    return jax.lax.cond(
        jnp.sum(stock) == 0,
        lambda _: jnp.array(0),
        lambda _: env_kwargs["max_useful_life"]
        - jnp.flip(jnp.where(stock > 0, 1, 0)).argmax(),
        None,
    )


def issue_lifo(policy_params, obs, rng, env_kwargs):
    """Action is idx of youngest unit, plus one.
    Zero if no units are present"""
    obs = obs.obs
    stock = obs[
        env_kwargs["lead_time"]
        - 1 : env_kwargs["max_useful_life"]
        + env_kwargs["lead_time"]
        - 1
    ]
    return jax.lax.cond(
        jnp.sum(stock) == 0,
        lambda _: jnp.array(0),
        lambda _: 1 + (jnp.where(stock > 0, 1, 0)).argmax(),
        None,
    )


# Policies for the Meneses Perishable Env
# These are designed to be used with the FixedPolicy class, we do not
# envisage optimizing the parameters.

# In these policies, the action has the same dimensions as the number of products
# If the action vector is all 0s, then no products are issued


def issue_exact_match(policy_params, obs, rng, env_kwargs):
    """Issue the requested type if available, otherwise nothing.
    Use OUFO for units of the matching type.
    policy_params is not used for this policy"""
    total_stock_by_product = obs.stock.sum(axis=-1)
    action = jnp.zeros_like(total_stock_by_product)
    action = jax.lax.select(
        total_stock_by_product[obs.request_type] > 0,
        action.at[obs.request_type].set(1),
        action,
    )
    return action


def issue_priority_match(policy_params, obs, rng, env_kwargs):
    """Issue the highest priority available unit, or nothing if no compatible units are available.
    For best available matching type, use OUFO.
    policy_params is an (n_products, n_products) matrix of priorities.
    Each row indicates a product type. Within a row, idx of the best match is in first col, next best in second col, etc.
    Use -1 to pad where only some types are compatible
    """
    total_stock_by_product = obs.stock.sum(axis=-1)
    action = jnp.zeros_like(total_stock_by_product)
    rt = obs.request_type
    in_stock_and_compatible = jnp.where(
        total_stock_by_product[policy_params[rt]] > 0, 1, 0
    ) * jnp.where(policy_params[rt] >= 0, 1, 0)
    action = jax.lax.select(
        jnp.any(in_stock_and_compatible),
        action.at[policy_params[rt][in_stock_and_compatible.argmax()]].set(1),
        action,
    )
    return action
