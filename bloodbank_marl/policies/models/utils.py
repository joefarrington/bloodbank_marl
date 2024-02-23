import jax
import jax.numpy as jnp

# Functions for preprocessing observations

### Replenishment ###


def total_stock_on_hand_and_in_transit_by_product(obs):
    return jnp.sum(obs.stock, axis=-1) + jnp.sum(obs.in_transit, axis=-1)


def rep_obs(obs):
    return obs.rep_obs


def one_hot_day_of_week(obs):
    return obs.obs_with_one_hot_day_of_week
