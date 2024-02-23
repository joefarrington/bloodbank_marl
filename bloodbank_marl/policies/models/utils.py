import jax
import jax.numpy as jnp

# Functions for preprocessing observations

### Replenishment ###


def total_stock_on_hand_and_in_transit_by_product(obs):
    return obs.obs_total_per_product


def total_stock_on_hand_and_in_transit_by_product_one_hot_day_of_week(obs):
    return obs.obs_total_per_product_with_one_hot_day_of_week


def rep_obs(obs):
    return obs.rep_obs


def one_hot_day_of_week(obs):
    return obs.obs_with_one_hot_day_of_week
