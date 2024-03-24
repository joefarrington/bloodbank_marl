import jax
import jax.numpy as jnp

# Functions for preprocessing observations

### Replenishment ###


def obs_total_per_product(obs):
    return obs.obs_total_per_product


def obs_total_per_product_and_weekday(obs):
    return obs.obs_total_per_product_and_weekday


def obs_basic(obs):
    return obs.obs


def rep_obs(obs):
    return obs.rep_obs


def rep_obs_with_one_hot_day_of_week(obs):
    return obs.rep_obs_with_one_hot_day_of_week


def one_hot_day_of_week(obs):
    return obs.obs_with_one_hot_day_of_week
