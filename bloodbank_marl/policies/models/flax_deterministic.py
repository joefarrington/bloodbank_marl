import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Union, Dict, Optional, Any, List
from bloodbank_marl.utils.gymnax_fitness import make
from bloodbank_marl.utils.yaml import from_yaml, to_yaml
from bloodbank_marl.environments.environment import MarlEnvironment
from bloodbank_marl.scenarios.meneses_perishable.jax_env import (
    MenesesPerishableEnv,
    EnvObs,
    EnvParams,
    EnvInfo,
    EnvState,
)
import numpy as np
import pandas as pd


class RepDiscreteMLP(nn.Module):
    n_hidden: Union[int, list]
    n_actions: int
    action_pad: int = 0
    preprocess_observation: callable = lambda obs: obs.obs

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = self.preprocess_observation(obs)
        # Handle single or multiple hidden layers
        n_hidden = [self.n_hidden] if isinstance(self.n_hidden, int) else self.n_hidden
        for h in n_hidden:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        # x = nn.Dense(self.n_hidden)(x)
        # x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = jnp.hstack([x, jnp.zeros((x.shape[:-1] + (self.action_pad,)))])
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        x = jnp.argmax(x, axis=-1)
        return x


class RepMultiProductMLP(nn.Module):
    n_hidden: Union[int, list]
    n_actions: int
    action_pad: int = 0
    preprocess_observation: callable = lambda obs: obs.obs

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        # x = obs.stock.sum(axis=-1) # Alternative is just have total number of stock per product
        x = self.preprocess_observation(obs)
        # x = nn.Dense(self.n_hidden)(x)
        # x = nn.relu(x)
        n_hidden = [self.n_hidden] if isinstance(self.n_hidden, int) else self.n_hidden
        for h in n_hidden:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        # x = nn.tanh(x)
        return x


class IssueDiscreteMLP(nn.Module):
    n_hidden: Union[int, list]
    n_actions: int
    action_pad: int = 0
    preprocess_observation: callable = lambda obs: obs.stock

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = self.preprocess_observation(obs)
        n_hidden = [self.n_hidden] if isinstance(self.n_hidden, int) else self.n_hidden
        for h in n_hidden:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = jnp.hstack([x, jnp.zeros((x.shape[:-1] + (self.action_pad,)))])
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        x = jnp.argmax(x, axis=-1)
        return x


class IssueMultiProductMLP(nn.Module):
    # NOTE: Here we expect to be selecting between products, and then we'll automatically
    # use an OUFO policy to choose between units of that product.

    # NOTE: This is designed for the Meneses environment, and for use with NeuroEvo.
    # Action spaces need to be the same size, so this has one action dim per product type
    # And will be all zeros if no products are issued.

    n_hidden: Union[int, list]
    n_actions: int  # This will be the number of products
    action_pad: int = 0
    preprocess_observation: callable = lambda obs: obs.obs

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = self.preprocess_observation(obs)
        n_hidden = [self.n_hidden] if isinstance(self.n_hidden, int) else self.n_hidden
        for h in n_hidden:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        a = jnp.where(x == x.max(), 1, 0)
        a = (
            a * obs.action_mask
        )  # Catch the case where no stock and so first element is argmax by default
        return a
