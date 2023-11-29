from bloodbank_marl.scenarios.meneses_perishable.jax_env import (
    MenesesPerishableEnv,
    EnvObs,
)
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import numpy as np
import chex
from flax import struct
from bloodbank_marl.utils.gymnax_fitness import make


class DiscreteActorCritic(nn.Module):
    n_actions: int
    n_hidden: int = 64
    action_pad: int = 0
    activation: str = "tanh"
    preprocess_observation: callable = lambda obs: obs.obs

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = self.preprocess_observation(obs)  # Flat representation
        actor_mean = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        # Apply action masking to logits
        actor_mean = jnp.hstack(
            [actor_mean, jnp.zeros((actor_mean.shape[:-1] + (self.action_pad,)))]
        )
        actor_mean = actor_mean + jnp.where(obs.action_mask == 0, -1e8, 0.0)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ContinuousActorCritic(nn.Module):
    n_actions: int
    n_hidden: int = 64
    action_pad: int = 0  # Not currently used
    activation: str = "tanh"
    preprocess_observation: callable = lambda obs: obs.obs

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = self.preprocess_observation(obs)  # Flat representation
        actor_mean = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.n_actions,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# # TODO MAybe rename this multiproduct?
class DiscreteIssuingActorCritic(nn.Module):
    n_actions: int
    n_hidden: int = 64
    action_pad: int = 0
    activation: str = "tanh"
    preprocess_observation: callable = lambda obs: obs.obs

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = self.preprocess_observation(obs)  # Flat representation
        actor_mean = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.n_actions + 1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(
            actor_mean
        )  # Add 1 to actions, 0 in the categorical dist is issuing nothing
        # Apply action masking to logits
        ones_shape = obs.action_mask.shape[:-1] + (1,)
        ones = jnp.ones(ones_shape)
        actor_mean = actor_mean + jnp.hstack(
            [ones, jnp.where(obs.action_mask == 0, -1e8, 0.0)]
        )  # Stack the action mask with 1, can always issue nothing
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.n_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
