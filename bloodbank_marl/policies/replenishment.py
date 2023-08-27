import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Optional


class RepMLP(nn.Module):
    n_actions: int  # For now, number of actions is the max_order_quantity + 1

    @nn.compact
    def __call__(self, x, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        s = jnp.argmax(x, axis=-1)
        return s
