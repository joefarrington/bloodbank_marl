# Modified from https://github.com/RobertTLange/evosax/blob/main/evosax/strategy.py

import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional, Union
from functools import partial
from flax import struct
from evosax.utils import get_best_fitness_member
from evosax.core import ParameterReshaper, FitnessShaper


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    best_member: chex.Array
    best_fitness: float
    gen_counter: int


@struct.dataclass
class EvoParams:
    sigma_init: float = 0.03
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class Strategy(object):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Base Class for an Evolution Strategy."""
        self.popsize = popsize

        # Setup optional parameter reshaper
        self.use_param_reshaper = pholder_params is not None
        if self.use_param_reshaper:
            self.param_reshaper = ParameterReshaper(pholder_params, n_devices)
            self.num_dims = self.param_reshaper.total_params
        else:
            self.num_dims = num_dims
        assert (
            self.num_dims is not None
        ), "Provide either num_dims or pholder_params to strategy."

        # Mean exponential decay coefficient m' = coeff * m
        # Implements form of weight decay regularization
        self.mean_decay = mean_decay
        self.use_mean_decay = mean_decay > 0.0

        # Setup optional fitness shaper
        self.fitness_shaper = FitnessShaper(**fitness_kwargs)

    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        params = self.params_strategy
        return params

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self,
        rng: chex.PRNGKey,
        params: Optional[EvoParams] = None,
        init_mean: Optional[Union[chex.Array, chex.ArrayTree]] = None,
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Initialize strategy based on strategy-specific initialize method
        if init_mean is not None:
            if self.use_param_reshaper:
                init_mean = self.param_reshaper.flatten_single(init_mean)
            else:
                init_mean = jnp.asarray(init_mean)

        state = self.initialize_strategy(rng, params, init_mean)

        if init_mean is not None:
            state = state.replace(mean=init_mean)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> Tuple[Union[chex.Array, chex.ArrayTree], EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Generate proposal based on strategy-specific ask method
        x, state = self.ask_strategy(rng, state, params)
        # Clip proposal candidates into allowed range
        x_clipped = jnp.clip(x, params.clip_min, params.clip_max)

        # Reshape parameters into pytrees
        if self.use_param_reshaper:
            x_out = self.param_reshaper.reshape(x_clipped)
        else:
            x_out = x_clipped
        return x_out, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: Union[chex.Array, chex.ArrayTree],
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Flatten params if using param reshaper for ES update
        if self.use_param_reshaper:
            x = self.param_reshaper.flatten(x)

        # Perform fitness reshaping inside of strategy tell call (if desired)
        fitness_re = self.fitness_shaper.apply(x, fitness)

        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, fitness_re, state, params)

        # Check if there is a new best member & update trackers
        best_member, best_fitness = get_best_fitness_member(
            x, fitness, state, self.fitness_shaper.maximize
        )

        # Exponentially decay mean if coefficient > 0.0
        if self.use_mean_decay:
            state = state.replace(mean=(1 - self.mean_decay) * state.mean)

        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )

    def initialize_strategy(
        self,
        rng: chex.PRNGKey,
        params: EvoParams,
        init_mean: Optional[Union[chex.Array, chex.ArrayTree]] = None,
    ) -> EvoState:
        """Search-specific `initialize` method. Returns initial state."""
        raise NotImplementedError

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """Search-specific `ask` request. Returns proposals & updated state."""
        raise NotImplementedError

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """Search-specific `tell` update. Returns updated state."""
        raise NotImplementedError

    def get_eval_params(self, state: EvoState):
        """Return reshaped parameters to evaluate."""
        if self.use_param_reshaper:
            x_out = self.param_reshaper.reshape_single(state.mean)
        else:
            x_out = state.mean
        return x_out
