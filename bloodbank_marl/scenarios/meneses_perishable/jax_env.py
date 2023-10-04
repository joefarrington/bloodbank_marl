import jax
import chex
from typing import Tuple, Union, Optional, Dict
from flax import struct
import jax.numpy as jnp
from gymnax.environments import spaces
import numpy as np
from bloodbank_marl.environments.environment import MarlEnvironment
import distrax

jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32

# TODO: Should the actions have an explicit action for issuing nothing?
# If so, we need an extra action in the action spact, and so we'd probably need to
# pad the replenishment action by one to make them the same shape

# TODO: Action masking

n_products = 8


# TODO: Recheck all defaults
@struct.dataclass
class EnvParams:
    poisson_demand_mean: float = 4
    product_probabilities: chex.Array = jnp.array(
        [
            0.08614457,
            0.36024097,
            0.08192771,
            0.36485943,
            0.0126506,
            0.0684739,
            0.00522088,
            0.02048193,
        ]
    )
    age_on_arrival_distribution_probs: chex.Array = jnp.array([1] + [0] * (35 - 1))
    fixed_order_costs: float = 160
    variable_order_costs: chex.Array = jnp.array([0] * n_products)
    shortage_costs: chex.Array = jnp.array([1340] * n_products)
    wastage_costs: chex.Array = jnp.array([130] * n_products)
    holding_costs: chex.Array = jnp.array([1.1] * n_products)
    substitution_costs: chex.Array = jnp.ones((n_products, n_products)) - jnp.eye(
        n_products
    )
    max_days_in_episode: int = 365
    max_steps_in_episode: int = 1e10
    gamma: float = 1.0


@struct.dataclass
class EnvInfo:
    demand: chex.Array
    shortages: chex.Array
    expiries: chex.Array
    holding: chex.Array
    allocations: chex.Array
    orders: chex.Array
    order_placed: chex.Array
    day_counter: chex.Array

    @classmethod
    def create_empty_infos(cls, n_agents: int, n_products: int, max_useful_life: int):
        return cls(
            demand=jnp.zeros((n_agents, n_products), dtype=jnp_int),
            shortages=jnp.zeros((n_agents, n_products), dtype=jnp_int),
            expiries=jnp.zeros((n_agents, n_products), dtype=jnp_int),
            holding=jnp.zeros((n_agents, n_products), dtype=jnp_int),
            allocations=jnp.zeros((n_agents, n_products, n_products), dtype=jnp_int),
            orders=jnp.zeros((n_agents, n_products), dtype=jnp_int),
            order_placed=jnp.zeros((n_agents,), dtype=jnp_int),
            day_counter=jnp.zeros((n_agents,), dtype=jnp_int),
        )

    def reset_infos_one_agent(self, agent_id: int):
        return self.replace(
            demand=self.demand.at[agent_id].set(0),
            shortages=self.shortages.at[agent_id].set(0),
            expiries=self.expiries.at[agent_id].set(0),
            holding=self.holding.at[agent_id].set(0),
            allocations=self.allocations.at[agent_id].set(0),
            orders=self.orders.at[agent_id].set(0),
            order_placed=self.order_placed.at[agent_id].set(0),
            day_counter=self.day_counter.at[agent_id].set(0),
        )

    def update_infos_one_agent(self, agent_id: int, info: EnvInfo):
        return self.replace(
            demand=self.demand.at[agent_id].add(info.demand),
            shortages=self.shortages.at[agent_id].add(info.shortages),
            expiries=self.expiries.at[agent_id].add(info.expiries),
            holding=self.holding.at[agent_id].add(info.holding),
            allocations=self.allocations.at[agent_id].add(info.allocations),
            orders=self.orders.at[agent_id].add(info.orders),
            order_placed=self.order_placed.at[agent_id].add(info.order_placed),
            day_counter=self.day_counter.at[agent_id].add(info.day_counter),
        )


@struct.dataclass
class EnvState:
    stock: chex.Array
    in_transit: chex.Array
    request_time: float  # time of next request
    request_type: int  # type of next request
    agent_id: int  # Agent to act
    cumulative_rewards: chex.Array
    infos: EnvInfo
    truncations: chex.Array
    terminations: chex.Array
    live_agents: chex.Array
    day: int  # Measures days
    time: float  # Measures time
    step: int  # Measure total steps, rep and alloc combined


@struct.dataclass
class EnvObs:
    agent_id: int  # Following Tianhou;
    time: float
    request_type: int
    in_transit: chex.Array
    stock: chex.Array
    # mask: chex.Array # TODO: Action masking

    @property
    def obs(self):
        return jnp.hstack(
            [
                self.time,
                self.request_type,
                self.in_transit.reshape(-1),
                self.stock.reshape(-1),
            ]
        )


class MenesesPerishableEnv(MarlEnvironment):
    def __init__(
        self,
        agent_names=["replenishment", "issuing"],
        n_products: int = 8,
        max_useful_life: int = 35,
        lead_time: int = 1,
        max_order_quantities: list = [100] * 8,
    ):
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array(max_order_quantities)

        self.possible_agents = agent_names
        self.agent_ids = {agent_name: i for i, agent_name in enumerate(agent_names)}
        self.n_agents = len(agent_names)

    # TODO: We can remove unless we want to add a function
    # because we're customizing them etc
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def live_step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[EnvObs, EnvState, float, bool, bool, EnvInfo]:
        pol_key, age_key = jax.random.split(key)

        # TODO: Difference between truncation and termination
        state = jax.lax.switch(
            state.agent_id,
            [self._replenishment_step, self._issuing_step],
            pol_key,
            state,
            action,
            params,
        )

        # Select the next agent
        # If no remainine demand, next agent is replenishment (agent_id 0)
        next_agent_id = jax.lax.cond(
            state.request_time >= state.day + 1, lambda: 0, lambda: 1
        )

        # If the next agent is replenishent, we need to age the stock, work out the holding cost, update the day (and the time to start of next day)
        # Otherwise, just update the time to the next request time
        state = jax.lax.cond(
            next_agent_id == 0,
            self._age_stock,
            lambda age_key, state, params: state.replace(time=state.request_time),
            age_key,
            state,
            params,
        )

        # Check for termination - for now we don't make a distinction between truncation and termination
        # And either both or noen of the agents are done
        # TODO: Separate termination for each agent? Not necessary for what we're doing.
        trunc = jax.lax.cond(
            jax.lax.bitwise_or(
                state.day
                >= params.max_days_in_episode,  # Day is incremented by self._age_stock
                state.step + 1 >= params.max_steps_in_episode,
            ),
            lambda _: True,
            lambda _: False,
            None,
        )
        truncations = state.truncations.at[:].set(trunc)
        live = jax.lax.cond(truncations[state.agent_id], lambda: 0, lambda: 1)
        live_agents = state.live_agents.at[state.agent_id].set(live)

        state = state.replace(
            truncations=truncations,
            live_agents=live_agents,
            step=state.step + 1,
            agent_id=next_agent_id,
        )

        return (
            self.get_obs(state, next_agent_id),
            state,
            state.cumulative_rewards[next_agent_id],
            state.truncations[next_agent_id],
            state.terminations[next_agent_id],
            self.get_info(state, next_agent_id),
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[EnvObs, EnvState]:
        """Environment-specific reset."""
        state = EnvState(
            stock=jnp.zeros((self.n_products, self.max_useful_life), dtype=jnp_int),
            in_transit=jnp.zeros((self.n_products, self.lead_time), dtype=jnp.int32),
            request_time=0,
            request_type=0,
            agent_id=0,
            cumulative_rewards=jnp.zeros((self.n_agents,)),
            infos=EnvInfo.create_empty_infos(
                self.n_agents, self.n_products, self.max_useful_life
            ),
            truncations=jnp.array([False] * self.n_agents),
            terminations=jnp.array([False] * self.n_agents),
            live_agents=jnp.array([1] * self.n_agents),
            day=0,
            time=0.0,
            step=0,
        )
        # Sample the first request and update the state with it
        # In some cases (e.g. if we have a weekday) demand would depend on state so need to create
        # a state first, then sample, then update state based on result of sample
        request_interval, request_type = self._sample_next_request(key, state, params)
        state = state.replace(
            request_time=state.time + request_interval, request_type=request_type
        )

        return self.get_obs(state, state.agent_id), state

    def get_obs(self, state: EnvState, agent_id: int) -> EnvObs:
        """Applies observation function to state, in PettinZoo AECEnv the equivalent is .observe()"""
        # TODO: For now, each agent gets the same observation and we'll deal with it at the agent level

        request_type = jax.lax.select(
            agent_id == 0, 0, state.request_type
        )  # If a rep action, pad this space with zero so both observations same shape
        return EnvObs(
            state.agent_id,
            state.time,
            state.request_type,
            state.in_transit[: self.n_products, 1 : self.lead_time],
            state.stock,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        return jax.lax.cond(
            jax.lax.bitwise_or(
                state.day >= params.max_days_in_episode,
                state.step >= params.max_steps_in_episode,
            ),
            lambda _: True,
            lambda _: False,
            None,
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "MensesesPerishable"

    @property
    def num_actions(self, agent_id: int) -> int:
        """Number of actions possible in environment for agent with id `agent_id`."""
        # TODO Add in number of actions
        raise NotImplementedError

    def action_space(self, params: EnvParams, agent_id: int):
        """Action space of the agent with id `agent_id`"""
        rep_space = spaces.Box(
            low=0, high=params.max_order_quantities + 1, shape=(self.n_products,)
        )
        issue_space = spaces.Box(low=0, high=1, shape=(self.n_products,))
        return jax.lax.switch(agent_id, [lambda: rep_space, lambda: issue_space])

    def observation_space(self, params: EnvParams, agent_id: int = 1):
        """Observation space of the agent with id `agent_id`. For now, both the same"""
        return spaces.Box(
            low=0,
            high=jnp.hstack(
                [
                    1e10,
                    self.n_products,
                    jnp.repeat(params.max_order_quantities, self.lead_time - 1),
                    jnp.repeat(params.max_order_quantities, self.max_useful_life),
                ]
            ),
            shape=(
                self.lead_time - 1 + self.max_useful_life + 2
            ),  # Additional 2 elements are the time, and the type of the request
            dtype=jnp.float32,
        )

    # TODO: Double check. And think whether it might be better to have like a training-state thing
    # that includes both state, info etc.
    def state_space(self, params: EnvParams, agent_id: int):
        """State space of the environment."""
        return spaces.Dict(
            {
                "stock": spaces.Box(
                    low=0,
                    # TODO: This is fine when all arrive fresh, but should probably be max storage etc otherwise
                    high=jnp.array(
                        [self.max_order_quantities] * self.max_useful_life
                    ).transpose(),
                    shape=(self.n_products, self.max_useful_life),
                    dtype=jnp.int32,
                ),
                "in_transit": spaces.Box(
                    low=0,
                    high=jnp.array(
                        [self.max_order_quantities] * self.lead_time
                    ).transpose(),
                    shape=(self.n_products, self.lead_time),
                    dtype=jnp.int32,
                ),
                "request_time": spaces.Box(
                    low=0, high=1e10, shape=(1,), dtype=jnp.float32
                ),
                "request_type": spaces.Discete(self.n_products),
                "agent_id": spaces.Discete(self.n_agents),
                "cumulative_rewards": spaces.Box(
                    low=0, high=1e10, shape=(self.n_agents,), dtype=jnp.int32
                ),
                "infos": spaces.Dict(
                    {
                        "demand": spaces.Box(
                            low=0,
                            high=1e10,
                            shape=(self.n_agents, self.n_products),
                            dtype=jnp.int32,
                        ),
                        "shortages": spaces.Box(
                            low=0,
                            high=1e10,
                            shape=(self.n_agents, self.n_products),
                            dtype=jnp.int32,
                        ),
                        "expiries": spaces.Box(
                            low=0,
                            high=1e10,
                            shape=(self.n_agents, self.n_products),
                            dtype=jnp.int32,
                        ),
                        "holding": spaces.Box(
                            low=0,
                            high=1e10,
                            shape=(self.n_agents, self.n_products),
                            dtype=jnp.int32,
                        ),
                        "allocations": spaces.Box(
                            low=0,
                            high=1e10,
                            shape=(self.n_agents, self.n_products, self.n_products),
                            dtype=jnp.int32,
                        ),
                        "orders": spaces.Box(
                            low=0,
                            high=1e10,
                            shape=(self.n_agents, self.n_products),
                            dtype=jnp.int32,
                        ),
                        "order_placed": spaces.Box(
                            low=0, high=1e10, shape=(self.n_agents,), dtype=jnp.int32
                        ),  # Expect to be binary as expect more than one request per day
                    }
                ),
                "truncations": spaces.Box(
                    low=0, high=1, shape=(self.n_agents,), dtype=jnp.bool
                ),  # TODO: is this okay?
                "terminations": spaces.Box(
                    low=0, high=1, shape=(self.n_agents,), dtype=jnp.bool
                ),  # TODO: is this okay?
                "live_agents": spaces.Box(
                    low=0, high=1, shape=(self.n_agents,), dtype=jnp.bool
                ),  # TODO: is this okay?
                "day": spaces.Box(low=0, high=1e10, shape=(1,), dtype=jnp.int32),
                "time": spaces.Box(low=0, high=1e10, shape=(1,), dtype=jnp.float32),
                "step": spaces.Box(low=0, high=1e10, shape=(1,), dtype=jnp.int32),
            }
        )

    def _age_stock(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ) -> Tuple[EnvState, chex.Array]:
        """Ages stock by one time period and calculates expiry and holding cost."""
        stock, in_transit, infos, cumulative_rewards = (
            state.stock,
            state.in_transit,
            state.infos,
            state.cumulative_rewards,
        )
        # Age the stock by one day and calculate wastage cost
        expired = stock[: self.n_products, self.max_useful_life - 1]
        infos = infos.replace(
            expiries=infos.expiries.at[: self.n_agents, : self.n_products].add(expired)
        )
        wastage_cost = jnp.dot(expired, -params.wastage_costs)
        cumulative_rewards = cumulative_rewards + wastage_cost

        # Age stock
        stock = jnp.roll(stock, axis=1, shift=1)
        stock = stock.at[:n_products, 0].set(0)

        # Calculate holding cost
        holding = stock.sum(axis=-1)
        infos = infos.replace(
            holding=infos.holding.at[: self.n_agents, : self.n_products].add(holding)
        )
        holding_cost = jnp.dot(holding, -params.holding_costs)
        cumulative_rewards = cumulative_rewards + holding_cost

        # Receive the units ordered lead_time days ago
        # TODO If lead_time ==0, we wouldn't want to do this
        # Instead, we'd do a similar procedure immediately after order placed. Would be good to account for this.
        stock_received = self._sample_ages_on_arrival(
            key, params.age_on_arrival_distribution_probs, in_transit[:n_products, -1]
        )
        stock = stock + stock_received
        in_transit = jnp.roll(in_transit, axis=1, shift=1)
        in_transit = in_transit.at[:n_products, 0].set(0)

        infos = infos.replace(day_counter=infos.day_counter.at[:].add(1))
        state = state.replace(
            stock=stock,
            in_transit=in_transit,
            infos=infos,
            cumulative_rewards=cumulative_rewards,
            day=state.day + 1,
            time=jnp.ceil(state.time),
        )

        return state

    def _replenishment_step(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[EnvState, chex.Array]:
        """Replenishment action step."""
        stock, in_transit, infos, cumulative_rewards = (
            state.stock,
            state.in_transit,
            state.infos,
            state.cumulative_rewards,
        )
        orders = jnp.clip(action, a_min=0, a_max=self.max_order_quantities)
        infos = infos.replace(orders=infos.orders.at[:].add(orders))
        order_placed = jax.lax.cond(jnp.sum(orders) > 0, lambda: 1, lambda: 0)
        infos = infos.replace(order_placed=infos.order_placed.at[:].add(order_placed))

        # Place the order
        variable_order_cost = jnp.dot(orders, params.variable_order_costs)
        fixed_order_cost = order_placed * params.fixed_order_costs
        cumulative_rewards = cumulative_rewards + variable_order_cost + fixed_order_cost
        # TODO: Handle case where lead_time == 0
        in_transit = in_transit.at[0:n_products, 0].set(orders)

        state = state.replace(
            stock=stock,
            in_transit=in_transit,
            infos=infos,
            cumulative_rewards=cumulative_rewards,
        )

        return state

    def _issuing_step(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[EnvState, chex.Array]:
        stock, infos, cumulative_rewards = (
            state.stock,
            state.infos,
            state.cumulative_rewards,
        )
        # Raw action is one-hot encoded so both agents have same shape of action
        product_idx = jnp.argmax(action, axis=-1)

        # Log the demand for the type of unit
        infos = infos.replace(demand=infos.demand.at[:, state.request_type].add(1))

        # Record if there was a shortage; either we have selected action 0 or no stock of the allocated type
        shortage = jax.lax.select(
            jax.lax.bitwise_or(
                jnp.sum(action) == 0,
                stock[product_idx, 0 : self.max_useful_life].sum() == 0,
            ),
            1,
            0,
        )
        infos = infos.replace(
            shortages=infos.shortages.at[:, state.request_type].add(shortage)
        )
        shortage_cost = jnp.dot(
            params.shortage_costs,
            jnp.zeros(n_products).at[state.request_type].set(shortage),
        )
        cumulative_rewards = cumulative_rewards + shortage_cost

        # Issue the select unit FIFO if there isn't a shortage, otherwise do nothing
        stock = jax.lax.cond(
            shortage < 1,
            self._issue_one_unit,
            lambda stock, product_idx: stock,
            stock,
            product_idx,
        )

        # If we've allocated a unit (no shortage), then record the allocation and calculate the substitution cost
        allocations = jax.lax.select(
            shortage < 1,
            infos.allocations.at[:, state.request_type, product_idx].add(1),
            infos.allocations,
        )
        infos = infos.replace(allocations=allocations)
        substitution_cost = (allocations * params.substitution_costs).sum()
        cumulative_rewards = cumulative_rewards + substitution_cost

        # Sample the details of the next request
        request_interval, request_type = self._sample_next_request(key, state, params)

        state = state.replace(
            stock=stock,
            infos=infos,
            cumulative_rewards=cumulative_rewards,
            request_time=state.time + request_interval,
            request_type=request_type,
        )
        return state

    def _issue_one_unit(self, stock: chex.Array, product_idx: int) -> chex.Array:
        return stock.at[product_idx].set(self._issue_fifo(stock[product_idx], 1))

    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> Tuple[int, int]:
        """Fill demand with stock of one age, representing one element in the state"""
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _sample_next_request(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ) -> Tuple[int, int]:
        interval_key, product_key = jax.random.split(key, 2)
        # Interval until next request
        # Gamma distribution with concentration of 1 is equivalent to an Exponential dist - modelling time between Poisson events
        request_interval = distrax.Gamma(
            concentration=1, rate=params.poisson_demand_mean
        ).sample(seed=interval_key)
        # Type of unit (e.g. blood group of patient who next request made for)
        request_type = distrax.Categorical(params.product_probabilities).sample(
            seed=product_key
        )
        return request_interval, request_type

    def _sample_ages_on_arrival(
        self,
        key: chex.PRNGKey,
        age_on_arrival_distribution_probs: chex.Array,
        order_received: chex.Array,
    ) -> chex.Array:
        # The nice thing about this is that if we have different probs for each
        # product, we can feed them in an an array (n_prods x max_useful_life) and
        # this will still work
        d = distrax.Multinomial(
            total_count=order_received, probs=age_on_arrival_distribution_probs
        )
        return d.sample(seed=key)
