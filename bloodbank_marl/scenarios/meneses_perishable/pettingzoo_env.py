import gymnasium
import numpy as np
import pettingzoo
import functools
import distrax
from functools import partial
from typing import List, Optional, Dict, Union
import chex

# NOTE: We should add a sublcass that uses JAX sampling for the demand (the only stochastic component)
# for testing purposes, because it allows us to directly compare results/transitions
# between this environment and the JAX multiagent implementation.

# NOTE: For now, we follow the MA JAX version of the env where we sample intervals, but we might not
# want to bother

# TODO: Go through carefully and think about how we are updating the time and the day.
# Firstly is it correct and secondly is it clear

# Do we want to move to a more functional approach to make the connections to the Gymnax multiagent environment
# more clear?

# TODO: Handle warmup if needed

# TODO: We could add the target KPI penalties

# TODO: With obs spaces, we should try to make sure that the way we create the flat "observations" key is consistent with how it
# would be done automtically by gymnasium. May want the inner components to be nested, e.g action_mask, raw_obs (stock, requested_type, etc), t
# then observations which is flat.

C = 1e10
substitution_cost_ratios = [
    # Unit O-, O+, A-, A+, B-, B+, AB-, AB+
    [0, C, C, C, C, C, C, C],  # O- pt
    [
        1 / 8,
        0,
        C,
        C,
        C,
        C,
        C,
        C,
    ],  # O+ pt
    [
        1 / 8,
        C,
        0,
        C,
        C,
        C,
        C,
        C,
    ],  # A- pt
    [3 / 8, 2 / 8, 1 / 8, 0, C, C, C, C],  # A+ pt
    [
        1 / 8,
        C,
        C,
        C,
        0,
        C,
        C,
        C,
    ],  # B- pt
    [3 / 8, 2 / 8, C, C, 1 / 8, 0, C, C],  # B+ pt
    [3 / 8, C, 2 / 8, C, 1 / 8, C, 0, C],  # AB- pt
    [7 / 8, 6 / 8, 5 / 8, 4 / 8, 3 / 8, 2 / 8, 1 / 8, 0],  # AB+ pt
]


class MenesesPerishableMA(pettingzoo.AECEnv):
    def __init__(
        self,
        render_mode=None,
        n_products: int = 8,
        max_useful_life: int = 35,
        lead_time: int = 1,
        max_order_quantities: List[int] = [100] * 8,
        max_demand: int = 100,
        poisson_demand_mean: float = 49.8,
        product_probabilities: List[float] = [
            0.08614457,
            0.36024097,
            0.08192771,
            0.36485943,
            0.0126506,
            0.0684739,
            0.00522088,
            0.02048193,
        ],
        age_on_arrival_distribution_probs: List[float] = [1] + [0] * (35 - 1),
        fixed_order_costs: float = 0,
        variable_order_costs: List[float] = [160] * 8,
        shortage_costs: List[float] = [1340] * 8,
        wastage_costs: List[float] = [130] * 8,
        holding_costs: List[float] = [1.1] * 8,
        substitution_cost_ratios: List[List[float]] = substitution_cost_ratios,
        max_substitution_cost: float = 1340,
        max_days_in_episode=365,
    ):
        self.render_mode = render_mode

        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time

        self.max_order_quantities = np.array(max_order_quantities)
        self.max_demand = max_demand

        self.poisson_demand_mean = poisson_demand_mean
        self.product_probabilities = product_probabilities
        self.age_on_arrival_distribution_probs = age_on_arrival_distribution_probs

        self.fixed_order_costs = fixed_order_costs
        self.variable_order_costs = np.array(variable_order_costs)
        self.shortage_costs = np.array(shortage_costs)
        self.wastage_costs = np.array(wastage_costs)
        self.holding_costs = np.array(holding_costs)
        self.substitution_costs = (
            np.array(substitution_cost_ratios) * max_substitution_cost
        )

        self.max_days_in_episode = max_days_in_episode

        self.agents = ["replenishment", "issuing"]
        self.possible_agents = self.agents[:]

        self.observation_spaces = {
            "replenishment": gymnasium.spaces.Dict(
                {
                    "action_mask": gymnasium.spaces.Box(
                        low=0,
                        high=1,
                        shape=(
                            self.n_products,
                        ),  # Will not be used, just for consistency
                    ),
                    "in_transit": gymnasium.spaces.Box(
                        low=0,
                        high=np.max(self.max_order_quantities),
                        shape=(
                            self.n_products,
                            self.lead_time - 1,
                        ),
                    ),
                    "stock": gymnasium.spaces.Box(
                        low=0,
                        high=np.max(self.max_order_quantities),
                        shape=(self.n_products, self.max_useful_life),
                    ),
                    "observations": gymnasium.spaces.Box(
                        low=0,
                        high=np.max(self.max_order_quantities),
                        shape=(
                            self.n_products
                            * (self.lead_time + self.max_useful_life - 1),
                        ),
                    ),
                }  # Flat
            ),
            "issuing": gymnasium.spaces.Dict(
                {
                    "action_mask": gymnasium.spaces.Box(
                        low=0, high=1, shape=(self.n_products + 1,)
                    ),
                    "stock": gymnasium.spaces.Box(
                        low=0,
                        high=np.max(self.max_order_quantities),
                        shape=(self.n_products, self.max_useful_life),
                    ),
                    "requested_type": gymnasium.spaces.Discrete(self.n_products),
                    "observations": gymnasium.spaces.Box(
                        low=0,
                        high=np.max(self.max_order_quantities),
                        shape=(
                            self.n_products * (self.max_useful_life + 1),
                        ),  # Reflect the fact that when flatted by gymnasium, Discrete turns to one-hot
                    ),
                }
            ),
        }
        self.action_spaces = {
            "replenishment": gymnasium.spaces.Box(
                low=0, high=self.max_order_quantities, shape=(self.n_products,)
            ),
            "issuing": gymnasium.spaces.Discrete(
                self.n_products + 1
            ),  # idx 0 for nothing issued, then 1 for each product in order
        }

        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {
            "replenishment": self._get_empty_info(),
            "issuing": self._get_empty_info(),
        }

        self.agent_selection = "replenishment"

        self._np_random = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        if agent == "replenishment":
            return self._observe_replenishment()
        elif agent == "issuing":
            return self._observe_issuing()
        else:
            raise ValueError("Agent must be one of {}".format(self.possible_agents))

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed."""
        if self._np_random is None:
            self._np_random, seed = gymnasium.utils.seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """

        if seed is not None:
            self._np_random, seed = gymnasium.utils.seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {
            "replenishment": self._get_empty_info(),
            "issuing": self._get_empty_info(),
        }

        self.stock = np.zeros((self.n_products, self.max_useful_life))
        self.in_transit = np.zeros((self.n_products, self.lead_time))

        self.request_time = 0.0
        self.request_type = 0
        self.request_intervals = np.zeros(self.max_demand + 1)
        self.request_types = np.zeros(self.max_demand + 1)
        self.request_idx = 0

        self.day = 0
        self.time = 0
        self.remaining_demand = 0

        self.state = {
            "replenishment": {
                "in_transit": self.in_transit[:, 1:],
                "stock": self.stock,
            },
            "issuing": {
                "in_transit": self.in_transit[:, 1:],
                "stock": self.stock,
                "request_type": self.request_types[self.request_idx],
            },
        }
        # TODO: Check if this next line is what we intend
        self.observations = {a: self.observe(a) for a in self.agents}

        self.agent_selection = "replenishment"

    def _get_empty_info(self):
        return {
            "demand": np.zeros(self.n_products),
            "shortages": np.zeros(self.n_products),
            "expiries": np.zeros(self.n_products),
            "holding": np.zeros(self.n_products),
            "allocations": np.zeros(
                (self.n_products, self.n_products, self.max_useful_life)
            ),
            "orders": np.zeros(self.n_products),
            "order_placed": 0,
            "day_counter": 0,
        }

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        # TODO: Update when we decide exactly what will go in info

        self.infos[agent] = self._get_empty_info()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards[agent] = 0.0

        if agent == "replenishment":
            self._replenishment_step(action)
        elif agent == "issuing":
            self._issuing_step(action)
        else:
            raise ValueError("Agent must be one of {}".format(self.possible_agents))

        self.request_time = self.time + self.request_intervals[self.request_idx]
        self.request_type = self.request_types[self.request_idx]

        # TODO: Check what the condition should be for request_idx based on where it gets updated
        # Basically,
        if self.request_time >= self.day + 1 or self.request_idx >= self.max_demand:
            self._age_stock()
            # After aging stock it is now a new day
            self.day += 1
            self.time = np.ceil(self.time)
            self.agent_selection = "replenishment"
        else:
            self.agent_selection = "issuing"
            self.time = self.request_time

        # Update the state and observation for each agent
        self.state["replenishment"] = {
            "in_transit": self.in_transit[:, 1:],
            "stock": self.stock,
        }
        self.observations["replenishment"] = self.state["replenishment"]
        self.state["issuing"] = {
            "in_transit": self.in_transit[:, 1:],
            "stock": self.stock,
            "request_type": self.request_type,
        }
        self.observations["issuing"] = self.state["issuing"]

        # For now, use truncation rather than termination
        # TODO: Check how this should really be handled in PettingZoo
        self.truncations = {
            agent: self.time + self.request_intervals[self.request_idx + 1]
            >= self.max_days_in_episode
            for agent in self.agents
        }

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def _replenishment_step(self, action):
        # Clip order to between 0 and maximum order
        orders = np.clip(action, a_min=0, a_max=self.max_order_quantities)
        order_placed = 1 if orders.sum() > 0 else 0

        # Place the order
        variable_order_cost = np.dot(orders, -self.variable_order_costs)
        fixed_order_cost = order_placed * -self.fixed_order_costs
        for agent in self.agents:
            self.rewards[agent] += variable_order_cost + fixed_order_cost
        # TODO: Handle case where lead_time == 0
        self.in_transit[:, 0] = orders

        # Sample the demand for the coming day
        self.request_intervals = self.np_random.gamma(
            shape=1, scale=(1 / self.poisson_demand_mean), size=(self.max_demand,)
        )
        self.request_types = self.np_random.choice(
            self.n_products, size=(self.max_demand,), p=self.product_probabilities
        )

        # Reset request index
        self.request_idx = 0

        # Record to info
        for agent in self.agents:
            self.infos[agent]["orders"] += orders
            self.infos[agent]["order_placed"] += order_placed

    def _issuing_step(self, action):
        # Product_idx is action -1 because action 0 is no product issued
        product_idx = action - 1

        # Record if there was a shortage; either we have selected action 0 or no stock of the allocated type
        shortage = 1 if ((self.stock[product_idx].sum() == 0) or (action == 0)) else 0
        shortage_cost = -self.shortage_costs[self.request_type] * shortage
        for agent in self.agents:
            self.rewards[agent] += shortage_cost

        # Issue the select unit FIFO if there isn't a shortage, otherwise do nothing
        if shortage == 0:
            stock_before_issue = self.stock.copy()
            self._issue_one_unit(product_idx)

            # If we've allocated a unit (no shortage), then record the allocation and calculate the substitution cost
            issued = stock_before_issue - self.stock

            # TODO Something here with allocations
            # TODO: Works here because dealing with one demand at a time
            substitution_cost = -self.substitution_costs[self.request_type, product_idx]

            for agent in self.agents:
                self.rewards[agent] += substitution_cost
                self.infos[agent]["allocations"][self.request_type] += issued

        # Record to info
        for agent in self.agents:
            self.infos[agent]["demand"][self.request_type] += 1
            self.infos[agent]["shortages"][self.request_type] += (
                1 if shortage == 1 else 0
            )

        # Updated the request_index
        self.request_idx += 1

    def _age_stock(self):
        # Age the stock by one day and calculate wastage cost
        expired = self.stock[:, -1]
        wastage_cost = np.dot(expired, -self.wastage_costs)
        for agent in self.agents:
            self.rewards[agent] += wastage_cost

        # Age stock
        self.stock = np.roll(self.stock, axis=1, shift=1)
        self.stock[:, 0] = 0

        # Calculate holding cost
        holding = self.stock.sum(axis=-1)
        holding_cost = np.dot(holding, -self.holding_costs)
        for agent in self.agents:
            self.rewards[agent] += holding_cost

        # Receive the units ordered lead_time days ago
        # TODO If lead_time ==0, we wouldn't want to do this
        # Instead, we'd do a similar procedure immediately after order placed. Would be good to account for this.
        stock_received = self._sample_ages_on_arrival(self.in_transit[:, -1])
        self.stock = self.stock + stock_received
        self.in_transit = np.roll(self.in_transit, axis=1, shift=1)
        self.in_transit[:, 0] = 0

        # Record to info
        for agent in self.agents:
            self.infos[agent]["expiries"] += expired
            self.infos[agent]["holding"] += holding
            self.infos[agent]["day_counter"] += 1

    def _issue_one_unit(self, product_idx):
        self.stock[product_idx] = self._issue_fifo(self.stock[product_idx])

    def _issue_fifo(self, stock_of_unit):
        age_idx = (self.max_useful_life - 1) - (stock_of_unit[::-1] > 0).argmax()
        stock_of_unit[age_idx] -= 1
        return np.clip(stock_of_unit, a_min=0, a_max=None)

    def _get_next_request(self):
        raise NotImplementedError

    def _sample_ages_on_arrival(self, order_received: np.ndarray):
        return self.np_random.multinomial(
            n=order_received.astype(int), pvals=self.age_on_arrival_distribution_probs
        )

    def _observe_replenishment(self):
        action_mask = np.ones(self.n_products)
        observations = np.hstack(
            [self.in_transit[:, 1:].reshape(-1), self.stock.reshape(-1)]
        )
        return {
            "action_mask": action_mask,
            "in_transit": self.in_transit[:, 1:],
            "stock": self.stock,
            "observations": observations,
        }

    def _observe_issuing(self):
        action_mask = np.hstack(
            [np.array([1]), (self.stock.sum(axis=1) > 0).astype(int)]
        )
        request_type_one_hot = np.zeros(self.n_products)
        request_type_one_hot[self.request_type] + 1
        observations = np.hstack([self.stock.reshape(-1), request_type_one_hot])
        return {
            "action_mask": action_mask,
            "stock": self.stock,
            "request_type": self.request_type,
            "observations": observations,
        }

    @classmethod
    def calculate_kpis(cls, cum_info):
        """Calculate KPIs based on accumulated info from one agent"""

        mean_demand_by_pt_blood_group = cum_info["demand"] / cum_info["day_counter"]
        mean_order_by_product = cum_info["orders"] / cum_info["day_counter"]
        service_level_pc_by_pt_blood_group = (
            (cum_info["demand"] - cum_info["shortages"]) * 100
        ) / cum_info["demand"]
        expiries_pc_by_product = (cum_info["expiries"] * 100) / cum_info["orders"]
        mean_holding_by_product = cum_info["holding"] / cum_info["day_counter"]
        mean_age_at_transfusion_by_pt_blood_group = (
            cls._calculate_mean_age_at_transfusion_by_pt_blood_group(cum_info)
        )
        exact_match_pc_by_pt_blood_group = (
            cls._calculate_exact_match_pc_by_pt_blood_group(cum_info)
        )
        mean_total_order = cum_info["orders"].sum(axis=-1) / cum_info["day_counter"]
        service_level_pc = (
            ((cum_info["demand"].sum(axis=(-1)) - cum_info["shortages"].sum(axis=-1)))
            * 100
            / cum_info["demand"].sum(axis=-1)
        )
        expiries_pc = (cum_info["expiries"].sum(axis=-1) * 100) / cum_info[
            "orders"
        ].sum(axis=-1)
        mean_holding = cum_info["holding"].sum(axis=-1) / cum_info["day_counter"]
        exact_match_pc = cls._calculate_exact_match_pc(cum_info)
        mean_age_at_transfusion = cls._calculate_mean_age_at_transfusion(cum_info)
        unmet_demand_units = cum_info["shortages"].sum(axis=-1)
        expired_units = cum_info["expiries"].sum(axis=-1)

        return {
            "mean_demand_by_pt_blood_group": mean_demand_by_pt_blood_group,
            "mean_order_by_product": mean_order_by_product,
            "service_level_%_by_pt_blood_group": service_level_pc_by_pt_blood_group,
            "expiries_%_by_product": expiries_pc_by_product,
            "mean_holding_by_product": mean_holding_by_product,
            "mean_age_at_transfusion_by_pt_blood_group": mean_age_at_transfusion_by_pt_blood_group,
            "exact_match_%_by_pt_blood_group": exact_match_pc_by_pt_blood_group,
            "mean_total_order": mean_total_order,
            "service_level_%": service_level_pc,
            "expiries_%": expiries_pc,
            "mean_holding": mean_holding,
            "exact_match_%": exact_match_pc,
            "mean_age_at_transfusion": mean_age_at_transfusion,
            "unmet_demand_units": unmet_demand_units,
            "expired_units": expired_units,
        }

    @classmethod
    def _calculate_exact_match_pc_by_pt_blood_group(cls, cum_info: Dict) -> chex.Array:
        """Calculate the exact match percentage by product type and blood group"""
        n_groups = cum_info["allocations"].shape[-2]
        exact_matches_by_pt_blood_group = cum_info["allocations"].sum(axis=(-1))[
            np.arange(n_groups), np.arange(n_groups)
        ]
        total_allocated_by_pt_blood_group = cum_info["allocations"].sum(axis=(-2, -1))
        return (
            exact_matches_by_pt_blood_group * 100
        ) / total_allocated_by_pt_blood_group

    @classmethod
    def _calculate_exact_match_pc(cls, cum_info: Dict) -> float:
        """Calculate the exact match percentage"""
        exact_matches = np.trace(
            cum_info["allocations"].sum(axis=(-1)),
            axis1=-2,
            axis2=-1,
        )
        total_allocated = cum_info["allocations"].sum()
        return (exact_matches * 100) / total_allocated

    @classmethod
    def _calculate_mean_age_at_transfusion_by_pt_blood_group(
        cls, cum_info: Dict
    ) -> chex.Array:
        """Calculate the mean age at transfusion by patient blood group (i.e. irrespective of what they were allocated)"""
        # TODO: We might also want to do this by product type
        ages = np.arange(cum_info["allocations"].shape[-1])
        age_weighted_allocations = cum_info["allocations"] * ages[None, None, :]
        total_age_per_request_type = age_weighted_allocations.sum(axis=(-2, -1))
        total_allocated_per_request_type = cum_info["allocations"].sum(axis=(-2, -1))
        return total_age_per_request_type / total_allocated_per_request_type

    @classmethod
    def _calculate_mean_age_at_transfusion(cls, cum_info: Dict) -> float:
        """Calculate the mean age at transfusion"""
        ages = np.arange(cum_info["allocations"].shape[-1])
        age_weighted_allocations = cum_info["allocations"].sum(axis=(-3, -2)) * ages
        return np.sum(age_weighted_allocations) / np.sum(cum_info["allocations"])
