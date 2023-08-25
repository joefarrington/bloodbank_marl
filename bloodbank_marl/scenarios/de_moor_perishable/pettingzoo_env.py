import gymnasium
import numpy as np
import pettingzoo
import functools
import jax
import jax.numpy as jnp
import distrax


# NOTE: This will work as long as L >= 1, which is true for the cases considered in DeMoor.
# TODO: Might want action masking to prevent issuing when there is no stock of that age
# at the moment we are allowed to choose that action but it results in a shortage.

# NOTE: We have a sublcass that uses JAX sampling for the demand (the only stochastic component)
# This exists for testing purposes, because it allows us to directly compare results/transitions
# between this environment and the JAX multiagent implementation.


class DeMoorPerishableMA(pettingzoo.AECEnv):
    def __init__(
        self,
        render_mode=None,
        max_useful_life=2,
        lead_time=1,
        max_order_quantity=10,
        max_demand=100,
        demand_gamma_mean=4.0,
        demand_gamma_cov=0.5,
        variable_order_cost=3.0,
        shortage_cost=5.0,
        wastage_cost=7.0,
        holding_cost=1.0,
        max_days=365,
    ):
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity
        self.max_demand = max_demand

        self.demand_gamma_shape = 1 / (demand_gamma_cov**2)
        self.demand_gamma_scale = 1 / (demand_gamma_mean * demand_gamma_cov**2)

        self.variable_order_cost = variable_order_cost
        self.shortage_cost = shortage_cost
        self.wastage_cost = wastage_cost
        self.holding_cost = holding_cost

        self.max_days = max_days

        self.possible_agents = ["replenishment", "issuing"]

        self._np_random = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "replenishment":
            return gymnasium.spaces.Box(
                low=0,
                high=self.max_order_quantity,
                shape=(self.lead_time - 1 + self.max_useful_life,),
            )
        elif agent == "issuing":
            return gymnasium.spaces.Box(
                low=0, high=self.max_order_quantity, shape=(self.max_useful_life,)
            )
        else:
            raise ValueError("Agent must be one of {}".format(self.possible_agents))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == "replenishment":
            return gymnasium.spaces.Discrete(self.max_order_quantity + 1)
        elif agent == "issuing":
            return gymnasium.spaces.Discrete(self.max_useful_life + 1)
        else:
            raise ValueError("Agent must be one of {}".format(self.possible_agents))

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

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
        #

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
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {
            "replenishment": {"holding": 0, "wastage": 0, "shortage": 0, "demand": 0},
            "issuing": {},
        }

        self.stock = np.zeros(self.max_useful_life)
        self.in_transit = np.zeros(self.lead_time)

        self.time = 0
        self.remaining_demand = 0

        self.state = {
            "replenishment": np.hstack([self.in_transit[1:], self.stock]),
            "issuing": self.stock,
        }
        self.observations = self.state

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self.agent_selection = "replenishment"

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
        if agent == "replenishment":
            self.infos["replenishment"] = {
                "holding": 0,
                "wastage": 0,
                "shortage": 0,
                "demand": 0,
            }
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards[agent] = 0

        if agent == "replenishment":
            self._replenishment_step(action)
        elif agent == "issuing":
            self._issuing_step(action)
        else:
            raise ValueError("Agent must be one of {}".format(self.possible_agents))

        # Select the next agent
        if (
            self.remaining_demand == 0
        ):  # End of day, so need to do expiries and then place next order
            self._age_stock()
            self.agent_selection = "replenishment"
            self.time += 1
        else:
            self.agent_selection = "issuing"

        # Update the state and observation for each agent
        self.state["replenishment"] = np.hstack([self.in_transit[1:], self.stock])
        self.observations["replenishment"] = self.state["replenishment"]
        self.state["issuing"] = self.stock
        self.observations["issuing"] = self.state["issuing"]

        # For now, use truncation rather than termination
        # TODO: Check how this should really be handled in PettingZoo
        self.truncations = {agent: self.time >= self.max_days for agent in self.agents}

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def _replenishment_step(self, action):
        # Clip order to between 0 and maximum order
        order = np.clip(action, 0, self.max_order_quantity)

        # Charge the variable order cost
        for agent in self.agents:
            self.rewards[agent] -= order * self.variable_order_cost

        # Place the order
        self.in_transit[0] = order

        # Sample demand from trucated gamma distribution
        self.remaining_demand = np.clip(
            np.round(
                self.np_random.gamma(self.demand_gamma_shape, self.demand_gamma_scale)
            ),
            0,
            self.max_demand,
        )
        self.infos["replenishment"]["demand"] += self.remaining_demand

    def _issuing_step(self, action):
        # Update the remaining demand
        self.remaining_demand -= 1

        # If we choose not to issue a unit, or have no stock of that age, then we have a shortage
        if action == 0 or self.stock[action - 1] == 0:
            for agent in self.agents:
                self.rewards[agent] -= self.shortage_cost
            self.infos["replenishment"]["shortage"] += 1
        # Otherwise issue the unit specified by the action
        else:
            self.stock[action - 1] -= 1

    def _age_stock(self):
        # Age the stock by one day and apply expiry cost
        expired = self.stock[-1]
        self.infos["replenishment"]["wastage"] += expired
        for agent in self.agents:
            self.rewards[agent] -= expired * self.wastage_cost

        # Age stock
        self.stock = np.roll(self.stock, 1)
        self.stock[0] = 0

        # Calculate units in stock at end of day and apply holding cost
        holding = self.stock.sum()
        self.infos["replenishment"]["holding"] += holding
        for agent in self.agents:
            self.rewards[agent] -= holding * self.holding_cost

        # Receive the units ordered L days ago
        self.stock[0] = self.in_transit[-1]
        self.in_transit = np.roll(self.in_transit, 1)
        self.in_transit[0] = 0


class DeMoorPerishableMAJAXSampling(DeMoorPerishableMA):
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        #

        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """

        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        else:
            self.rng = jax.random.PRNGKey(0)

        super().reset(seed, options)

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

        # Split the key twice to represent the split outside the env
        # and the split inside the step function
        rng_step, self.rng = jax.random.split(self.rng)
        key, key_rest = jax.random.split(rng_step)

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        if agent == "replenishment":
            self.infos["replenishment"] = {
                "holding": 0,
                "wastage": 0,
                "shortage": 0,
                "demand": 0,
            }
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards[agent] = 0

        if agent == "replenishment":
            self._replenishment_step(key, action)
        elif agent == "issuing":
            self._issuing_step(action)
        else:
            raise ValueError("Agent must be one of {}".format(self.possible_agents))

        # Select the next agent
        if (
            self.remaining_demand == 0
        ):  # End of day, so need to do expiries and then place next order
            self._age_stock()
            self.agent_selection = "replenishment"
            self.time += 1
        else:
            self.agent_selection = "issuing"

        # Update the state and observation for each agent
        self.state["replenishment"] = np.hstack([self.in_transit[1:], self.stock])
        self.observations["replenishment"] = self.state["replenishment"]
        self.state["issuing"] = self.stock
        self.observations["issuing"] = self.state["issuing"]

        # For now, use truncation rather than termination
        # TODO: Check how this should really be handled in PettingZoo
        self.truncations = {agent: self.time >= self.max_days for agent in self.agents}

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def _replenishment_step(self, key, action):
        # Clip order to between 0 and maximum order
        order = np.clip(action, 0, self.max_order_quantity)

        # Charge the variable order cost
        for agent in self.agents:
            self.rewards[agent] -= order * self.variable_order_cost

        # Place the order
        self.in_transit[0] = order

        # Sample demand from trucated gamma distribution
        demand_dist = distrax.Gamma(
            concentration=self.demand_gamma_shape, rate=self.demand_gamma_scale
        )
        self.remaining_demand = int(
            jnp.round(
                demand_dist.sample(seed=key)
            ).clip(  # Round because Gamma is continuous and demand discrete
                0, self.max_demand
            )  # Truncate at max demand
        )
        self.infos["replenishment"]["demand"] += self.remaining_demand
