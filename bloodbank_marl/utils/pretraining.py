import jax
import jax.numpy as jnp
import torch
from bloodbank_marl.scenarios.de_moor_perishable.gymnax_env import (
    EnvObs as DeMoorEnvObs,
)
from bloodbank_marl.utils.gymnax_fitness import make
import itertools
import chex


def collate_fn_pytree(batch):
    # https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    def _tree_stack(trees):
        """Takes a list of trees and stacks every corresponding leaf.
        For example, given two trees ((a, b), c) and ((a', b'), c'), returns
        ((stack(a, a'), stack(b, b')), stack(c, c')).
        Useful for turning a list of objects into something you can feed to a
        vmapped function.
        """
        leaves_list = []
        treedef_list = []
        for tree in trees:
            leaves, treedef = jax.tree_util.tree_flatten(tree)
            leaves_list.append(leaves)
            treedef_list.append(treedef)

        grouped_leaves = zip(*leaves_list)
        result_leaves = [jnp.stack(l) for l in grouped_leaves]
        return treedef_list[0].unflatten(result_leaves)

    xs = _tree_stack([batch[i][0] for i in range(len(batch))])
    ys = jnp.hstack([batch[i][1] for i in range(len(batch))])
    return xs, ys


def collate_fn_single_label(batch):
    xs = jnp.vstack([batch[i][0] for i in range(len(batch))])
    ys = jnp.array([batch[i][1] for i in range(len(batch))])
    return xs, ys


def collate_fn_multi_label(batch):
    xs = jnp.vstack([batch[i][0] for i in range(len(batch))])
    # Note we use vstack here because the targets are multidimensional (one for each blood group)
    ys = jnp.vstack([batch[i][1] for i in range(len(batch))])
    return xs, ys


class RepDataset(torch.utils.data.Dataset):
    def __init__(self, obs, targets):
        self.obs = obs
        self.targets = targets

    def __getitem__(self, index):
        return jax.tree_util.tree_map(lambda x: x[index], self.obs), self.targets[index]

    def __len__(self):
        return len(self.targets)


@jax.jit
def ordinal_categorical_cross_entropy_with_integer_labels(
    logits: chex.Array,
    labels: chex.Array,
) -> chex.Array:
    """Computes ordinal categorical cross entropy between sets of logits and integer labels

    Formatted to match as optax loss function, and Adapted from the the optax implementation of
    softmax_cross_entropy_with_integer_labels
    https://github.com/google-deepmind/optax/blob/main/optax/losses/_classification.py#L106#L138

    As described here:
    https://stats.stackexchange.com/questions/87826/machine-learning-with-ordered-labels/611604#611604

    Args:
      logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
      labels: Integers specifying the correct class for each input, with shape
        `[...]`.

    Returns:
      Ordinal categorical cross entropy between each prediction and the corresponding target
      distributions, with shape `[...]`.
    """
    chex.assert_type([logits], float)
    chex.assert_type([labels], int)

    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)
    label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    ce_loss = log_normalizers - label_logits

    weighting = jnp.abs(jnp.argmax(logits, axis=-1) - labels) / (logits.shape[-1] - 1)

    return (1 + weighting) * ce_loss


def get_obs_de_moor_perishable(env_kwargs, env_params, stock_limit=None):
    """Function to get states for pretraining for DeMoor perishable scenario. Function will calculate
    all possible states, and then filter out any with stock in transit and on hand greater than `stock_limit`
    if `stock_limit` is provided.
    """
    env, default_env_params = make("DeMoorPerishableGymnax", **env_kwargs)
    env_params = default_env_params.replace(**env_params)
    max_order_quantity = env.max_order_quantity
    lead_time = env.lead_time
    max_useful_life = env.max_useful_life

    possible_orders = range(0, max_order_quantity + 1)
    product_arg = [possible_orders] * (max_useful_life + lead_time - 1)
    state_tuples = jnp.array(list(itertools.product(*product_arg)))

    if stock_limit is not None:
        filtered_state_tuples = state_tuples[
            jnp.sum(state_tuples, axis=1) <= stock_limit
        ]
    else:
        filtered_state_tuples = state_tuples

    action_mask = jnp.ones((len(state_tuples), max_order_quantity + 1), dtype=jnp.int32)
    all_obs = DeMoorEnvObs(
        stock=state_tuples[:, lead_time - 1 :],
        action_mask=action_mask,
        in_transit=state_tuples[:, : lead_time - 1],
    )
    action_mask = jnp.ones(
        (len(filtered_state_tuples), max_order_quantity + 1), dtype=jnp.int32
    )
    filtered_obs = DeMoorEnvObs(
        stock=filtered_state_tuples[:, lead_time - 1 :],
        action_mask=action_mask,
        in_transit=filtered_state_tuples[:, : lead_time - 1],
    )
    return filtered_obs


# Preprocessing
def get_obs(x):
    """Enable the use of EnvObs.obs as a preprocessing function."""
    return x.obs


def passthrough(x):
    """Used when no preprocessing is required."""
    return x
