from orbax.checkpoint import CheckpointManager
import jax
import jax.numpy as jnp
import pandas as pd
from typing import Optional


def load_flax_policy_params(
    param_key: str,
    checkpoint_manager: CheckpointManager,
    reshape_for_fitness: bool = True,
    internal_key: Optional[str] = None,
    checkpoint_id: Optional[int] = None,
    latest_checkpoint: bool = False,
):
    """Load policy parameters from a checkpoint"""

    if latest_checkpoint and checkpoint_id is not None:
        raise ValueError("Choose either latest_checkpoint or checkpoint_id, not both")

    if latest_checkpoint:
        checkpoint_id = checkpoint_manager.latest_step()

    policy_params = checkpoint_manager.restore(checkpoint_id)[param_key]
    if internal_key:
        policy_params = policy_params[internal_key]
    if reshape_for_fitness:
        # This will add a batch dimension to the policy parameters
        policy_params = jax.tree_map(lambda x: x.reshape((1,) + x.shape), policy_params)
    return policy_params


def load_heuristic_policy_params(filepath: str, reshape_for_fitness: bool = True):
    """Load policy parameters from a file"""
    policy_params = jnp.load(filepath)
    if reshape_for_fitness:
        # This will add a batch dimension to the policy parameters
        policy_params = policy_params.reshape((1,) + policy_params.shape)
    return policy_params


def convert_viso_jax_output_to_params(
    in_filepath: str, new_shape: tuple, out_filepath: str
):
    """In viso_jax, we saved best policy as pandas df. Us this function to convert it to jax array
    that can then be loaded using load_vi_policy_params"""
    policy_params = pd.read_csv(in_filepath, index_col=0)
    jnp.save(out_filepath, policy_params.values.reshape(new_shape))
    return None


def load_vi_policy_params(filepath: str, reshape_for_fitness: bool = True):
    """Load policy parameters from a file"""
    policy_params = jnp.load(filepath)
    if reshape_for_fitness:
        # This will add a batch dimension to the policy parameters
        policy_params = policy_params.reshape((1,) + policy_params.shape)
    return policy_params


def load_none(reshape_for_fitness: bool = True):
    """Load a placeholder parameter when no param needed (e.g OUFO)"""
    policy_params = jnp.array(0)
    if reshape_for_fitness:
        # This will add a batch dimension to the policy parameters
        policy_params = jnp.array([[0]])
    return policy_params
