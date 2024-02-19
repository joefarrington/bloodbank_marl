from orbax.checkpoint import CheckpointManager
import jax
import jax.numpy as jnp


def load_flax_policy_params(
    param_key: str, checkpoint_id: int, checkpoint_manager: CheckpointManager
):
    """Load policy parameters from a checkpoint and reshape for use with our fitness functions"""
    policy_params = checkpoint_manager.restore(checkpoint_id)[param_key]
    policy_params = jax.tree_map(lambda x: x.reshape((1,) + x.shape), policy_params)
    return policy_params


def load_heuristic_policy_params(filepath: str):
    """Load policy parameters from a file, by default when saved from simopt will already be ready for use in fitness functions"""
    policy_params = jnp.load(filepath)
    return policy_params
