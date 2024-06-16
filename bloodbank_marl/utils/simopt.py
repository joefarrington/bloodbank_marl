from omegaconf.dictconfig import DictConfig
from typing import Dict, List
from bloodbank_marl.policies.common import HeuristicPolicy


def param_search_bounds_from_config(
    cfg: DictConfig, policy: HeuristicPolicy
) -> Dict[str, int]:
    """Create a dict of search bounds for each parameter from the conf
    g file"""
    # Specify search bounds for each parameter
    if cfg.param_search.search_bounds.all_params is None:
        try:
            search_bounds = {
                p: {
                    "low": cfg.param_search.search_bounds[p]["low"],
                    "high": cfg.param_search.search_bounds[p]["high"],
                }
                for p in policy.param_names.flat
            }
        except:
            raise ValueError(
                "Ranges for each parameter must be specified if not using same range for all parameters"
            )
    # Otherwise, use the same range for all parameters
    else:
        search_bounds = {
            p: {
                "low": cfg.param_search.search_bounds.all_params.low,
                "high": cfg.param_search.search_bounds.all_params.high,
            }
            for p in policy.param_names.flat
        }
    return search_bounds


def grid_search_space_from_config(
    search_bounds: Dict[str, int], policy: HeuristicPolicy
) -> Dict[str, List[int]]:
    """Create a grid search space from the search bounds"""
    search_space = {
        p: list(
            range(
                search_bounds[p]["low"],
                search_bounds[p]["high"] + 1,
            )
        )
        for p in policy.param_names.flat
    }
    return search_space
