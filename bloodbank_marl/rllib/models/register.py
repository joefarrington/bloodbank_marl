from ray.rllib.models import ModelCatalog
from bloodbank_marl.rllib.models.action_mask import (
    TorchActionMaskModel,
    TorchDictObsSpaceModel,
)


def register_custom_models():
    ModelCatalog.register_custom_model("TorchActionMaskModel", TorchActionMaskModel)
    ModelCatalog.register_custom_model("TorchDictObsSpaceModel", TorchDictObsSpaceModel)
