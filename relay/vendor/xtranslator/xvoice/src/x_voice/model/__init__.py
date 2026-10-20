from x_voice.model.backbones.dit import DiT
from x_voice.model.backbones.mmdit import MMDiT
from x_voice.model.backbones.unett import UNetT
from x_voice.model.cfm import CFM
from x_voice.model.cfm_sft import CFM_SFT

__all__ = ["CFM", "CFM_SFT", "UNetT", "DiT", "MMDiT", "Trainer", "Trainer_SFT"]


def __getattr__(name):
    # Inference must not import training-only datasets/wandb/accelerate. Preserve
    # the upstream trainer API for callers that explicitly request training.
    if name == "Trainer":
        from x_voice.model.trainer import Trainer
        return Trainer
    if name == "Trainer_SFT":
        from x_voice.model.trainer_sft import Trainer_SFT
        return Trainer_SFT
    raise AttributeError(name)
