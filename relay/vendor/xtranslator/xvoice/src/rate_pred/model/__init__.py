from .speed_predictor import SpeedPredictor

__all__ = ["Trainer", "SpeedPredictor"]


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    raise AttributeError(name)
