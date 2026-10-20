"""Movie-dubbing producer of the KevFLIX app (W10). Enabled with actor_beta.movie_pipeline.enabled (with actor_beta and
the actor speech engine). relay.coordinator.Coordinator.start() then runs MoviePipeline instead of the serial loop.

Modules: pipeline (MoviePipeline: the one coordinator thread + workers), stages (adapters around the real relay
functions), actors (fenced actor-id rebind/commit), snapshot (private analysis snapshots), outcomes (classification
and original-window publication with committed carried English), resident_demucs (resident GPU Demucs in the lease Job).
Interfaces with the guard/lease runtime (relay.app_runtime, W7): INTERFACES.md.
"""


def enabled(config):
    try:
        if config['speech_engine'] != 'actor':
            return False
    except (KeyError, TypeError):
        return False
    from .pipeline import options
    return options(config) is not None


def __getattr__(name):
    if name == 'MoviePipeline':
        from .pipeline import MoviePipeline
        return MoviePipeline
    raise AttributeError(name)
