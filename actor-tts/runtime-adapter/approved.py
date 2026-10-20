"""The Kevin-approved movie Vox engine: its exact model, runtime artifacts, synthesis settings and refused references.

Pure constants and path rules, no machine locations: the runtime manifest, model and native libraries are located by
config (`actor_runtime_manifest`); this file pins only their identities (sha256). No relative imports: it is imported
both as vox_actor_adapter.approved (renderer/host) and as a top-level module by the child script (vox_movie_child.py).
"""

from pathlib import Path

PROTOCOL = 'vox-movie-child-v2'
ENGINE = 'VoxCPM2/audio.cpp'
QUANTIZATION = 'Q4_K, AudioVAE F16'
MODEL_SHA256 = '1d0b2970e08894e086bed0f1af5a041ebce7ee9801bb667086b7f9b9a76f896e'
MODEL_BYTES = 1697232640
# The approved native runtime, by content: every artifact the manifest lists must have exactly these hashes (the
# manifest's own paths are machine-specific, so the manifest FILE is not pinned; ActorVoice re-verifies every artifact
# and the model at load).
RUNTIME_ARTIFACTS_SHA256 = {
    'audiocpp.dll': 'cb04189d9aba2829ce0d065ba42d7163608aea6e3883bb320522c8826f4c6513',
    'cublas64_13.dll': 'f1d500d0cd892f5b8c6b6cdbffd82d0c55d5f5427215668e7ceb55aeeccc1b63',
    'cublasLt64_13.dll': 'b592cd016d7673e9cb97716a22b27c4010ee635377a3ba28f37070a9bdb76a68',
    'cudart64_13.dll': 'b00ca6f53699120da815bf3e06e2e4285fae2f201235b883dcbb50eec51e2a2a',
    'cufft64_12.dll': '35e531c8b4796b1a073948447dba618069fc839ded3da625416dfc7de496c1a1',
    'ggml-base.dll': '701ec90f031a9b3cdd78a74136588b3d1d251dcc731e2ee5e60cb0d8ec9be0dc',
    'ggml-cpu-alderlake.dll': '9433dd2d6aede80e55c5214116208f93b2e189bcfda92a337d888f2e4c310a95',
    'ggml-cpu-cannonlake.dll': '32b4f7361b0ce5cab9a66f97ef0f8d33f99cec29bf5ea200aec570bf8bf2df8e',
    'ggml-cpu-cascadelake.dll': 'd7d6a9840db1dc2b5917f70708d00f0aa05b94434d0fd482d52855ba8be464ae',
    'ggml-cpu-haswell.dll': '266f77568f93de080b1f67084b9ef7079b8bac72031e4a047319fee347607b9f',
    'ggml-cpu-icelake.dll': 'bed42ca0315ec2622bc6ab28f7c17dae0b2f0909b135d79451a4241f50342433',
    'ggml-cpu-sandybridge.dll': '8e2937a6622ea34314ee8963307c6c879edef94c105b564b5229afef828d0d5e',
    'ggml-cpu-skylakex.dll': 'c4cb78705b28fc9c96cfd57b794eac32e8dd194e1f4feb9893bd8bcc54cae90c',
    'ggml-cpu-sse42.dll': 'a7062fb5c0eaa0d4f6ae1997f3b02917273703e277b4097b321f00089be80b3a',
    'ggml-cpu-x64.dll': '2cfb166b4a2af995caf9a8c5b6bbaffd9cc1f0aece90468af7f698a997297267',
    'ggml-cuda.dll': '66df3bce1a32dcb0179cc493dd6a54dea5f78849fbe21e08b1252cd64ff70e6a',
    'ggml.dll': '91b52acb0501ac083140a911aa739cefaff72290a068aa879a443fa870c076d7',
    'LICENSE': '7e125adced85b7f84c04371179d15d51fbd702cefd060a37043fe38d6eb1722e',
    'msvcp140_1.dll': 'bfad5aef4c63a669e3c140655cdfdf395b6c979b400a447bd5dcb65ed8826c3d',
    'msvcp140_2.dll': '3ea06f0ee098b4823cb79599df3780e7f23cce52c19aac31d2a0d47efe33a5e9',
    'msvcp140_atomic_wait.dll': '640b2aefced484d0368eea5bdd06addd0658a3a70a49256e560d6923b404a479',
    'msvcp140_codecvt_ids.dll': 'f2069a52880ec885ee7f0511186100eb7fada0411a2b4948fafea7735b878a18',
    'msvcp140.dll': '0f885b509a685d2bbfa652fed26b5fb31d88fbdab0a978c641d1c7b8aa460aa9',
    'vcruntime140_1.dll': '1f2d41c4aa5db0bc33ebf7b66d72943a817d7ce6cbe880502a9403823633093f',
    'vcruntime140_threads.dll': '219915cf20822f34d5e7c1fdd4e21ae7f3396881096c51036225fb8f84b47afa',
    'vcruntime140.dll': 'd5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066',
}
# relay/actor_voice.py (ActorVoice, audio.cpp binding). Pinned: its synthesize() defaults and native options ARE the
# approved settings, so the child refuses any other copy.
ACTOR_VOICE_SHA256 = '8a97e12544b18776c5da442ec851fcc6d8591f61fc1ae5ad4f483f10ac67f96f'
# relay/resource_budget.py: the child execs exactly these bytes.
RESOURCE_BUDGET_SHA256 = 'f8f068962c45db45bd515b7e833392ae72e1cbfc21042d694e9e148f95d0f819'

SETTINGS = {'backend': 'cuda', 'device': 0, 'threads': 2, 'max_tokens': 350, 'steps': 10, 'guidance': 1.6}
# Fixed inside the pinned actor_voice.py (recorded for receipts; not overridable).
NATIVE_FIXED = {'min_tokens': 2, 'retry_badcase': 'false', 'stream_frames_per_event': 1, 'stream_left_context': 3,
                'weight_type': 'native', 'audiovae_weight_type': 'native', 'mem_saver': 'true', 'prompt_cache_slots': 1}
SEEDS = (42, 7)          # 42 = default take; 7 = the approved retry seed
CLONING_MODE = 'reference+transcript'

# Agent-prepared review fixtures (never a dub reference; refused by hash wherever they are copied to).
REFERENCES = {
    '05e90603b040d02ec690a663ba0d9ebb239cca381e66528469a97589ec43717a': {
        'label': 'viy-two-actor-preview/speaker-001', 'beta_admissible': False},
    '7978d8e49fe5351f931dcc7fd4ca9d97d8b1dd8f339914175c076e394e20100d': {
        'label': 'viy-two-actor-preview/speaker-002', 'beta_admissible': False},
}
# Sem's own voice reference hashes: never a dub reference.
SEM_REFERENCE_PREFIXES = ('1d767951', 'ab7b0ddb')

# Refused reference / child-command locations, by STRUCTURE (path components under concrete roots; never a substring
# match). Defaults below; configure() adds the machine's Sem roots (actor_beta.vox.sem_roots, e.g. Sem's install
# folder) and the allowed movie roots (the relay cache_dir and actor_beta.vox.movie_roots), which may lie under a
# default Sem root. Beta references must also be cut by the pipeline under <cache_dir>/<job_id>/ with program
# provenance (reference_provenance.py); these roots are defence in depth and keep Sem paths off child command lines.
DEFAULT_SEM_ROOTS = ('C:/Sem-Voice', 'C:/Sem-Trials', 'C:/Sem-Data')
SEM_ROOTS = list(DEFAULT_SEM_ROOTS)
ALLOWED_ROOTS = []
AGENT_PREPARED_ROOTS = []


def configure(*, sem_roots=(), allowed_roots=(), agent_prepared_roots=()):
    """Set the machine's roots from config (process-wide; idempotent for the same config)."""
    SEM_ROOTS[:] = list(DEFAULT_SEM_ROOTS) + [str(r) for r in sem_roots or () if r]
    ALLOWED_ROOTS[:] = [str(r) for r in allowed_roots or () if r]
    AGENT_PREPARED_ROOTS[:] = [str(r) for r in agent_prepared_roots or () if r]


def _parts(path):
    return [p.lower() for p in Path(str(path)).resolve().parts]


def _under_root(path, root):
    child, parent = _parts(path), _parts(root)
    return len(child) > len(parent) and child[:len(parent)] == parent


def refused_reference_root(path):
    """'sem' / 'agent-prepared' when `path` lies under a refused root (component-wise), else None. A path under an
    allowed movie root (the cache) is never 'sem'."""
    if any(_under_root(path, r) for r in SEM_ROOTS) and not any(_under_root(path, r) for r in ALLOWED_ROOTS):
        return 'sem'
    if any(_under_root(path, r) for r in AGENT_PREPARED_ROOTS):
        return 'agent-prepared'
    return None


# est. seconds of 48 kHz audio per generated VoxCPM2 token (one 0.16 s patch per PCM event): fallback token-cap guard
# when the child receipt lacks stream_chunks.
SECONDS_PER_TOKEN_EST = 0.16


def reference_label(reference_sha256):
    """Fixture label for a reference hash (None when the voice is not one of the agent-prepared fixtures)."""
    entry = REFERENCES.get(str(reference_sha256 or '').lower())
    return None if entry is None else {'label': entry['label'], 'beta_admissible': entry['beta_admissible']}


def encoder_sample_capacity(capacity_seconds):
    """The exact audiovae_encoder_sample_capacity relay/actor_voice.py derives from max_reference_seconds."""
    return ((int(capacity_seconds * 16000) + 2559) // 2560) * 2560
