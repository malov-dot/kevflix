from __future__ import annotations

import hashlib
import re
import json
import sys
import threading
from pathlib import Path

from .ct2_policy import (CUDA_COMPUTE_TYPE, policy_metadata as ct2_policy_metadata,
                         require_loaded_runtime, require_runtime)
from .movie_faults import (MovieConfigurationError, MovieContentFallback,
                           MovieFault, MovieResourceWait)

ISO = {"eng": "en", "spa": "es", "fra": "fr", "fre": "fr", "deu": "de",
       "ger": "de", "jpn": "ja", "zho": "zh", "chi": "zh", "kor": "ko",
       "ita": "it", "por": "pt", "rus": "ru", "ara": "ar", "hin": "hi",
       "nld": "nl", "dut": "nl", "pol": "pl", "swe": "sv", "tur": "tr"}


def iso(code):
    value = str(code or "").lower().replace("_", "-").split("-")[0]
    return ISO.get(value, value)


def model_device(device='cuda'):
    """Resolve an explicit model device. Never substitute CPU for CUDA."""
    value = str(device).lower()
    if value == 'cpu':
        return 'cpu', 0
    if value == 'cuda':
        return 'cuda', 0
    match = re.fullmatch(r'cuda:(\d+)', value)
    if match:
        return 'cuda', int(match.group(1))
    raise MovieConfigurationError('invalid_model_device',
                                  'Model device must be cuda, cuda:N, or explicit cpu')


def prepare_model_runtime(device='cuda'):
    if model_device(device)[0] == 'cuda':
        try:
            from .gpu_environment import configure_cuda_dll_search
            configure_cuda_dll_search()
        except Exception as error:
            raise MovieConfigurationError('cuda_dll_search_failed', str(error)) from error


def torch_model_device(device='cuda'):
    backend, index = model_device(device)
    prepare_model_runtime(device)
    try:
        import torch
    except ImportError as error:
        raise MovieConfigurationError('torch_runtime_missing', str(error)) from error
    if backend == 'cuda':
        try:
            available = torch.cuda.is_available() and index < torch.cuda.device_count()
        except Exception as error:
            raise MovieConfigurationError('cuda_unavailable', str(error)) from error
        if not available:
            raise MovieConfigurationError('cuda_unavailable',
                                          'The requested CUDA device is unavailable; CPU inference was not substituted')
    return torch.device(backend, index) if backend == 'cuda' else torch.device('cpu')


def backend_fault(stage, error, device='cuda'):
    """Convert backend failures without turning configuration faults into retries."""
    if isinstance(error, MovieFault):
        return error
    message = str(error)
    lower = message.lower()
    # These are the explicit CUDA allocator errors emitted by torch/CT2/ORT.
    # Other model/DLL/driver failures are terminal configuration faults.
    if str(device).startswith('cuda') and (
            'cuda out of memory' in lower or
            'cuda failed with error out of memory' in lower or
            'cuda_error_out_of_memory' in lower or
            ('cuda failure' in lower and 'out of memory' in lower)):
        return MovieResourceWait('gpu_out_of_memory', stage + ': ' + message)
    return MovieConfigurationError(stage + '_failed', message)


def ct2_compute_type(device='cuda'):
    """One reviewed CUDA FP16 policy for Whisper and translation; never auto/INT8.

    CT2 4.6 can advertise INT8 on Blackwell although particular vocabulary
    shapes fail in cuBLAS. Capability membership is only a necessary check.
    """
    backend, index = model_device(device)
    prepare_model_runtime(device)
    try:
        runtime = require_runtime(sys.executable)
        import ctranslate2
        require_loaded_runtime(ctranslate2, runtime)
        if backend == 'cuda' and index >= ctranslate2.get_cuda_device_count():
            raise MovieConfigurationError('cuda_unavailable',
                                          'CTranslate2 cannot use the requested CUDA device')
        supported = ctranslate2.get_supported_compute_types(backend, device_index=index)
    except Exception as error:
        raise backend_fault('ct2_runtime', error, device) from error
    # CPU is explicit offline tooling only, never a fallback for the movie path.
    chosen = CUDA_COMPUTE_TYPE if backend == 'cuda' else 'float32'
    if chosen in supported:
        return chosen
    raise MovieConfigurationError('ct2_compute_type_unavailable',
                                  'Requested device does not support the required %s policy; no alternate precision/device selected' % chosen)


def _ct2_model_state(model, requested, device, stage):
    """Read the native model's actual properties after load, not requested flags."""
    backend, index = model_device(device)
    actual = {'compute_type': str(getattr(model, 'compute_type', '')),
              'device': str(getattr(model, 'device', '')),
              'device_index': getattr(model, 'device_index', None)}
    indices = actual['device_index']
    if isinstance(indices, int):
        indices = [indices]
    elif isinstance(indices, (list, tuple)):
        indices = list(indices)
    else:
        indices = []
    actual['device_index'] = indices
    if actual['compute_type'] != requested or actual['device'] != backend or (backend == 'cuda' and indices != [index]):
        raise MovieConfigurationError('ct2_loaded_policy_mismatch',
                                      '%s loaded %s; required %s on %s' % (stage, actual, requested, device))
    return actual


# ---- Qwen3-8B translator (settings translation_engine == "qwen3") -----------------------------------------------------
# Qwen3-8B (Qwen team, Apache-2.0) as a CUDA-only torch bfloat16 AutoModelForCausalLM. Owner-approved demo settings:
# chat template with enable_thinking=False, greedy decoding (do_sample=False), max_new_tokens=160, the prompt below.
# The M2M100 translator in LanguageTools.english() is untouched and stays the default ('m2m100').
#
# One deliberate difference from the demo script: the demo did `from_pretrained(...).to('cuda:0')`, which first
# materialises all ~16 GB of bf16 weights in host RAM. This module is also loaded inside the lease-owned analysis child,
# whose LeaseGuard aborts the process above 8 GiB working set (relay/demucs_bed.MAX_WORKING_SET), and accelerate (needed
# for device_map) is not installed. load_qwen3() therefore builds the model directly on the GPU and copies each
# safetensors tensor through a plain file read (host RAM peak = the largest single tensor, ~1.2 GB). Weights, dtype,
# tokenizer, prompt and decoding are identical; every checkpoint tensor must match a model tensor or the load refuses.
TRANSLATION_ENGINES = ('m2m100', 'qwen3')
# translation_quantization (Qwen3 only; relay/config.py QUANTIZATION_MODES must list the same names):
#   'none'  the default: every weight stays bf16 exactly as before (16.4 GB resident for Qwen3-8B).
#   'int8'  weight-only INT8, applied to the weights while they are streamed in (same local bf16 safetensors, nothing is
#           downloaded or written). Every decoder-layer Linear (q/k/v/o and gate/up/down) is stored as int8 plus ONE
#           float32 scale per output row (symmetric, scale = max|w| / 127, computed in float32 on the GPU). Embeddings,
#           the final/layer/head norms and lm_head stay bf16 and activations stay bf16. Int8Linear dequantizes one weight
#           matrix per call into a bf16 temporary that is freed again, so a bf16 copy of the model never exists.
TRANSLATION_QUANTIZATIONS = ('none', 'int8')
QWEN3_BASIS = 'qwen3-8b-bf16-to-English'
QWEN3_BASIS_INT8 = 'qwen3-8b-int8-to-English'
QWEN3_PROMPT_TEMPLATE = ('Translate this {language} sentence into natural, accurate English. Keep the meaning exactly, '
                         'including negation and numbers. Output only the English translation.\n\n{text}')
# Owner decision 2026-10-19: a line whose English cannot be spoken inside its slot is translated again, shorter. Only a row
# that carries max_words uses this prompt; every other row keeps the approved prompt above, byte for byte.
QWEN3_CONCISE_TEMPLATE = ('Translate this {language} sentence into natural English that is short enough to be spoken aloud '
                          'quickly: use at most {words} words. Keep the core meaning, including negation and numbers, and '
                          'drop less important detail if needed. Output only the English translation.\n\n{text}')
# Optional lever (actor_beta.fuller_retranslation, default off): a line whose English is far shorter than the slot the original
# speaker occupied is translated again with a MINIMUM word budget. Only a row that carries min_words uses this prompt.
QWEN3_FULLER_TEMPLATE = ('Translate this {language} sentence into natural English using at least {words} words, with fuller '
                         'natural phrasing (complete forms instead of terse shortcuts) so that it takes about as long to say as '
                         'the original. Keep exactly the same meaning, including negation and numbers, and add no new information, '
                         'names or details. Output only the English translation.\n\n{text}')
QWEN3_MAX_WORDS = 60
QWEN3_MAX_NEW_TOKENS = 160
QWEN3_MAX_BATCH = 16                      # rows per left-padded generate() call
QWEN3_MAX_INPUT_CHARS = 2000
QWEN3_MAX_INPUT_TOKENS = 384              # source tokens; the 160-token answer budget cannot carry much more
QWEN3_RUNTIME_HEADROOM_MIB = 1536         # above the weights: largest-tensor load transient + KV cache of a 16-row batch
# int8 mode derives its headroom from the model's own shapes (qwen3_int8_budget) instead of that constant:
QWEN3_PROMPT_OVERHEAD_TOKENS = 96         # chat template + instruction tokens around the source text (about 50 in practice)
INT8_LOAD_CHUNK_ELEMENTS = 8 << 20        # a weight matrix is quantized in row slices of at most this many weights (GPU transient)
INT8_MAX_CODE = 127                       # symmetric int8 codes are clamped to [-127, 127]; -128 is never produced
QWEN3_WEIGHT_BYTES = (15_500_000_000, 17_500_000_000)   # Qwen3-8B bf16 is 16.38 GB; other Qwen3 sizes fall outside
QWEN3_LANGUAGE_NAMES = {
    'ru': 'Russian', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
    'pl': 'Polish', 'uk': 'Ukrainian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic',
    'tr': 'Turkish', 'nl': 'Dutch', 'cs': 'Czech', 'hi': 'Hindi', 'sv': 'Swedish', 'el': 'Greek', 'he': 'Hebrew',
    'fa': 'Persian', 'id': 'Indonesian', 'vi': 'Vietnamese', 'th': 'Thai', 'ro': 'Romanian', 'hu': 'Hungarian',
    'fi': 'Finnish', 'da': 'Danish', 'no': 'Norwegian', 'bg': 'Bulgarian', 'sr': 'Serbian', 'hr': 'Croatian',
    'sk': 'Slovak', 'bn': 'Bengali', 'ca': 'Catalan'}
_QWEN3_ISO3 = {'ces': 'cs', 'cze': 'cs', 'ukr': 'uk', 'ell': 'el', 'gre': 'el', 'heb': 'he', 'fas': 'fa', 'per': 'fa',
               'ind': 'id', 'vie': 'vi', 'tha': 'th', 'ron': 'ro', 'rum': 'ro', 'hun': 'hu', 'fin': 'fi', 'dan': 'da',
               'nor': 'no', 'bul': 'bg', 'srp': 'sr', 'hrv': 'hr', 'slk': 'sk', 'slo': 'sk', 'ben': 'bn', 'cat': 'ca'}
_QUOTE_PAIRS = {'"': '"', "'": "'", '“': '”', '‘': '’', '«': '»', '„': '“',
                '「': '」'}
_SAFETENSORS_DTYPES = {'BF16': 'bfloat16', 'F16': 'float16', 'F32': 'float32'}
_SAFETENSORS_HEADER_LIMIT = 64 << 20


def translation_engine_of(settings):
    """'m2m100' (unset/default) or 'qwen3' from a settings mapping/Config; anything else is a configuration error."""
    settings = getattr(settings, 'data', settings)
    value = settings.get('translation_engine')
    if value in (None, ''):
        return 'm2m100'
    if not isinstance(value, str) or value not in TRANSLATION_ENGINES:
        raise MovieConfigurationError('translation_engine_invalid',
                                      'translation_engine must be one of %s, not %r' % (', '.join(TRANSLATION_ENGINES), value))
    return value


def check_quantization(value):
    """'none' or 'int8' (anything else is a configuration error); the one validator behind every quantization argument."""
    if not isinstance(value, str) or value not in TRANSLATION_QUANTIZATIONS:
        raise MovieConfigurationError('translation_quantization_invalid',
                                      'translation_quantization must be one of %s, not %r'
                                      % (', '.join(TRANSLATION_QUANTIZATIONS), value))
    return value


def translation_quantization_of(settings):
    """'none' (unset/default) or 'int8' from a settings mapping/Config; anything else is a configuration error.
    Only the qwen3 engine reads it; for m2m100 the value is ignored (like a stale translation_llm_model)."""
    settings = getattr(settings, 'data', settings)
    value = settings.get('translation_quantization')
    if value in (None, ''):
        return 'none'
    return check_quantization(value)


def qwen3_basis(quantization='none'):
    """The 'basis' string returned with every Qwen3 translation: names the weights that produced it."""
    return QWEN3_BASIS_INT8 if check_quantization(quantization) == 'int8' else QWEN3_BASIS


def qwen3_language_name(code):
    """English language name for the prompt: from the ISO code, else the (normalised) code itself."""
    value = iso(code)
    value = _QWEN3_ISO3.get(value, value)
    return QWEN3_LANGUAGE_NAMES.get(value, value)


def row_max_words(value):
    """A usable word budget (int 1..QWEN3_MAX_WORDS) or None."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= QWEN3_MAX_WORDS:
        return None
    return value


def qwen3_prompt(text, language, max_words=None, min_words=None):
    """The exact user message of the approved demo prompt (max_words: the concise variant for one over-long line; min_words, when
    there is no max_words: the fuller variant for one far-too-short line)."""
    if row_max_words(max_words) is not None:
        return QWEN3_CONCISE_TEMPLATE.format(language=qwen3_language_name(language), words=max_words, text=text)
    if row_max_words(min_words) is not None:
        return QWEN3_FULLER_TEMPLATE.format(language=qwen3_language_name(language), words=min_words, text=text)
    return QWEN3_PROMPT_TEMPLATE.format(language=qwen3_language_name(language), text=text)


def qwen3_policy(quantization='none'):
    """Everything that decides what a cached translation means; hashed into the analysis cache namespace.

    quantization 'none' returns exactly the policy that existed before the int8 mode (so bf16 cache entries and engine
    identities are unchanged); 'int8' changes basis and precision and adds the quantization scheme, so int8 and bf16
    translations can never share a cache namespace."""
    policy = {'engine': 'qwen3', 'basis': QWEN3_BASIS, 'precision': 'torch.bfloat16', 'device': 'cuda only, no CPU fallback',
              'prompt': QWEN3_PROMPT_TEMPLATE, 'language_names': dict(sorted(QWEN3_LANGUAGE_NAMES.items())),
              'chat_template': 'tokenizer.apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=False)',
              'decoding': {'do_sample': False, 'max_new_tokens': QWEN3_MAX_NEW_TOKENS},
              'batching': 'left padded, at most %d rows per generate(); rows are independent' % QWEN3_MAX_BATCH,
              'limits': {'input_chars': QWEN3_MAX_INPUT_CHARS, 'input_tokens': QWEN3_MAX_INPUT_TOKENS},
              'output': 'surrounding quote pair and <think> block removed; empty or truncated output is refused'}
    if check_quantization(quantization) == 'int8':
        policy.update(
            basis=QWEN3_BASIS_INT8, precision='torch.bfloat16 activations, int8 weight-only decoder linears',
            quantization={'mode': 'int8',
                          'scheme': 'symmetric per output row, scale = max|w|/%d in float32, round half to even, '
                                    'codes clamped to +-%d' % (INT8_MAX_CODE, INT8_MAX_CODE),
                          'quantized': 'decoder-layer q/k/v/o and gate/up/down projections',
                          'kept_bf16': 'embeddings, all norms, lm_head',
                          'dequantization': 'per call into a bf16 temporary of one weight matrix (weights * scale)'})
    return policy


def qwen3_policy_text(quantization='none'):
    """One cache-policy string: names the engine and pins the full policy by hash."""
    digest = hashlib.sha256(json.dumps(qwen3_policy(quantization), sort_keys=True).encode('utf-8')).hexdigest()
    label = 'int8-weights' if quantization == 'int8' else 'bf16'
    return 'qwen3-8b %s greedy max_new_tokens=%d to English; policy sha256=%s' % (label, QWEN3_MAX_NEW_TOKENS, digest)


def qwen3_linear_shapes(config):
    """[(name, out_features, in_features)] of the seven decoder-layer Linear layers, from a Qwen3 config.json mapping."""
    hidden, inter = int(config['hidden_size']), int(config['intermediate_size'])
    heads = int(config['num_attention_heads'])
    kv_heads = int(config.get('num_key_value_heads') or heads)
    head_dim = int(config.get('head_dim') or hidden // heads)
    if min(hidden, inter, heads, kv_heads, head_dim) <= 0:
        raise ValueError('non-positive Qwen3 dimension')
    q_out, kv_out = heads * head_dim, kv_heads * head_dim
    return [('q_proj', q_out, hidden), ('k_proj', kv_out, hidden), ('v_proj', kv_out, hidden), ('o_proj', hidden, q_out),
            ('gate_proj', inter, hidden), ('up_proj', inter, hidden), ('down_proj', hidden, inter)]


def qwen3_int8_budget(config, *, batch=QWEN3_MAX_BATCH, input_tokens=QWEN3_MAX_INPUT_TOKENS,
                      new_tokens=QWEN3_MAX_NEW_TOKENS, overhead_tokens=QWEN3_PROMPT_OVERHEAD_TOKENS):
    """Exact resident size and computed runtime headroom (bytes and MiB) of the int8 translator. No weights are read.

    resident_weights_bytes is exact (it equals resident_weight_bytes(model) of the loaded model), from the config.json
    shapes (L layers, hidden H, intermediate I, vocabulary V, default rotary embedding):
        int8 linears  L * sum(out*in over q,k,v,o,gate,up,down)         1 byte each
        scales        L * sum(out)                                      4 bytes each (float32, one per output row)
        scales (bf16) the same rows again, 2 bytes each (activation-dtype copy that saves a kernel launch per call)
        embedding     V*H*2 bytes (bf16);  lm_head another V*H*2 unless tie_word_embeddings
        norms         (L*(2H + 2*head_dim) + H) * 2 bytes (bf16);  attention biases 2 bytes each when attention_bias
        rotary        (head_dim // 2) * 4 bytes (float32 inverse frequencies)
    headroom_bytes is the larger of two bounds on top of the weights:
        runtime  max(KV cache of the full batch after generation,
                     KV cache of the prompts + prefill activations) + one dequantized weight matrix + the logits, with
                     KV bytes/token = 2 * L * kv_heads * head_dim * 2,  prompt tokens = input_tokens + overhead_tokens,
                     batch rows = batch, prefill activations = batch * prompt tokens * (4*I + 4*H) * 2 bytes
        load     INT8_LOAD_CHUNK_ELEMENTS * (2 + 4 + 4) bytes (bf16 slice + float32 slice + float32 quotient)
    The batch bound assumes every row of a 16-row batch is as long as the limits allow, so it is a ceiling, not the
    typical need. footprint_mib = ceil(resident/MiB) + ceil(headroom/MiB) is what the admission check compares."""
    mib = 1 << 20
    hidden, inter, layers, vocab = (int(config[key]) for key in ('hidden_size', 'intermediate_size',
                                                                  'num_hidden_layers', 'vocab_size'))
    heads = int(config['num_attention_heads'])
    kv_heads = int(config.get('num_key_value_heads') or heads)
    head_dim = int(config.get('head_dim') or hidden // heads)
    shapes = qwen3_linear_shapes(config)
    if min(layers, vocab) <= 0:
        raise ValueError('non-positive Qwen3 dimension')
    tied, biased = bool(config.get('tie_word_embeddings', False)), bool(config.get('attention_bias', False))
    scale_rows = layers * sum(rows for _, rows, _ in shapes)
    parts = {
        'int8_weight_bytes': layers * sum(rows * columns for _, rows, columns in shapes),
        'scale_bytes': scale_rows * 4,
        'scale_activation_bytes': scale_rows * 2,
        'embedding_bytes': vocab * hidden * 2,
        'lm_head_bytes': 0 if tied else vocab * hidden * 2,
        'norm_bytes': (layers * (2 * hidden + 2 * head_dim) + hidden) * 2,
        'bias_bytes': layers * (heads * head_dim + 2 * kv_heads * head_dim + hidden) * 2 if biased else 0,
        'rotary_bytes': (head_dim // 2) * 4}
    resident = sum(parts.values())
    kv_token = 2 * layers * kv_heads * head_dim * 2
    prompt_tokens = input_tokens + overhead_tokens
    kv_prompts = batch * prompt_tokens * kv_token
    kv_full = batch * (prompt_tokens + new_tokens) * kv_token
    activations = batch * prompt_tokens * (4 * inter + 4 * hidden) * 2
    temporary = max(rows * columns for _, rows, columns in shapes) * 2
    logits = batch * vocab * 4 * 2
    runtime = max(kv_full, kv_prompts + activations) + temporary + logits
    load = INT8_LOAD_CHUNK_ELEMENTS * (2 + 4 + 4)
    headroom = max(runtime, load)
    resident_mib, headroom_mib = -(-resident // mib), -(-headroom // mib)
    return {**parts, 'resident_weights_bytes': resident, 'kv_bytes_per_token': kv_token,
            'kv_cache_full_batch_bytes': kv_full, 'kv_cache_prompts_bytes': kv_prompts,
            'prefill_activation_bytes': activations, 'dequantized_matrix_bytes': temporary, 'logits_bytes': logits,
            'runtime_peak_bytes': runtime, 'load_transient_bytes': load, 'headroom_bytes': headroom,
            'resident_mib': resident_mib, 'headroom_mib': headroom_mib, 'footprint_mib': resident_mib + headroom_mib,
            'batch_rows': batch, 'prompt_tokens': prompt_tokens, 'new_tokens': new_tokens}


def qwen3_model_report(model_dir, quantization='none'):
    """Static check of a user-supplied Qwen3-8B directory. Imports no model library, never reads weights.

    Returns {'ok', 'problems', 'path', 'shard_files', 'weights_bytes', 'footprint_mib', 'quantization'} (plus
    'resident_weights_bytes' once the weights are known and, for int8, 'budget' = qwen3_int8_budget()). footprint_mib is
    the VRAM this translator needs on its torch allocator: bf16 ('none') weights + QWEN3_RUNTIME_HEADROOM_MIB, None
    when the weights are unknown; int8 the exact int8 layout plus the computed headroom of qwen3_int8_budget()."""
    quantization = check_quantization(quantization)
    text = str(model_dir or '').strip()
    problems, shard_files = [], []
    report = {'ok': False, 'problems': problems, 'path': text, 'shard_files': shard_files, 'weights_bytes': 0,
              'footprint_mib': None, 'quantization': quantization}
    if not text:
        problems.append('translation_llm_model is not set')
        return report
    path = Path(text)
    if not path.is_dir():
        problems.append('%s is not an existing directory' % path)
        return report
    try:
        metadata = json.loads((path / 'config.json').read_text(encoding='utf-8-sig'))
        if not isinstance(metadata, dict):
            raise ValueError('not a JSON object')
    except (OSError, ValueError) as error:
        metadata = None
        problems.append('config.json is missing or unreadable (%s)' % error)
    if metadata is not None and (metadata.get('model_type') != 'qwen3'
                                 or 'Qwen3ForCausalLM' not in (metadata.get('architectures') or ())):
        problems.append('config.json is not a Qwen3ForCausalLM model (model_type %r)' % (metadata.get('model_type'),))
    if not (path / 'generation_config.json').is_file():
        problems.append('generation_config.json is missing (it carries the stop tokens)')
    if not ((path / 'tokenizer.json').is_file() or ((path / 'vocab.json').is_file() and (path / 'merges.txt').is_file())):
        problems.append('tokenizer files are missing (tokenizer.json, or vocab.json and merges.txt)')
    index, single = path / 'model.safetensors.index.json', path / 'model.safetensors'
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding='utf-8-sig')).get('weight_map')
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError('weight_map is missing')
            names = sorted({str(value) for value in weight_map.values()})
            if any(Path(name).name != name or not name.endswith('.safetensors') for name in names):
                raise ValueError('weight_map names a file outside the model folder')
            shard_files.extend(names)
        except (OSError, ValueError, AttributeError) as error:
            problems.append('model.safetensors.index.json is unreadable (%s)' % error)
    elif single.is_file():
        shard_files.append(single.name)
    else:
        problems.append('no model.safetensors.index.json and no model.safetensors (safetensors weights are required)')
    total = 0
    for name in shard_files:
        try:
            size = (path / name).stat().st_size
        except OSError:
            problems.append('weight file %s is missing' % name)
            continue
        if size <= 0:
            problems.append('weight file %s is empty' % name)
        total += size
    report['weights_bytes'] = total
    if shard_files and total and not QWEN3_WEIGHT_BYTES[0] <= total <= QWEN3_WEIGHT_BYTES[1]:
        problems.append('weights total %.2f GB; Qwen3-8B bf16 is about 16.4 GB' % (total / 1e9))
    if total:
        report['footprint_mib'] = -(-total // (1 << 20)) + QWEN3_RUNTIME_HEADROOM_MIB
    if quantization == 'int8':
        # The checkpoint on disk is the same bf16 data (the size window above still applies), but the GPU holds the
        # int8 layout: its exact size and headroom come from the config.json shapes, not from the file sizes.
        report['footprint_mib'] = None
        if metadata is not None:
            try:
                budget = qwen3_int8_budget(metadata)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                problems.append('config.json lacks the usable Qwen3 shapes for the int8 VRAM estimate (%s: %s)'
                                % (type(error).__name__, error))
            else:
                report['budget'] = budget
                report['resident_weights_bytes'] = budget['resident_weights_bytes']
                report['footprint_mib'] = budget['footprint_mib']
    elif total:
        report['resident_weights_bytes'] = total          # bf16: the GPU holds the checkpoint bytes as they are
    report['ok'] = not problems
    return report


def clean_translation(value):
    """Model output -> English text: drop a <think> block, whitespace and one surrounding quote pair."""
    text = re.sub(r'<think>.*?</think>', '', str(value), flags=re.DOTALL).strip()
    for _ in range(2):
        if len(text) >= 2 and text[0] in _QUOTE_PAIRS and text[-1] == _QUOTE_PAIRS[text[0]]:
            inner = text[1:-1]
            if text[0] not in inner and _QUOTE_PAIRS[text[0]] not in inner:
                text = inner.strip()
                continue
        break
    return text


def _safetensors_entries(path):
    """(data start, [(offset, name, dtype, shape, nbytes)]) from the file's own header, bounds-checked."""
    size = Path(path).stat().st_size
    with open(path, 'rb') as handle:
        raw = handle.read(8)
        length = int.from_bytes(raw, 'little') if len(raw) == 8 else 0
        if not 0 < length <= _SAFETENSORS_HEADER_LIMIT or 8 + length > size:
            raise MovieConfigurationError('translation_llm_weights_invalid', '%s has no valid safetensors header' % path)
        header = json.loads(handle.read(length).decode('utf-8'))
    base, entries = 8 + length, []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        begin, end = info['data_offsets']
        if not 0 <= begin <= end or base + end > size:
            raise MovieConfigurationError('translation_llm_weights_invalid',
                                          '%s: tensor %s lies outside the file' % (path, name))
        entries.append((begin, name, info['dtype'], [int(x) for x in info['shape']], end - begin))
    entries.sort()
    return base, entries


def stream_safetensors(model, files, torch, *, tied_head=False, quantized=None):
    """Copy every tensor of the safetensors files into the same-named model tensor, one tensor at a time, through
    plain file reads (no mmap, no whole-shard host copy). Refuses unexpected, mismatched or missing tensors.

    quantized: {checkpoint tensor name: Int8Linear} (int8 mode only). A checkpoint weight with such a name is not
    copied: the module quantizes it into its int8 weight and float32 row scales, a slice of rows at a time. The module's
    own buffers (weight_q, scale) are internal and are not checkpoint tensors."""
    quantized = quantized or {}
    state = model.state_dict()                        # shares storage with the parameters: copy_ updates the model
    internal = {prefix + suffix for prefix in (name[:-len('weight')] for name in quantized)
                for suffix in ('weight_q', 'scale')}
    for key in internal:
        state.pop(key, None)
    loaded = set()
    with torch.no_grad():
        for path in files:
            base, entries = _safetensors_entries(path)
            with open(path, 'rb') as handle:
                for offset, name, dtype_name, shape, nbytes in entries:
                    module = quantized.get(name)
                    target = module.weight_q if module is not None else state.get(name)
                    dtype = _SAFETENSORS_DTYPES.get(dtype_name)
                    if target is None or dtype is None:
                        raise MovieConfigurationError('translation_llm_weights_unexpected',
                                                      '%s: tensor %s (%s) does not belong to this model'
                                                      % (Path(path).name, name, dtype_name))
                    dtype = getattr(torch, dtype)
                    count = 1
                    for extent in shape:
                        count *= extent
                    if list(target.shape) != shape or count * torch.empty((), dtype=dtype).element_size() != nbytes:
                        raise MovieConfigurationError('translation_llm_weights_mismatch',
                                                      '%s: tensor %s is %s, the model needs %s'
                                                      % (Path(path).name, name, shape, list(target.shape)))
                    if nbytes:
                        buffer = bytearray(nbytes)
                        handle.seek(base + offset)
                        if handle.readinto(buffer) != nbytes:
                            raise MovieConfigurationError('translation_llm_weights_invalid',
                                                          '%s: tensor %s is truncated' % (Path(path).name, name))
                        host = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
                        if module is not None:
                            try:
                                module.load_weight(host)
                            except ValueError as error:
                                raise MovieConfigurationError('translation_llm_weights_invalid', '%s: tensor %s: %s'
                                                              % (Path(path).name, name, error)) from error
                        else:
                            target.copy_(host)
                        del host, buffer
                    loaded.add(name)
    missing = sorted((set(state) | set(quantized)) - loaded - ({'lm_head.weight'} if tied_head else set()))
    if missing:
        raise MovieConfigurationError('translation_llm_weights_incomplete',
                                      '%d model tensors have no checkpoint weights, e.g. %s' % (len(missing), ', '.join(missing[:3])))
    return len(loaded)


def int8_quantize_rows(weight):
    """Symmetric per-output-row INT8 quantization of a [rows, columns] float tensor -> (int8 codes, float32 scales).

    scale[r] = max|w[r, :]| / 127 in float32 (1.0 for an all-zero row, whose codes are all 0); code = round-half-to-even
    (w / scale) clamped to [-127, 127]; the weight is recovered as code * scale. The error per weight is at most
    scale / 2. max|w| is taken as max(rowmax, -rowmin) so no |w| temporary is made; the input is never modified.
    Non-finite weights are refused (ValueError)."""
    import torch
    if weight.dim() != 2:
        raise ValueError('expected a 2-D weight, got %s' % (tuple(weight.shape),))
    wide = weight.float()                              # float32 working copy (the same tensor when already float32)
    peak = torch.maximum(wide.amax(dim=1), -wide.amin(dim=1))
    if not bool(torch.isfinite(peak).all()):
        raise ValueError('the weight holds NaN or infinite values')
    scale = torch.where(peak > 0, peak / INT8_MAX_CODE, torch.ones_like(peak))
    codes = (wide / scale.unsqueeze(1)).round_().clamp_(-INT8_MAX_CODE, INT8_MAX_CODE).to(torch.int8)
    return codes, scale


_INT8_LINEAR = None
_INT8_LOCK = threading.Lock()


def _int8_linear_class():
    """The Int8Linear nn.Module, defined on first use: importing relay.language (preflight, provisioning) never imports torch."""
    global _INT8_LINEAR
    with _INT8_LOCK:
        if _INT8_LINEAR is None:
            import torch
            import torch.nn.functional as functional

            class Int8Linear(torch.nn.Module):
                """Weight-only INT8 stand-in for nn.Linear: int8 codes [out, in] + one float32 scale per output row.

                The buffers weight_q / scale are the only copy of the weight. forward() rebuilds the bf16 weight of THIS
                layer for the duration of one call, F.linear(x, weight_q * scale_act[:, None]), and frees it again, so the
                temporary is one matrix and a dequantized copy of the model never exists. The product is ONE fused
                kernel (int8 * activation-dtype scale: computed in float32, rounded once into the activation dtype; a 7-bit
                code times a bf16 scale is exact in float32). scale_act is the float32 scale converted once to the
                activation dtype (non-persistent, made by prepare()) so the hot path launches no conversion kernel; the
                conversion costs at most 2^-9 relative per row, below the quantization step itself."""

                def __init__(self, in_features, out_features, bias=False, device=None):
                    super().__init__()
                    self.in_features, self.out_features = int(in_features), int(out_features)
                    self.register_buffer('weight_q', torch.empty((self.out_features, self.in_features),
                                                                 dtype=torch.int8, device=device))
                    self.register_buffer('scale', torch.empty(self.out_features, dtype=torch.float32, device=device))
                    self.register_buffer('scale_act', None, persistent=False)
                    if bias:
                        self.bias = torch.nn.Parameter(torch.empty(self.out_features, dtype=torch.bfloat16, device=device),
                                                       requires_grad=False)
                    else:
                        self.register_parameter('bias', None)

                def load_weight(self, weight):
                    """Quantize a [out, in] float tensor (any float dtype/device) into the buffers, a row slice at a time."""
                    if tuple(weight.shape) != tuple(self.weight_q.shape):
                        raise ValueError('weight is %s, the layer needs %s' % (tuple(weight.shape), tuple(self.weight_q.shape)))
                    step = max(1, INT8_LOAD_CHUNK_ELEMENTS // max(1, self.in_features))
                    with torch.no_grad():
                        for start in range(0, self.out_features, step):
                            part = weight[start:start + step].to(self.weight_q.device)
                            codes, scale = int8_quantize_rows(part)
                            self.weight_q[start:start + step].copy_(codes)
                            self.scale[start:start + step].copy_(scale)
                            del part, codes, scale

                def prepare(self, dtype):
                    """Keep the scales in the activation dtype as well (rows * 2 bytes) after the weights are loaded."""
                    self.scale_act = self.scale if dtype == self.scale.dtype else self.scale.to(dtype)

                def dequantized_weight(self, dtype):
                    """code * scale rounded once into dtype: exactly what forward() builds per call."""
                    scale = self.scale_act if (self.scale_act is not None and self.scale_act.dtype == dtype) \
                        else self.scale.to(dtype)
                    return self.weight_q * scale.unsqueeze(1)

                def forward(self, x):
                    return functional.linear(x, self.dequantized_weight(x.dtype), self.bias)

                def extra_repr(self):
                    return 'in_features=%d, out_features=%d, int8 weight-only, bias=%s' % (
                        self.in_features, self.out_features, self.bias is not None)

            Int8Linear.__module__, Int8Linear.__qualname__ = __name__, 'Int8Linear'
            _INT8_LINEAR = Int8Linear
    return _INT8_LINEAR


def __getattr__(name):                                  # PEP 562: language.Int8Linear without a top-level torch import
    if name == 'Int8Linear':
        return _int8_linear_class()
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def quantize_decoder_linears(model, torch):
    """Replace every nn.Linear of the decoder layers by an (uninitialised, same device as the layer) Int8Linear.

    Returns {checkpoint tensor name: Int8Linear}. Only model.model.layers is touched: embeddings, norms and lm_head are
    never quantized (a tied lm_head is the embedding table)."""
    Int8Linear = _int8_linear_class()
    layers = model.model.layers
    found = [(name, module) for name, module in layers.named_modules() if isinstance(module, torch.nn.Linear)]
    quantized = {}
    for name, module in found:
        replacement = Int8Linear(module.in_features, module.out_features, bias=module.bias is not None,
                                 device=module.weight.device)
        parent_name, _, child = name.rpartition('.')
        setattr(layers.get_submodule(parent_name) if parent_name else layers, child, replacement)
        quantized['model.layers.%s.weight' % name] = replacement
    expected = len(layers) * 7
    if len(quantized) != expected:
        raise MovieConfigurationError('translation_llm_layout_unsupported',
                                      'expected %d decoder Linear layers (q,k,v,o,gate,up,down) to quantize, found %d'
                                      % (expected, len(quantized)))
    return quantized


def resident_weight_bytes(model):
    """Bytes of every parameter and buffer of the model (shared tensors once): what the weights occupy on their device."""
    return sum(tensor.numel() * tensor.element_size() for tensor in list(model.parameters()) + list(model.buffers()))


def _build_qwen3_model_int8(path, device, torch, files):
    """The int8 twin of build_qwen3_model: the decoder Linears exist only as int8 + scales, never as bf16.

    The architecture is created on the meta device (no memory), the decoder Linears are swapped for Int8Linear, and only
    then is real storage allocated on `device` (to_empty): embeddings, norms and lm_head in bf16, the int8 codes and float32
    scales. No bf16 copy of a quantized weight is ever allocated, so the load peak is the final size plus one row slice."""
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    quantized = quantize_decoder_linears(model, torch)
    model.to_empty(device=device)                      # uninitialised storage: every tensor is filled or refused below
    tied = bool(getattr(config, 'tie_word_embeddings', False))
    if tied:
        model.tie_weights()                            # to_empty() gave lm_head its own table; share the embedding again
    rotary = getattr(model.model, 'rotary_emb', None)
    if rotary is not None:                             # its non-persistent inv_freq was garbage after to_empty()
        model.model.rotary_emb = type(rotary)(config=config, device=device)
    stream_safetensors(model, files, torch, tied_head=tied, quantized=quantized)
    for module in quantized.values():
        module.prepare(torch.bfloat16)
    model.generation_config = GenerationConfig.from_pretrained(str(path), local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    if any(tensor.is_meta for tensor in list(model.parameters()) + list(model.buffers())):
        raise MovieConfigurationError('translation_llm_device_mismatch', 'Qwen3 int8 model still has meta tensors')
    odd = sorted({str(p.dtype) for p in model.parameters()} - {'torch.bfloat16'})
    if odd:
        raise MovieConfigurationError('translation_llm_precision_mismatch',
                                      'Qwen3 parameters are %s; torch.bfloat16 is required (the quantized decoder '
                                      'weights are int8 buffers)' % ', '.join(odd))
    if any(m.weight_q.dtype != torch.int8 or m.scale.dtype != torch.float32 for m in quantized.values()) or any(
            isinstance(m, torch.nn.Linear) for m in model.model.layers.modules()):
        raise MovieConfigurationError('translation_llm_precision_mismatch',
                                      'Qwen3 int8 layout check failed: a decoder Linear is not int8 weight-only')
    return model


def build_qwen3_model(path, device, torch, files, quantization='none'):
    """Qwen3ForCausalLM in bfloat16 built on `device` and filled from `files`; eval mode, no gradients.

    quantization 'int8' builds the weight-only int8 variant instead (_build_qwen3_model_int8); 'none' is unchanged."""
    if check_quantization(quantization) == 'int8':
        return _build_qwen3_model_int8(path, device, torch, files)
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    tied = bool(getattr(config, 'tie_word_embeddings', False))
    stream_safetensors(model, files, torch, tied_head=tied)
    if tied:
        model.tie_weights()
    model.generation_config = GenerationConfig.from_pretrained(str(path), local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    odd = sorted({str(p.dtype) for p in model.parameters()} - {'torch.bfloat16'})
    if odd:
        raise MovieConfigurationError('translation_llm_precision_mismatch',
                                      'Qwen3 parameters are %s; torch.bfloat16 is required' % ', '.join(odd))
    return model


def load_qwen3(model_dir, device_name, quantization='none'):
    """Production loader: (tokenizer, model) on the requested CUDA device. Never CPU.

    quantization 'none' (default): the bf16 model exactly as before. 'int8': weight-only INT8 decoder linears (see
    TRANSLATION_QUANTIZATIONS); the allocator's cached blocks of the load are returned to the driver afterwards so the
    process holds only the weights."""
    quantization = check_quantization(quantization)
    report = qwen3_model_report(model_dir, quantization)
    if not report['ok']:
        raise MovieConfigurationError('translation_llm_assets_invalid',
                                      'Qwen3 model folder %s: %s' % (model_dir, '; '.join(report['problems'])))
    target = torch_model_device(device_name)
    if target.type != 'cuda':
        raise MovieConfigurationError('translation_qwen3_cuda_only', 'Qwen3 translation runs on CUDA only')
    import torch
    from transformers import AutoTokenizer
    path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    files = [path / name for name in report['shard_files']]
    model = (build_qwen3_model(path, target, torch, files, 'int8') if quantization == 'int8'
             else build_qwen3_model(path, target, torch, files))
    if quantization == 'int8':
        torch.cuda.empty_cache()                       # the free-VRAM floor sees reserved-but-unused allocator blocks too
    if next(model.parameters()).device != target:
        raise MovieConfigurationError('translation_llm_device_mismatch', 'Qwen3 loaded on %s, not %s'
                                      % (next(model.parameters()).device, target))
    return tokenizer, model


def _eos_ids(model, tokenizer):
    found = set()
    generation = getattr(model, 'generation_config', None)
    for value in (getattr(generation, 'eos_token_id', None), getattr(tokenizer, 'eos_token_id', None)):
        for item in (value if isinstance(value, (list, tuple, set)) else [value]):
            if isinstance(item, int) and not isinstance(item, bool):
                found.add(item)
    return found


class Qwen3Translator:
    """Qwen3-8B bf16 -> English on one CUDA device. Lazy load once, then thread-safe: one lock serialises the GPU.

    translate_batch() returns, for every input row, either (english, QWEN3_BASIS) or a MovieFault instance (returned,
    not raised) for that row alone. Only a failure to load the model (assets, CUDA, memory) raises for the whole call.
    `loader` replaces load_qwen3 (tests inject fake tokenizer/model objects). `quantization` ('none' | 'int8') is what the
    production loader builds and names the basis returned with every translation (qwen3_basis); an injected loader owns
    what it builds."""

    def __init__(self, model_dir, device='cuda:0', *, loader=None, quantization='none'):
        backend, index = model_device(device)
        if backend != 'cuda':
            raise MovieConfigurationError('translation_qwen3_cuda_only',
                                          'Qwen3 translation runs on CUDA only; CPU inference is not substituted')
        self.model_dir, self.device_name = str(model_dir), 'cuda:%d' % index
        self.quantization = check_quantization(quantization)
        self._loader = loader
        self._lock = threading.RLock()
        self.tokenizer = self.model = self.state = None
        self.eos_ids = set()

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            try:
                if self._loader is not None:
                    tokenizer, model = self._loader(self.model_dir, self.device_name)
                elif self.quantization == 'none':
                    tokenizer, model = load_qwen3(self.model_dir, self.device_name)
                else:
                    tokenizer, model = load_qwen3(self.model_dir, self.device_name, self.quantization)
                tokenizer.padding_side = 'left'          # batched decoder-only generation
                if getattr(tokenizer, 'pad_token', True) is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as error:
                self._release_cache()
                raise backend_fault('translation_load', error, self.device_name) from error
            self.tokenizer, self.model, self.eos_ids = tokenizer, model, _eos_ids(model, tokenizer)
            try:
                parameter = next(model.parameters())
                self.state = {'backend': 'torch', 'compute_type': str(parameter.dtype), 'device': str(parameter.device)}
            except Exception:
                self.state = {'backend': 'torch', 'compute_type': None, 'device': None}
            if self.quantization != 'none':              # bf16 states keep their original shape
                self.state['quantization'] = self.quantization
                self.state['weights'] = ('int8 weight-only decoder linears with float32 row scales; embeddings, norms and '
                                         'lm_head bf16; compute_type is the activation dtype')
                try:
                    self.state['resident_weight_bytes'] = resident_weight_bytes(model)
                except Exception:
                    self.state['resident_weight_bytes'] = None

    @staticmethod
    def _precheck(text, code):
        if code == 'en':
            return (text, 'already-English')
        if not code or code in ('und', 'unknown'):
            return MovieContentFallback('source_language_unresolved', 'The source language has not been resolved')
        if not re.fullmatch(r'[a-z]{2,3}', code):        # the code is placed in the prompt: letters only
            return MovieContentFallback('source_language_unsupported',
                                        'Local translation does not accept source language %r' % (code,))
        if not text.strip():
            return MovieContentFallback('translation_input_empty', 'There is no source text to translate')
        if len(text) > QWEN3_MAX_INPUT_CHARS:
            return MovieContentFallback('translation_input_too_long', 'Dialogue is too long for one translation')
        return None

    def translate_batch(self, rows):
        """rows: [(text, iso_language_code[, max_words[, min_words]])] -> aligned list of (english, basis) | MovieFault."""
        results, work = [None] * len(rows), {}
        for index, row in enumerate(rows):
            text, code = str(row[0]), iso(row[1])
            words = row_max_words(row[2]) if len(row) > 2 else None
            fuller = row_max_words(row[3]) if len(row) > 3 else None       # same 1..60 range as a maximum
            early = self._precheck(text, code)
            if early is not None:
                results[index] = early
            else:
                work.setdefault((text, code, words, fuller), []).append(index)
        if not work:
            return results
        with self._lock:
            self.load()
            sized = []
            for (text, code, words, fuller), indices in work.items():
                try:
                    tokens = len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
                except Exception as error:
                    self._settle(results, indices, backend_fault('translation', error, self.device_name))
                    continue
                if tokens > QWEN3_MAX_INPUT_TOKENS:
                    self._settle(results, indices, MovieContentFallback('translation_input_too_long',
                                                                        'Dialogue is too long for one translation'))
                else:
                    sized.append((tokens, text, code, words, fuller, indices))
            sized.sort(key=lambda item: item[0])         # similar lengths share a batch: less padding
            for start in range(0, len(sized), QWEN3_MAX_BATCH):
                chunk = sized[start:start + QWEN3_MAX_BATCH]
                outcomes = self._run([(text, code, words, fuller) for _, text, code, words, fuller, _ in chunk])
                for (_, _, _, _, _, indices), outcome in zip(chunk, outcomes):
                    self._settle(results, indices, outcome)
        return results

    @staticmethod
    def _settle(results, indices, outcome):
        for index in indices:
            results[index] = outcome

    def _run(self, items):
        """Translate one batch; a batch-level failure is retried row by row so only the failing row reports it."""
        try:
            return self._generate(items)
        except Exception as error:
            self._release_cache()
            if len(items) == 1:
                return [backend_fault('translation', error, self.device_name)]
        outcomes = []
        for item in items:
            outcomes.extend(self._run([item]))
        return outcomes

    def _generate(self, items):
        import torch
        tokenizer, model = self.tokenizer, self.model
        prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': qwen3_prompt(text, code, words, *rest)}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
                   for text, code, words, *rest in items]          # rest: the optional minimum word budget (fuller variant)
        encoded = tokenizer(prompts, return_tensors='pt', padding=True).to(self.device_name)
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=QWEN3_MAX_NEW_TOKENS, do_sample=False)
        new = generated[:, encoded['input_ids'].shape[-1]:]
        texts, identifiers = tokenizer.batch_decode(new, skip_special_tokens=True), new.tolist()
        outcomes = []
        for text, ids in zip(texts, identifiers):
            if self.eos_ids and len(ids) >= QWEN3_MAX_NEW_TOKENS and not self.eos_ids.intersection(ids):
                outcomes.append(MovieContentFallback('translation_truncated',
                                                     'The translation did not finish within %d tokens' % QWEN3_MAX_NEW_TOKENS))
                continue
            english = clean_translation(text)
            outcomes.append((english, qwen3_basis(self.quantization)) if english else
                            MovieContentFallback('translation_empty', 'Translation returned no English text'))
        return outcomes

    @staticmethod
    def _release_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _batch_row(row):
    """(text, language) from a {'text','language'} mapping or a 2-sequence; MovieContentFallback when malformed."""
    if isinstance(row, dict):
        text, language = row.get('text'), row.get('language')
    elif isinstance(row, (list, tuple)) and len(row) in (2, 3):
        text, language = row[0], row[1]
    else:
        raise MovieContentFallback('translation_row_invalid', 'A translation row needs text and language')
    if not isinstance(text, str):
        raise MovieContentFallback('translation_row_invalid', 'A translation row needs string text')
    return text, language


def _batch_max_words(row):
    """The optional word budget of a batch row ({'max_words'} or a third item), validated; None when absent or unusable."""
    if isinstance(row, dict):
        return row_max_words(row.get('max_words'))
    return row_max_words(row[2]) if isinstance(row, (list, tuple)) and len(row) == 3 else None


def _batch_min_words(row):
    """The optional MINIMUM word budget of a batch row ({'min_words'} only; fuller retranslation), validated; None otherwise."""
    return row_max_words(row.get('min_words')) if isinstance(row, dict) else None


def _qwen3_row(text, language, words, fuller):
    """One Qwen3 row: the 2- or 3-item row of before, plus the minimum budget as a fourth item only when there is one."""
    if fuller:
        return (text, language, words or None, fuller)
    return (text, language) + ((words,) if words else ())


class LanguageTools:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device, self.device_index = model_device(device)
        self.device_name = ('cuda:' + str(self.device_index)) if self.device == 'cuda' else 'cpu'
        prepare_model_runtime(self.device_name)
        self.ct2_runtime = require_runtime(sys.executable)
        self.ct2_loaded_runtime = None
        self.asr = None
        self.translator = None
        self.qwen3 = None
        self._qwen3_loader = None                  # tests only: replaces load_qwen3
        self._qwen3_lock = threading.Lock()
        self.tokenizer = None
        self.ct2 = False
        self.asr_precision = self.translation_precision = None
        self.asr_state = self.translation_state = None
        self.vad_requested = None
        try:
            from lingua import LanguageDetectorBuilder
            self.detector = LanguageDetectorBuilder.from_all_languages().build()
        except Exception as error:
            raise MovieConfigurationError('text_language_runtime_failed', str(error)) from error

    def device_report(self):
        report = {'requested': self.device_name, 'cpu_fallback': False,
                  'asr': self.device_name if self.asr is not None else None,
                  'translation': self.device_name if self.translator is not None else None,
                  'asr_precision': self.asr_precision,
                  'translation_precision': self.translation_precision,
                  'asr_actual': self.asr_state, 'translation_actual': self.translation_state,
                  'ct2_policy': self.policy_metadata(),
                  'vad_filter': {'enabled':self.vad_requested,
                                 'device':'cpu' if self.vad_requested else None,
                                 'backend':'faster-whisper-silero-filter'},
                  'supporting_cpu_work': ['text-language-detection', 'tokenization', 'audio-decoding']}
        if self.qwen3 is not None:                 # only a qwen3 run reports the engine; m2m100 reports are unchanged
            report['translation_engine'] = 'qwen3'
            if self.qwen3.quantization != 'none':  # a bf16 qwen3 report keeps its original keys
                report['translation_quantization'] = self.qwen3.quantization
        return report

    def policy_metadata(self):
        """Ready/cache receipts distinguish planned FP16 from actual model loads."""
        return {'policy': ct2_policy_metadata(), 'runtime': dict(self.ct2_runtime),
                'loaded_runtime': None if self.ct2_loaded_runtime is None else dict(self.ct2_loaded_runtime),
                'requested_device': self.device_name,
                'requested_compute_type': CUDA_COMPUTE_TYPE if self.device == 'cuda' else 'float32',
                'capability_check_is_inference_proof': False}

    def text_language(self, text, declared=None):
        declared = iso(declared)
        if declared and declared not in ("und", "unknown"):
            return declared, 1.0, "subtitle-metadata"
        values = self.detector.compute_language_confidence_values(str(text))
        if not values:
            return "und", 0.0, "text-detection"
        best = values[0]
        return best.language.iso_code_639_1.name.lower(), float(best.value), "text-detection"

    def _asr_model(self):
        """The Whisper model, loaded on first use (shared by speech() and language_id())."""
        if self.asr is None:
            from faster_whisper import WhisperModel
            import ctranslate2
            self.ct2_loaded_runtime = require_loaded_runtime(ctranslate2, self.ct2_runtime)
            path = self.config["asr_model"]
            if not Path(path).is_dir():
                raise MovieConfigurationError('asr_assets_missing',
                                              'Local multilingual ASR assets are missing')
            precision = ct2_compute_type(self.device_name)
            self.asr = WhisperModel(path, device=self.device, device_index=self.device_index,
                                    compute_type=precision,
                                    cpu_threads=int(self.config["cpu_threads"]),
                                    local_files_only=True)
            self.asr_state = _ct2_model_state(self.asr.model, precision, self.device_name, 'Whisper')
            self.asr_precision = self.asr_state['compute_type']
        return self.asr

    def language_id(self, wav, spans):
        """Spoken language of each [start, end] second span of a 16 kHz wav, from the audio alone (Whisper language
        detection on the excerpt, padded to one second): one {'language', 'probability', 'top'} per span, None for a
        span shorter than 0.3 s. A recognizer that was told the wrong window language writes Russian speech down as
        English words (and the reverse); this is the independent evidence that catches it."""
        try:
            import numpy as np
            import soundfile as sf
            model = self._asr_model()
            x, rate = sf.read(str(wav), dtype='float32', always_2d=True)
            if rate != 16000:
                raise MovieConfigurationError('language_id_rate', 'language_id expects a 16 kHz excerpt, got %s Hz' % rate)
            x = x.mean(axis=1)
            out = []
            for start, end in spans:
                a, b = max(0, int(float(start) * 16000)), min(len(x), int(float(end) * 16000))
                if b - a < 0.3 * 16000:
                    out.append(None)
                    continue
                clip = x[a:b]
                clip = np.concatenate([clip, np.zeros(max(0, 16000 - len(clip)), dtype='float32')])
                try:
                    code, probability, every = model.detect_language(clip, language_detection_segments=1)
                except TypeError:
                    code, probability, every = model.detect_language(clip)
                top = sorted(every, key=lambda item: -item[1])[:4]
                out.append({'language': iso(code), 'probability': float(probability),
                            'top': [[iso(c), round(float(p), 3)] for c, p in top]})
            return out
        except MovieFault:
            raise
        except Exception as error:
            raise backend_fault('asr', error, self.device_name) from error

    def speech(self, wav, language=None, *, vad=True, beam_size=5, task='transcribe'):
        if beam_size not in (1,5):raise ValueError('Unsupported bounded ASR beam size')
        if task not in ('transcribe', 'translate'):
            raise MovieConfigurationError('asr_task_invalid', 'ASR task must be transcribe or translate')
        if task == 'translate':
            path = Path(self.config['asr_model'])
            try:
                metadata = json.loads((path / 'config.json').read_text(encoding='utf-8'))
            except (OSError, ValueError) as error:
                raise MovieConfigurationError('asr_model_config_invalid', str(error)) from error
            layers = metadata.get('num_decoder_layers', metadata.get('decoder_layers'))
            if 'turbo' in str(path).lower() or layers == 4:
                raise MovieConfigurationError('asr_translate_turbo_refused', 'Turbo is not an English translation model')
        self.vad_requested = bool(vad)
        try:
            self._asr_model()
            source_language = iso(language)
            if source_language in ('und', 'unknown', 'auto'):
                source_language = ''
            if task == 'translate' and getattr(self.asr.model, 'num_decoder_layers', None) == 4:
                raise MovieConfigurationError('asr_translate_turbo_refused', 'Four-decoder-layer model refused for translate')
            segments, info = self.asr.transcribe(
                str(wav), language=source_language or None, task=task, beam_size=beam_size,
                vad_filter=vad, word_timestamps=(task == 'transcribe'), condition_on_previous_text=False)
            # Consume the generator inside the failure boundary: CUDA execution
            # happens here, not necessarily inside transcribe().
            values = [{"start": float(x.start), "end": float(x.end), "text": x.text.strip(),
                       "log_probability": float(x.avg_logprob),
                       "no_speech_probability": float(x.no_speech_prob),
                       "words": [{"start": float(w.start), "end": float(w.end),
                                  "text": w.word, "probability": float(w.probability)}
                                 for w in (x.words or [])]}
                      for x in segments if x.text.strip()]
        except Exception as error:
            raise backend_fault('asr', error, self.device_name) from error
        return values, {"language": iso(info.language),
                        "probability": float(info.language_probability)}

    def english(self, text, language):
        language = iso(language)
        if language == "en":
            return text, "already-English"
        if not language or language in ("und", "unknown"):
            raise MovieContentFallback('source_language_unresolved', 'The source language has not been resolved')
        if self.translation_engine() == 'qwen3':
            result = self._qwen3_translate([(text, language)])[0]
            if isinstance(result, BaseException):
                raise result
            return result
        if self.translator is None:
            try:
                from transformers import M2M100Tokenizer
                path = self.config["translation_model"]
                if not Path(path).is_dir():
                    raise MovieConfigurationError('translation_assets_missing',
                                                  'Local multilingual translation assets are missing')
                self.tokenizer = M2M100Tokenizer.from_pretrained(path, local_files_only=True)
                settings=getattr(self.config,'data',self.config)
                self.ct2=bool(settings.get('translation_ct2_model'))
                if self.ct2:
                    import ctranslate2
                    self.ct2_loaded_runtime = require_loaded_runtime(ctranslate2, self.ct2_runtime)
                    model_path = settings['translation_ct2_model']
                    if not Path(model_path).is_dir():
                        raise MovieConfigurationError('translation_ct2_assets_missing',
                                                      'Configured local CTranslate2 translation assets are missing')
                    precision = ct2_compute_type(self.device_name)
                    self.translator=ctranslate2.Translator(
                        model_path, device=self.device, device_index=self.device_index, compute_type=precision,
                        inter_threads=1, intra_threads=int(settings.get('cpu_threads',2)))
                    self.translation_state = _ct2_model_state(self.translator, precision, self.device_name, 'M2M100 translation')
                    self.translation_precision = self.translation_state['compute_type']
                else:
                    import torch
                    from transformers import M2M100ForConditionalGeneration
                    target = torch_model_device(self.device_name)
                    dtype = torch.float16 if self.device == 'cuda' else torch.float32
                    self.translator = M2M100ForConditionalGeneration.from_pretrained(
                        path, local_files_only=True, torch_dtype=dtype).to(target).eval()
                    parameter = next(self.translator.parameters())
                    self.translation_precision = str(parameter.dtype)
                    self.translation_state = {'backend':'torch', 'compute_type':self.translation_precision,
                                              'device':str(parameter.device)}
            except Exception as error:
                raise backend_fault('translation_load', error, self.device_name) from error
        if language not in self.tokenizer.lang_code_to_id:
            raise MovieContentFallback('source_language_unsupported',
                                       f'Local translation does not support detected language {language}')
        self.tokenizer.src_lang = language
        try:
            if self.ct2:
                tokens=self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
                if len(tokens)>900:
                    raise MovieContentFallback('translation_input_too_long', 'Dialogue is too long for one translation')
                prefix=self.tokenizer.convert_ids_to_tokens(self.tokenizer.get_lang_id('en'))
                result=self.translator.translate_batch([tokens],target_prefix=[[prefix]],beam_size=5,max_decoding_length=256)
                output=self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(result[0].hypotheses[0]),skip_special_tokens=True).strip()
                engine = 'm2m100-ct2-' + self.translation_precision + '-to-English'
            else:
                import torch
                encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
                if encoded["input_ids"].shape[-1] > 900:
                    raise MovieContentFallback('translation_input_too_long', 'Dialogue is too long for one translation')
                encoded = {key: value.to(self.device_name) for key, value in encoded.items()}
                with torch.inference_mode():
                    result = self.translator.generate(
                        **encoded, forced_bos_token_id=self.tokenizer.get_lang_id("en"),
                        max_new_tokens=512, num_beams=5)
                output = self.tokenizer.batch_decode(result, skip_special_tokens=True)[0].strip()
                engine = 'm2m100-to-English'
        except Exception as error:
            raise backend_fault('translation', error, self.device_name) from error
        if not output:
            raise MovieContentFallback('translation_empty', 'Translation returned no English text')
        return output, engine

    # ---- translation engine selection / Qwen3 branch ------------------------------------------------------------
    def translation_engine(self):
        """'m2m100' (default, also when the key is absent) or 'qwen3', from the settings this object was given."""
        return translation_engine_of(self.config)

    def _qwen3_translator(self):
        with self._qwen3_lock:
            if self.qwen3 is None:
                settings = getattr(self.config, 'data', self.config)
                path = settings.get('translation_llm_model')
                if not str(path or '').strip() or not Path(str(path)).is_dir():
                    raise MovieConfigurationError('translation_llm_assets_missing',
                                                  'translation_engine is qwen3 but translation_llm_model is not an '
                                                  'existing local Qwen3-8B folder')
                self.qwen3 = Qwen3Translator(path, self.device_name, loader=self._qwen3_loader,
                                             quantization=translation_quantization_of(settings))
            return self.qwen3

    def _qwen3_translate(self, rows):
        """Qwen3 rows [(text, language[, max_words[, min_words]])] -> aligned [(english, basis) | MovieFault]; load failures raise."""
        translator = self._qwen3_translator()
        results = translator.translate_batch([(row[0], iso(row[1])) + tuple(row[2:4]) for row in rows])
        if translator.model is not None and self.translator is None:
            self.translator = translator               # device_report() shows the loaded Qwen3 model
            self.translation_state = dict(translator.state or {})
            self.translation_precision = self.translation_state.get('compute_type')
        return results

    def english_batch(self, rows):
        """Translate several cues at once (Qwen3: left-padded batches of at most 16 rows).

        rows: [{'text', 'language'}, ...] or [(text, language), ...]. Returns one entry per row, in order: either
        (english, basis) or a MovieFault instance (returned, never raised) describing that row's failure alone: an
        unresolved/unsupported language, empty or oversized text, empty or truncated model output, a generation
        failure. Only whole-call problems raise: a model that cannot be loaded or a missing CUDA device. 'en' rows pass
        through as (text, 'already-English'). With translation_engine 'm2m100' rows are translated one by one."""
        results, pending = [None] * len(rows), []
        for index, row in enumerate(rows):
            try:
                text, language = _batch_row(row)
            except MovieFault as error:
                results[index] = error
                continue
            code = iso(language)
            if code == 'en':
                results[index] = (text, 'already-English')
            elif not code or code in ('und', 'unknown'):
                results[index] = MovieContentFallback('source_language_unresolved', 'The source language has not been resolved')
            else:
                pending.append((index, text, language, _batch_max_words(row), _batch_min_words(row)))
        if not pending:
            return results
        if self.translation_engine() == 'qwen3':
            outcomes = self._qwen3_translate([_qwen3_row(text, language, words, fuller)
                                              for _, text, language, words, fuller in pending])
            for (index, _, _, _, _), outcome in zip(pending, outcomes):
                results[index] = outcome
        else:
            for index, text, language, _, _ in pending:
                try:
                    results[index] = self.english(text, language)
                except MovieContentFallback as error:
                    results[index] = error
        return results


def words(text):
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


def aligned_caption_style(translation,caption):
    """Use caption punctuation/articles only when its lexical content agrees."""
    def content(value):return [w for w in words(value) if w not in ('a','an','the')]
    if caption and content(translation) and content(translation)==content(caption):
        return caption.strip(),True
    return translation,False


def word_error(expected, heard):
    a, b = words(expected), words(heard)
    if not a:
        return 0.0 if not b else 1.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / len(a)
