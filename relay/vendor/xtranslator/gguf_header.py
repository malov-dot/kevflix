"""CPU metadata-only GGUF compatibility gate. No torch/vLLM/model construction.

Run explicitly with the provisioned vLLM Python (gguf 0.19.0), under the Linux
supervisor's heartbeat and process ownership. GGUFReader uses a read-only mmap.
The original HF tokenizer/config remain the runtime tokenizer and chat template.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def inspect(model, original):
    from gguf import GGUFReader, GGMLQuantizationType

    model, original = Path(model), Path(original)
    with model.open('rb') as handle:
        magic, version = struct.unpack('<4sI', handle.read(8))
    if magic != b'GGUF' or version not in (2, 3):
        raise ValueError('Unsupported GGUF magic/version')
    config = json.loads((original / 'config.json').read_text())
    tokenizer = json.loads((original / 'tokenizer.json').read_text())
    tokenizer_config_path = original / 'tokenizer_config.json'
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding='utf-8'))
    # Transformers 4.57.6 _from_pretrained gives the independent default Jinja
    # file priority over tokenizer_config.json. Both paths are inside the original
    # model directory already file-set/size/SHA-verified by the supervisor.
    template_path = original / 'chat_template.jinja'
    if template_path.is_file():
        template = template_path.read_text(encoding='utf-8')
        template_storage = 'external-jinja'
    else:
        template_path = tokenizer_config_path
        template = tokenizer_config.get('chat_template')
        if isinstance(template, list):
            template = {item['name']: item['template'] for item in template}
        if isinstance(template, dict):
            template = template.get('default')
        template_storage = 'tokenizer-config-json'
    if not isinstance(template, str) or not template.strip():
        raise ValueError('Pinned original tokenizer chat template is missing')
    template_bytes = template_path.read_bytes()
    template_identity = {
        'source': 'file' if template_storage == 'external-jinja' else 'tokenizer_config',
        'path': str(template_path), 'bytes': len(template_bytes),
        'sha256': hashlib.sha256(template_bytes).hexdigest(),
        'git_blob_sha1': hashlib.sha1(
            ('blob %d\0' % len(template_bytes)).encode('ascii') + template_bytes).hexdigest()}
    if config.get('model_type') != 'qwen3' or config.get('architectures') != ['Qwen3ForCausalLM']:
        raise ValueError('Original HF configuration is not Qwen3ForCausalLM')
    reader = GGUFReader(str(model), mode='r')

    def field(key):
        entry = reader.get_field(key)
        if entry is None:
            raise ValueError('GGUF required field missing: ' + key)
        return entry.contents()

    if field('general.architecture') != config['model_type']:
        raise ValueError('GGUF architecture differs from original HF config')
    mapping = {'qwen3.block_count': 'num_hidden_layers',
               'qwen3.attention.head_count': 'num_attention_heads',
               'qwen3.attention.head_count_kv': 'num_key_value_heads',
               'qwen3.embedding_length': 'hidden_size',
               'qwen3.feed_forward_length': 'intermediate_size'}
    dimensions = {}
    for key, hf_key in mapping.items():
        value = int(field(key))
        if value != int(config[hf_key]) or value <= 0:
            raise ValueError('GGUF/HF dimension mismatch: ' + key)
        dimensions[key] = value
    head_dim = int(config.get('head_dim', config['hidden_size'] // config['num_attention_heads']))
    for key in ('qwen3.attention.key_length', 'qwen3.attention.value_length'):
        entry = reader.get_field(key)
        value = int(entry.contents()) if entry is not None else int(config['hidden_size']) // int(config['num_attention_heads'])
        if value != head_dim:
            raise ValueError('GGUF head dimension differs from original HF config: ' + key)
        dimensions[key] = value
    vocab_size = int(config['vocab_size'])
    tokens = field('tokenizer.ggml.tokens')
    if not isinstance(tokens, list) or len(tokens) != vocab_size:
        raise ValueError('GGUF vocabulary differs from original model padded vocab_size')
    # HF tokenizer can have fewer entries than the padded model vocabulary.
    vocab = tokenizer['model']['vocab']
    ids = list(vocab.values()) + [item['id'] for item in tokenizer.get('added_tokens', [])]
    if not ids or any(type(value) is not int or not 0 <= value < vocab_size for value in ids):
        raise ValueError('Original tokenizer IDs lie outside the configured vocabulary')
    counts = {'Q8_0': 0, 'F32': 0}
    embedding_seen = False
    for tensor in reader.tensors:
        if tensor.tensor_type == GGMLQuantizationType.Q8_0:
            counts['Q8_0'] += 1
        elif tensor.tensor_type == GGMLQuantizationType.F32 and len(tensor.shape) == 1 and 'norm' in tensor.name:
            counts['F32'] += 1
        else:
            raise ValueError('Unsupported weight type/shape: ' + tensor.name + ':' + str(tensor.tensor_type))
        if tensor.name == 'token_embd.weight':
            embedding_seen = True
            if [int(value) for value in tensor.shape] != [int(config['hidden_size']), vocab_size]:
                raise ValueError('GGUF embedding tensor shape differs from original config')
    if not embedding_seen or not counts['Q8_0'] or not counts['F32']:
        raise ValueError('Expected Q8_0 weights/F32 norms/embedding are absent')
    return {'version': 'xt-gguf-header-v1', 'ok': True, 'gguf_version': version,
            'reader': 'gguf.GGUFReader read-only mmap', 'architecture': config['model_type'],
            'dimensions': dimensions, 'head_dim': head_dim, 'vocab_size': vocab_size,
            'tokenizer_explicit_entries': len(vocab), 'tensor_types': counts,
            'tokenizer': str(original), 'chat_template_source': str(template_path),
            'chat_template_storage': template_storage,
            'chat_template_source_sha256': template_identity['sha256'],
            'chat_template_identity': template_identity,
            'chat_template_sha256': hashlib.sha256(template.encode('utf-8')).hexdigest(),
            'tokenizer_config_sha256': hashlib.sha256(tokenizer_config_path.read_bytes()).hexdigest(),
            'inference': False, 'activation_dtype': 'half'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--original', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.model, args.original), sort_keys=True))
    except Exception as error:
        print(json.dumps({'ok': False, 'code': 'xt_mt_gguf_unsupported',
                          'error': type(error).__name__ + ': ' + str(error)}))
        raise SystemExit(2)
