"""Explicit offline-asset provisioning; does not instantiate any model.

Qwen3-8B (translation_engine "qwen3") is never downloaded or copied by this script and its weights are not part of the
repository: you supply the folder (about 16.4 GB) yourself. Verify it, without network access and without loading it:

    python scripts/provision_models.py --verify-translation-llm <folder>
    python scripts/provision_models.py --verify-translation-llm --config <relay-config.json>   (reads translation_llm_model)

Exit status 0 when the folder has config.json, the tokenizer, generation_config.json and safetensors shards of a
plausible Qwen3-8B size; 1 with the reasons otherwise.

The same bf16 folder serves both weight modes of the translation_quantization setting ("none" keeps bf16, about 16.4 GB
on the GPU; "int8" quantizes the decoder linears to weight-only INT8 while loading, about 9.4 GB). The check reports the
GPU size of the selected mode (translation_quantization from --config, or --translation-quantization none|int8).
"""
import argparse
import hashlib
import json
import os
import fnmatch
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    ("Systran/faster-whisper-small","536b0662742c02347bc0e980a01041f333bce120","faster-whisper-small",
     ["*.json","*.bin","*.txt","README.md"]),
    ("facebook/m2m100_418M","55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636","m2m100_418M",
     ["*.json","pytorch_model.bin","sentencepiece.bpe.model","README.md"]),
    ("pyannote/speaker-diarization-community-1","3533c8cf8e369892e6b79ff1bf80f7b0286a54ee","pyannote-community-1",
     ["config.yaml","*.bin","*.npz","**/*.bin","**/*.npz","README.md","**/README.md"]),
    ("adefossez/HTDemucs","cbc8a9b1a87023b7fd74e7b3412e6321c0eab003","htdemucs",
     ["*.yaml","*.json","*.safetensors","README.md"])
]


def verify_translation_llm(folder, config_file=None, quantization=None):
    """Print and return the exit status of the static Qwen3-8B folder check. No download, no model load.

    quantization: 'none' or 'int8' (the translation_quantization setting); None reads it from config_file, else 'none'."""
    raw = {}
    if config_file:
        try:
            raw = json.loads(Path(config_file).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            print("PROBLEM      : cannot read %s (%s)" % (config_file, error))
            return 1
    if not folder:
        if not config_file:
            print("Give the Qwen3-8B folder, or --config <relay-config.json> to read translation_llm_model from it.")
            return 1
        value = str(raw.get("translation_llm_model") or "").strip()
        if not value:
            print("translation_llm_model is not set in %s" % config_file)
            return 1
        path = Path(os.path.expandvars(value)).expanduser()
        folder = str(path if path.is_absolute() else (Path(config_file).resolve().parent / path).resolve())
    sys.path.insert(0, str(ROOT))
    from relay.language import qwen3_model_report, translation_quantization_of
    try:
        quantization = quantization or translation_quantization_of(raw)
        report = qwen3_model_report(folder, quantization)
    except Exception as error:                      # an unknown translation_quantization value
        print("PROBLEM      : %s" % error)
        return 1
    print("Qwen3-8B folder: %s" % folder)
    print("  quantization : %s" % ("int8 weight-only decoder linears (bf16 embeddings/norms/lm_head)"
                                   if quantization == "int8" else "none (bf16 weights)"))
    print("  weight files : %d (%.2f GB)" % (len(report["shard_files"]), report["weights_bytes"] / 1e9))
    if quantization == "int8" and report.get("budget"):
        budget = report["budget"]
        print("  GPU weights  : %d MiB resident (int8 %d + scales %d + bf16 tables/norms %d MiB)"
              % (budget["resident_mib"], budget["int8_weight_bytes"] >> 20,
                 (budget["scale_bytes"] + budget["scale_activation_bytes"]) >> 20,
                 (budget["embedding_bytes"] + budget["lm_head_bytes"] + budget["norm_bytes"]) >> 20))
    if report["footprint_mib"]:
        print("  GPU memory   : about %d MiB for the translator; the movie lease keeps its 3072 MiB free-VRAM floor on top"
              % report["footprint_mib"])
    for problem in report["problems"]:
        print("  PROBLEM      : " + problem)
    print("Folder is usable." if report["ok"] else "Folder is NOT usable. Nothing was downloaded; supply the Qwen3-8B files yourself.")
    print("Static check only: no model was loaded or run.")
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-model-terms", action="store_true")
    parser.add_argument("--verify-translation-llm", nargs="?", const="", metavar="FOLDER",
                        help="verify a user-supplied Qwen3-8B folder (no download, no model load) and exit")
    parser.add_argument("--config", help="with --verify-translation-llm and no FOLDER: read translation_llm_model from this settings file")
    parser.add_argument("--translation-quantization", choices=("none", "int8"),
                        help="with --verify-translation-llm: size the check for this weight mode "
                             "(default: translation_quantization from --config, else none)")
    args = parser.parse_args()
    if args.verify_translation_llm is not None:
        raise SystemExit(verify_translation_llm(args.verify_translation_llm, args.config, args.translation_quantization))
    if not args.accept_model_terms:
        raise SystemExit("Read THIRD-PARTY.md and accept upstream terms before provisioning.")
    from huggingface_hub import snapshot_download, hf_hub_download
    records = []
    for repo, revision, folder, patterns in MODELS:
        target = ROOT / "models" / folder
        # Reuse an exact already-provisioned revision without changing its cache.
        # Copy regular bytes, never create writable links into another project.
        from huggingface_hub.constants import HF_HUB_CACHE
        cached = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        required = {
            "faster-whisper-small": ["config.json","model.bin","tokenizer.json","vocabulary.txt"],
            "m2m100_418M": ["config.json","pytorch_model.bin","sentencepiece.bpe.model","vocab.json"],
            "pyannote-community-1": ["config.yaml","embedding/pytorch_model.bin",
                "segmentation/pytorch_model.bin","plda/plda.npz","plda/xvec_transform.npz"],
            "htdemucs": ["htdemucs.yaml","955717e8.safetensors"]
        }[folder]
        source = "pinned-download"
        if cached.is_dir() and all((cached / name).is_file() for name in required):
            for original in cached.rglob("*"):
                relative = original.relative_to(cached)
                if original.is_file() and any(fnmatch.fnmatch(relative.as_posix(), p) for p in patterns):
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, destination)
            source = "existing-exact-revision-copy"
            print("Copied existing assets: " + repo, flush=True)
        else:
            print("Provisioning pinned assets: " + repo, flush=True)
            snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=patterns,
                              token=os.environ.get("HF_TOKEN") or None, max_workers=2)
        records.append({"repo":repo,"revision":revision,"path":str(target.relative_to(ROOT)),
                        "source":source})
    prefix = "en/en_US/libritts_r/medium/"
    revision = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
    target = ROOT / "models/piper"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("en_US-libritts_r-medium.onnx","en_US-libritts_r-medium.onnx.json","MODEL_CARD"):
        print("Provisioning English voice asset: " + name, flush=True)
        downloaded = hf_hub_download("rhasspy/piper-voices", prefix + name, revision=revision)
        shutil.copy2(downloaded, target / name)
    records.append({"repo":"rhasspy/piper-voices","revision":revision,"path":"models/piper",
                    "voice":"en_US-libritts_r-medium"})
    files = {}
    for path in sorted((ROOT / "models").rglob("*")):
        if path.is_file() and ".cache" not in path.parts and path.name != "PROVENANCE.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b""):
                    digest.update(block)
            files[path.relative_to(ROOT).as_posix()] = digest.hexdigest()
    (ROOT / "models" / "PROVENANCE.json").write_text(json.dumps(
        {"models":records,"sha256":files,"inference_run":False},indent=2))
    print("Pinned model files saved; no model was loaded or tested.")


if __name__ == "__main__":
    main()
