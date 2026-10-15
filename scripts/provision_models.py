"""Explicit offline-asset provisioning; does not instantiate any model."""
import argparse
import hashlib
import json
import os
import fnmatch
import shutil
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-model-terms", action="store_true")
    args = parser.parse_args()
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
