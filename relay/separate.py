"""Bounded local Demucs runner, adapted from Sem's Movie Dub separation script.

Uses the upstream Demucs 4.1 safetensors loader with explicitly provisioned
assets. Model downloading and activation are separate from runtime jobs.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    import yaml
    import torch
    from demucs.hf import load_safetensors_model
    from demucs.apply import BagOfModels, apply_model
    from demucs.audio import AudioFile, save_audio
    root = Path(args.model_dir).resolve()
    manifest = root / (Path(args.model).name + ".yaml")
    bag = yaml.safe_load(manifest.read_text())
    paths = [(root / (str(signature) + ".safetensors")).resolve() for signature in bag["models"]]
    if any(p.parent != root for p in paths):
        raise ValueError("Unexpected separation model asset path")
    model = BagOfModels([load_safetensors_model(p) for p in paths],
                        bag.get("weights"), bag.get("segment")).to(args.device).eval()
    audio = AudioFile(args.media).read(streams=0, samplerate=model.samplerate,
                                      channels=model.audio_channels)
    with torch.inference_mode():
        stems = apply_model(model, audio[None], device=args.device)[0]
    index = list(model.sources).index("vocals")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_audio(stems[index], str(output / "vocals.wav"), samplerate=model.samplerate)
    save_audio(stems[[i for i in range(len(stems)) if i != index]].sum(dim=0),
               str(output / "bed.wav"), samplerate=model.samplerate)


if __name__ == "__main__":
    main()
