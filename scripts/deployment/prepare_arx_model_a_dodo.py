#!/usr/bin/env python3
"""Create a local inference view of the immutable H100 Task7 Model A export.

Run with the dodo Cosmos Python environment, which provides PyYAML. Large
weights stay in the copied package; this only links them and rewrites paths in
small runtime configs. It never opens the robot controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


DEFAULT_PACKAGE = Path(
    "/mnt/hdd16t/chenfu/cosmos_models/arx_model_a_5task_iter5000_20260817"
)
DEFAULT_VAE = Path("/mnt/hdd16t/chenfu/assets/Wan2.2_VAE.pth")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    package = args.package.resolve()
    source_model = package / "model"
    output = args.output or package / "runtime_dodo"
    output = output.absolute()
    runtime_model = output / "model"
    manifest = json.loads((package / "manifest.json").read_text())
    if (manifest.get("model_kind"), manifest.get("action_dim"),
            manifest.get("action_horizon"), manifest.get("control_hz")) != (
                "arx_task7_model_a_native_cosmos", 14, 32, 15):
        raise ValueError("package is not the expected Task7 Model A export")
    index = json.loads((source_model / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards or any(not (source_model / shard).is_file() for shard in shards):
        raise ValueError("model index references missing weight shards")
    if not args.vae.is_file():
        raise FileNotFoundError(args.vae)
    if output.exists():
        raise FileExistsError(f"runtime already exists: {output}")

    source_config = source_model / "config.json"
    source_training = package / "config/config.yaml"
    model_config = json.loads(source_config.read_text())
    training_config = yaml.safe_load(source_training.read_text())
    if not isinstance(training_config, dict):
        raise ValueError("Model A training config must be a mapping")

    output.mkdir(parents=True)
    runtime_model.mkdir()
    for source in source_model.iterdir():
        if source.name != "config.json":
            (runtime_model / source.name).symlink_to(source)

    model = model_config["model"]["config"]
    model["tokenizer"]["vae_path"] = str(args.vae)
    model["vlm_config"]["tokenizer"]["tokenizer_type"] = str(runtime_model)
    model["vlm_config"]["pretrained_weights"]["enabled"] = False
    model["vlm_config"]["pretrained_weights"]["backbone_path"] = str(runtime_model)
    _write_json(runtime_model / "config.json", model_config)

    training_config.pop("_type", None)
    training_config["checkpoint"]["load_path"] = str(runtime_model)
    training_config["checkpoint"]["load_from_object_store"]["enabled"] = False
    model = training_config["model"]["config"]
    model["tokenizer"]["vae_path"] = str(args.vae)
    model["vlm_config"]["tokenizer"]["tokenizer_type"] = str(runtime_model)
    model["vlm_config"]["pretrained_weights"]["enabled"] = False
    model["vlm_config"]["pretrained_weights"]["backbone_path"] = str(runtime_model)
    model["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    model["ema"]["enabled"] = False
    for dataloader_name in ("dataloader_train", "dataloader_val"):
        dataloader = training_config.get(dataloader_name, {}).get("dataloader", {})
        for entry in dataloader.get("datasets", {}).values():
            dataset = entry.get("dataset", {})
            tokenizer = dataset.get("tokenizer_config")
            if isinstance(tokenizer, dict):
                tokenizer["tokenizer_type"] = str(runtime_model)
    runtime_config = output / "config.dodo.yaml"
    runtime_config.write_text(yaml.safe_dump(training_config, sort_keys=False))

    _write_json(output / "provenance.json", {
        "source_package": str(package),
        "source_manifest_sha256": hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "source_training_config_sha256": hashlib.sha256(source_training.read_bytes()).hexdigest(),
        "runtime_model_config_sha256": hashlib.sha256((runtime_model / "config.json").read_bytes()).hexdigest(),
        "runtime_config_sha256": hashlib.sha256(runtime_config.read_bytes()).hexdigest(),
        "weight_shards": sorted(shards),
    })
    print(json.dumps({"runtime_model": str(runtime_model),
                      "runtime_config": str(runtime_config),
                      "weight_shards": sorted(shards)}, sort_keys=True))


if __name__ == "__main__":
    main()
