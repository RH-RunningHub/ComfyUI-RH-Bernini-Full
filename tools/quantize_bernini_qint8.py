#!/usr/bin/env python3
"""Create a quanto qint8 copy of ByteDance/Bernini-Diffusers.

The output directory keeps the original Diffusers layout but replaces the large
`bernini` checkpoint shards with a frozen quanto qint8 state dict plus
`quantization_map.json`. Other assets are hard-linked when possible so the
conversion is restartable and does not duplicate tokenizer/config files.
"""

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from optimum.quanto import freeze, qint8, quantization_map, quantize
from safetensors.torch import save_model


PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from bernini.models import BerniniConfig, BerniniModel  # noqa: E402
from bernini.pipeline import _localize_bernini_config  # noqa: E402


SKIP_NAMES = {
    "._____temp",
    ".msc",
    ".mv",
}


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _link_or_copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_layout(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        rel_root = root_path.relative_to(source)
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        (target / rel_root).mkdir(parents=True, exist_ok=True)

        for filename in files:
            if filename in SKIP_NAMES:
                continue
            src = root_path / filename
            rel = src.relative_to(source)
            if rel.parts and rel.parts[0] == "bernini":
                if filename.startswith("model-") and filename.endswith(".safetensors"):
                    continue
                if filename == "model.safetensors.index.json":
                    continue
            _link_or_copy_file(src, target / rel)


def validate_source(source: Path):
    config_path = source / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "bernini":
        raise ValueError(f"expected model_type='bernini', got {config.get('model_type')!r}")
    lowered = str(source).lower()
    if "bernini-r" in lowered or "bernini_r" in lowered:
        raise ValueError("Bernini-R source paths are not allowed")


def quantize_bernini(source: Path, target: Path):
    out_dir = target / "bernini"
    qstate_path = out_dir / "model.safetensors"
    qmap_path = out_dir / "quantization_map.json"
    if qstate_path.exists() and qmap_path.exists():
        print(f"[skip] existing qint8 files found in {out_dir}", flush=True)
        return

    config = BerniniConfig.from_pretrained(str(source))
    _localize_bernini_config(config, str(source))
    print("[load] BerniniModel.from_pretrained on CPU", flush=True)
    model = BerniniModel.from_pretrained(
        str(source),
        subfolder=config.bernini_ckpt_subfolder,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    setattr(model.diff_dec, "transformer_2", model.diff_dec_low.transformer_2)
    model.diff_dec_low = None
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _cleanup()

    print("[quantize] optimum.quanto weights=qint8", flush=True)
    quantize(model, weights=qint8)
    freeze(model)
    _cleanup()

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_state = qstate_path.with_suffix(".safetensors.tmp")
    tmp_map = qmap_path.with_suffix(".json.tmp")
    print(f"[save] {qstate_path}", flush=True)
    save_model(model, str(tmp_state))
    with tmp_map.open("w", encoding="utf-8") as f:
        json.dump(quantization_map(model), f, indent=2)
    os.replace(tmp_state, qstate_path)
    os.replace(tmp_map, qmap_path)

    with (out_dir / "qint8_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(source),
                "weights": "qint8",
                "backend": "optimum.quanto",
                "format": "low-level state_dict plus quantization_map",
            },
            f,
            indent=2,
        )
    del model
    _cleanup()


def main():
    parser = argparse.ArgumentParser(description="Quantize full Bernini-Diffusers to qint8")
    parser.add_argument("--source", required=True, help="Source Bernini-Diffusers directory")
    parser.add_argument("--target", required=True, help="Target Bernini-Diffusers-qint8 directory")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    if source == target:
        raise ValueError("source and target must be different directories")
    validate_source(source)
    prepare_layout(source, target)
    quantize_bernini(source, target)
    print(f"[done] qint8 model directory: {target}", flush=True)


if __name__ == "__main__":
    main()
