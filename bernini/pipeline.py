# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end full Bernini inference pipeline: preprocess -> plan -> sample -> decode -> save."""

import gc
import html
import json
import logging
import math
import os
import random
import re
from functools import partial
from types import SimpleNamespace
from typing import Optional

import ftfy
import numpy as np
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, Qwen2_5_VLModel

from .data.bernini_process import bernini_process_sample
from .data.bernini_template import BerniniTemplate
from .data.utils.video_utils import PathVideoReader, smart_video_nframes
from .data_utils import tensor_to_bytes, get_vit_features, get_vae_features, FakeVideoReader, create_fake_image, VAEVideoTransform
from .io_utils import save_output
from .models import BerniniConfig, BerniniModel
from .models.transformer_wan import WanRotaryPosEmbed

logger = logging.getLogger("bernini.pipeline")


def _cleanup_cuda(aggressive: bool = False):
    """Release Python refs and CUDA allocator cache between heavy stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            try:
                torch.cuda.ipc_collect()
            except Exception:
                logger.debug("torch.cuda.ipc_collect failed", exc_info=True)
    if aggressive and os.name == "posix":
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            logger.debug("malloc_trim failed", exc_info=True)


def _module_to(module, device, dtype=None):
    if module is None:
        return None
    if dtype is not None and _has_quanto_weight(module):
        return module.to(device)
    if dtype is None:
        return module.to(device)
    return module.to(device=device, dtype=dtype)


def _has_quanto_weight(module) -> bool:
    for child in module.modules():
        weight = getattr(child, "weight", None)
        if hasattr(weight, "_data") and hasattr(weight, "_scale"):
            return True
    return False


def _configure_vae_chunking(vae, enable_tiling: bool = False, enable_slicing: bool = False):
    """Use diffusers VAE chunking primitives when the current VAE exposes them."""
    if enable_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    elif hasattr(vae, "disable_tiling"):
        vae.disable_tiling()

    if enable_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    elif hasattr(vae, "disable_slicing"):
        vae.disable_slicing()


def _chunked(items, chunk_size: int):
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(items) or 1
    for start in range(0, len(items), max(1, chunk_size)):
        yield items[start:start + max(1, chunk_size)]


def _as_pil_image(image):
    if isinstance(image, str):
        with Image.open(image) as img:
            return img.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image


def _resolve_cached_hf_path(path: Optional[str]) -> Optional[str]:
    """Resolve a HF repo id (or repo id + subpath) to a local cached path.

    Examples:
    - ``ByteDance/Bernini-Diffusers`` -> ``~/.cache/.../snapshots/<rev>``
    - ``ByteDance/Bernini-Diffusers/vae/config.json`` ->
      ``~/.cache/.../snapshots/<rev>/vae/config.json``

    Returns the original ``path`` if it already exists locally or if it cannot
    be resolved from the local HF cache.
    """
    if path is None or os.path.exists(path):
        return path

    parts = path.split("/")
    if len(parts) < 2:
        return path

    repo_id = "/".join(parts[:2])
    subpath = os.path.join(*parts[2:]) if len(parts) > 2 else ""

    try:
        from huggingface_hub import snapshot_download

        repo_dir = snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return path

    candidate = os.path.join(repo_dir, subpath) if subpath else repo_dir
    if os.path.exists(candidate):
        if candidate != path:
            logger.info("resolved hub path '%s' to cached path '%s'", path, candidate)
        return candidate
    return path


def _prefer_local_dir(current, config_dir, *required):
    """Resolve a component base path from config.json against the directory
    passed to from_pretrained.

    Released configs may carry a hub repo id or a path relative to the repo
    root; when that path does not exist locally but `config_dir` contains the
    `required` entries, load from `config_dir` instead so an
    already-downloaded directory is not re-fetched from the Hub.
    """
    config_dir = _resolve_cached_hf_path(config_dir)
    if current is not None and os.path.exists(current):
        return current
    if os.path.isdir(config_dir) and all(
        os.path.exists(os.path.join(config_dir, r)) for r in required if r
    ):
        if current is not None and current != config_dir:
            logger.info(
                "component path '%s' not found locally; loading from '%s' instead",
                current,
                config_dir,
            )
        return config_dir
    current = _resolve_cached_hf_path(current)
    if current is not None and os.path.exists(current):
        return current
    return current


def _prefer_local_file(current, config_dir, *parts):
    """Like _prefer_local_dir, but the local candidate is an entry inside
    `config_dir` rather than the directory itself."""
    config_dir = _resolve_cached_hf_path(config_dir)
    if current is not None and os.path.exists(current):
        return current
    candidate = os.path.join(config_dir, *parts)
    if os.path.exists(candidate):
        return candidate
    current = _resolve_cached_hf_path(current)
    if current is not None and os.path.exists(current):
        return current
    return current


def _localize_bernini_config(config, config_dir):
    """Point the component paths of a BerniniConfig at `config_dir` when the
    paths baked into config.json (repo-root-relative in the released
    Bernini-Diffusers layout) do not exist locally."""
    config_dir = _resolve_cached_hf_path(config_dir)
    config.base_dir = _prefer_local_dir(config.base_dir, config_dir)
    config.diff_dec_config_path = _prefer_local_dir(config.diff_dec_config_path, config.base_dir or config_dir)
    config.mllm_config_path = _prefer_local_dir(
        config.mllm_config_path, config.base_dir or config_dir, config.mllm_subfolder
    )
    config.processor_config_path = _prefer_local_dir(
        config.processor_config_path, config.base_dir or config_dir, config.processor_subfolder
    )
    config.t5_text_encoder_path = _prefer_local_dir(
        config.t5_text_encoder_path, config.base_dir or config_dir, config.t5_text_encoder_subfolder
    )
    config.t5_tokenizer_path = _prefer_local_dir(
        config.t5_tokenizer_path, config.base_dir or config_dir, config.t5_tokenizer_subfolder
    )
    config.vae_model_path = _prefer_local_dir(
        config.vae_model_path, config.base_dir or config_dir, config.vae_subfolder or "vae"
    )
    config.vae_config_path = _prefer_local_file(
        config.vae_config_path, config.base_dir or config_dir, config.vae_subfolder or "vae", "config.json"
    )
    config.transformer_config_path = _prefer_local_file(
        config.transformer_config_path, config.base_dir or config_dir, "transformer_config.json"
    )
    config.transformer_2_config_path = _prefer_local_file(
        config.transformer_2_config_path, config.base_dir or config_dir, "transformer_2_config.json"
    )
    config.scheduler_config_path = _prefer_local_file(
        config.scheduler_config_path, config.base_dir or config_dir, "scheduler"
    )


def _load_full_bernini_model(config, ckpt: str):
    qmap_path = os.path.join(ckpt, config.bernini_ckpt_subfolder or "bernini", "quantization_map.json")
    qstate_path = os.path.join(ckpt, config.bernini_ckpt_subfolder or "bernini", "model.safetensors")

    if os.path.exists(qmap_path) and os.path.exists(qstate_path):
        logger.info("Loading quanto qint8 BerniniModel from '%s'", qstate_path)
        from safetensors import safe_open
        from safetensors.torch import load_file
        from optimum.quanto import requantize

        model = BerniniModel(config)
        setattr(model.diff_dec, "transformer_2", model.diff_dec_low.transformer_2)
        model.diff_dec_low = None
        with open(qmap_path, "r", encoding="utf-8") as f:
            qmap = json.load(f)
        state_dict = load_file(qstate_path, device="cpu")
        with safe_open(qstate_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        for alias_name, tensor_name in metadata.items():
            if alias_name not in state_dict and tensor_name in state_dict:
                state_dict[alias_name] = state_dict[tensor_name]
        requantize(model, state_dict, qmap, device=torch.device("cpu"))
        del state_dict
        _cleanup_cuda(aggressive=True)
    else:
        model = BerniniModel.from_pretrained(
            ckpt,
            subfolder=config.bernini_ckpt_subfolder,
            config=config,
        )
        # transformer_1 is loaded in diff_dec, while transformer_2 is loaded in diff_dec_low and then
        # attached back to diff_dec before sampling.
        setattr(model.diff_dec, "transformer_2", model.diff_dec_low.transformer_2)
        model.diff_dec_low = None

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class _LoRALinear(torch.nn.Module):
    def __init__(self, base, down: torch.Tensor, up: torch.Tensor, alpha: float, scale: float):
        super().__init__()
        self.base = base
        rank = max(1, int(down.shape[0]))
        self.multiplier = float(scale) * float(alpha) / float(rank)
        self.register_buffer("lora_down", down.contiguous())
        self.register_buffer("lora_up", up.contiguous())

    def forward(self, x):
        output = self.base(x)
        lora_dtype = self.lora_down.dtype
        lora = torch.nn.functional.linear(x.to(dtype=lora_dtype), self.lora_down)
        lora = torch.nn.functional.linear(lora, self.lora_up)
        if lora.dtype != output.dtype:
            lora = lora.to(dtype=output.dtype)
        return output.add_(lora, alpha=self.multiplier)


_WAN_LORA_TARGETS = {
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
}


def _get_submodule(root, path: str):
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set_submodule(root, path: str, value):
    parts = path.split(".")
    parent = _get_submodule(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _linear_weight_shape(module):
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    shape = getattr(weight, "shape", None)
    if shape is not None:
        return tuple(shape)
    data = getattr(weight, "_data", None)
    if data is not None:
        return tuple(data.shape)
    return None


def _collect_wan_lora_prefixes(keys):
    prefixes = []
    for key in keys:
        if key.endswith(".lora_down.weight"):
            prefix = key[: -len(".lora_down.weight")]
            up_key = f"{prefix}.lora_up.weight"
        elif key.endswith(".lora_A.weight"):
            prefix = key[: -len(".lora_A.weight")]
            up_key = f"{prefix}.lora_B.weight"
        else:
            continue
        if up_key in keys:
            prefixes.append((prefix, key, up_key))
    return prefixes


def _apply_wan_lora(transformer, lora_path: str, scale: float, dtype=torch.bfloat16) -> int:
    if transformer is None or not lora_path:
        return 0
    lora_path = os.path.abspath(os.path.expanduser(lora_path))
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"Wan LightX2V LoRA not found: {lora_path}")

    from safetensors import safe_open

    loaded = 0
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for prefix, down_key, up_key in _collect_wan_lora_prefixes(keys):
            match = re.match(r"^diffusion_model\.blocks\.(\d+)\.(.+)$", prefix)
            if not match:
                continue
            block_index, wan_target = match.groups()
            local_target = _WAN_LORA_TARGETS.get(wan_target)
            if local_target is None:
                continue

            module_path = f"blocks.{block_index}.{local_target}"
            try:
                base = _get_submodule(transformer, module_path)
            except AttributeError:
                logger.warning("Skipping LoRA target missing in transformer: %s", module_path)
                continue

            down = f.get_tensor(down_key).to(dtype=dtype)
            up = f.get_tensor(up_key).to(dtype=dtype)
            alpha_key = f"{prefix}.alpha"
            alpha = float(f.get_tensor(alpha_key).item()) if alpha_key in keys else float(down.shape[0])
            weight_shape = _linear_weight_shape(base)
            if weight_shape is not None and (tuple(up.shape[:1] + down.shape[1:]) != weight_shape):
                raise ValueError(
                    f"LoRA shape mismatch for {module_path}: "
                    f"delta {(up.shape[0], down.shape[1])}, base {weight_shape}"
                )

            _set_submodule(transformer, module_path, _LoRALinear(base, down, up, alpha=alpha, scale=scale))
            loaded += 1

    if loaded <= 0:
        raise ValueError(f"No compatible Wan LightX2V LoRA tensors were loaded from {lora_path}")
    logger.info("Applied %d Wan LightX2V LoRA projections from '%s'", loaded, lora_path)
    return loaded


def _prompt_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def _vae_encode(vae, x: torch.Tensor) -> torch.Tensor:
    """Encode `[1,C,T,H,W]` pixels into normalized VAE latents."""
    latents = vae.encode(x).latent_dist.mode()
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype, device=latents.device).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=latents.dtype, device=latents.device).view(1, z, 1, 1, 1)
    return (latents - mean) / std


def _get_t5_text_ids(text, tokenizer, max_length: int = 512):
    """Tokenize text for the T5 encoder, returning input_ids and attention_mask."""
    text = _prompt_clean(text)
    out = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return out.input_ids, out.attention_mask


def _vae_decode(vae, latents: torch.Tensor):
    """Decode VAE latents into a numpy clip `[T, H, W, C]` in [0, 1]."""
    latents = latents.to(vae.dtype)
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, z, 1, 1, 1)
    latents = latents * std + mean
    video = vae.decode(latents, return_dict=False)[0]
    processor = VideoProcessor(vae_scale_factor=2 ** len(vae.temperal_downsample))
    output = processor.postprocess_video(video, output_type="np")[0]
    del latents, video
    _cleanup_cuda()
    return output


class BerniniPipeline:
    """Loads the model once; each call generates one video / image."""

    def __init__(self, config, model, vae, t5_tokenizer, vit_processor, device):
        self.config = config
        self.model = model
        self.vae = vae
        self.t5_tokenizer = t5_tokenizer
        self.vit_processor = vit_processor
        self.device = device
        self.weight_dtype = torch.bfloat16
        self.text_encoder = model.mllm
        self.connector = getattr(model, "connector", None)

    def _load_planner_stage(self):
        if self.model is None:
            raise RuntimeError("Bernini pipeline has been destroyed and cannot be reused.")
        _module_to(self.model.mllm, self.device, self.weight_dtype)
        if self.connector is not None:
            _module_to(self.connector, self.device, self.weight_dtype)
        if getattr(self.model, "vit_decoder", None) is not None:
            _module_to(self.model.vit_decoder, self.device, self.weight_dtype)

    def _unload_planner_stage(self, aggressive: bool = False):
        if self.model is None:
            _cleanup_cuda(aggressive=aggressive)
            return
        _module_to(self.model.mllm, "cpu")
        if self.connector is not None:
            _module_to(self.connector, "cpu")
        if getattr(self.model, "vit_decoder", None) is not None:
            _module_to(self.model.vit_decoder, "cpu")
        _cleanup_cuda(aggressive=aggressive)

    def _load_t5_stage(self):
        if self.model is None:
            raise RuntimeError("Bernini pipeline has been destroyed and cannot be reused.")
        if getattr(self.model, "t5_text_encoder", None) is not None:
            _module_to(self.model.t5_text_encoder, self.device)

    def _unload_t5_stage(self, aggressive: bool = False):
        if getattr(self.model, "t5_text_encoder", None) is not None:
            _module_to(self.model.t5_text_encoder, "cpu")
        _cleanup_cuda(aggressive=aggressive)

    def _load_vae_stage(self, enable_tiling: bool = False, enable_slicing: bool = False):
        _configure_vae_chunking(self.vae, enable_tiling=enable_tiling, enable_slicing=enable_slicing)
        _module_to(self.vae, self.device)

    def _unload_vae_stage(self, aggressive: bool = False):
        _module_to(self.vae, "cpu")
        _cleanup_cuda(aggressive=aggressive)

    def unload_all(self, aggressive: bool = True):
        """Move every heavy stage to CPU and clear CUDA cache."""
        self._unload_vae_stage(aggressive=False)
        self._unload_planner_stage(aggressive=False)
        self._unload_t5_stage(aggressive=False)
        if self.model is not None and getattr(self.model, "diff_dec", None) is not None:
            _module_to(getattr(self.model.diff_dec, "transformer", None), "cpu")
            _module_to(getattr(self.model.diff_dec, "transformer_2", None), "cpu")
        _cleanup_cuda(aggressive=aggressive)

    def load_wan_lightx2v_loras(
        self,
        high_noise_lora_path: Optional[str] = None,
        low_noise_lora_path: Optional[str] = None,
        high_lora_scale: float = 1.0,
        low_lora_scale: float = 1.0,
    ):
        if self.model is None or getattr(self.model, "diff_dec", None) is None:
            raise RuntimeError("Bernini pipeline has no diffusion decoder to apply LoRA.")
        tag = (
            os.path.abspath(os.path.expanduser(high_noise_lora_path)) if high_noise_lora_path else "",
            os.path.abspath(os.path.expanduser(low_noise_lora_path)) if low_noise_lora_path else "",
            float(high_lora_scale),
            float(low_lora_scale),
        )
        if getattr(self, "_wan_lightx2v_lora_tag", None) == tag:
            return

        loaded = 0
        if high_noise_lora_path:
            loaded += _apply_wan_lora(
                getattr(self.model.diff_dec, "transformer", None),
                high_noise_lora_path,
                scale=high_lora_scale,
            )
        if low_noise_lora_path:
            loaded += _apply_wan_lora(
                getattr(self.model.diff_dec, "transformer_2", None),
                low_noise_lora_path,
                scale=low_lora_scale,
            )
        if loaded:
            self._wan_lightx2v_lora_tag = tag
        _cleanup_cuda(aggressive=True)

    def destroy(self, aggressive: bool = True):
        """Release model references after a one-shot run."""
        self.unload_all(aggressive=False)
        self.connector = None
        self.text_encoder = None
        self.model = None
        self.vae = None
        self.t5_tokenizer = None
        self.vit_processor = None
        self.config = None
        _cleanup_cuda(aggressive=aggressive)

    @classmethod
    def from_pretrained(
        cls,
        config_dir: str,
        ckpt: str = None,
        device="cuda",
        **config_overrides,
    ) -> "BerniniPipeline":
        config = BerniniConfig.from_pretrained(config_dir, **config_overrides)
        _localize_bernini_config(config, config_dir)
        if ckpt is None: ckpt = config_dir
        model = _load_full_bernini_model(config, ckpt)

        t5_tokenizer = AutoTokenizer.from_pretrained(
            config.t5_tokenizer_path,
            subfolder=config.t5_tokenizer_subfolder,
            trust_remote_code=True,
        )
        vit_processor = AutoProcessor.from_pretrained(
            config.processor_config_path,
            subfolder=config.processor_subfolder,
            padding_side="right",
            trust_remote_code=True,
        )

        vae = AutoencoderKLWan.from_pretrained(
            config.vae_model_path,
            subfolder=config.vae_subfolder,
            torch_dtype=torch.float32,
        )
        vae.eval()
        vae.requires_grad_(False)
        return cls(config, model, vae, t5_tokenizer, vit_processor, device)

    @torch.no_grad()
    def sample_vit_decoder(
        self,
        vit_embed,
        uncond_vit_embed,
        imgcond_vit_embed,
        vit_txt_cfg,
        sample_steps,
        vit_img_cfg=None,
        verbose=True,
    ):
        dtype = vit_embed.dtype
        if vit_img_cfg is not None and vit_txt_cfg > 1.0:
            vit_embed = torch.cat([vit_embed, uncond_vit_embed, imgcond_vit_embed], dim=1)
        elif vit_txt_cfg > 1.0:
            vit_embed = torch.cat([vit_embed, uncond_vit_embed], dim=1)

        vit_embed = (
            self.model.vit_decoder.sample(
                z=vit_embed[0],
                cfg=vit_txt_cfg,
                img_cfg=vit_img_cfg,
                num_inference_steps=sample_steps,
                verbose=verbose,
            )
            .unsqueeze(0)
            .to(dtype)
        )

        if vit_img_cfg is not None and vit_txt_cfg > 1.0:
            vit_embed = vit_embed[:, : vit_embed.shape[1] // 3, :]
        elif vit_txt_cfg > 1.0:
            vit_embed = vit_embed[:, : vit_embed.shape[1] // 2, :]

        return vit_embed

    @torch.no_grad()
    def preprocess_inputs(
        self,
        prompt,
        mllm_model,
        vae_model,
        vae_transform,
        row=None,
        vit_min_pixels: int = 3136,
        vit_max_pixels: int = 50176,
        vae_fps: int = 16,
        vit_fps: int = 2,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        video=None,
        image=None,
        images=None,
        max_duration: int = None,
        reference_image_chunk_size: int = 4,
    ):
        from bernini.data_utils import generate_unified_inputs
        # Build image/video path lists, filtering out None values.
        if images is not None:
            input_image_paths = [img for img in images if img is not None]
        elif image is not None:
            input_image_paths = [image]
        else:
            input_image_paths = []
        if video is None:
            input_video_paths = []
        elif isinstance(video, str):
            input_video_paths = [video]
        else:
            input_video_paths = [vid for vid in video if vid is not None]
        inputs_structure = generate_unified_inputs(
            prompt,
            input_image_paths=input_image_paths,
            input_video_paths=input_video_paths,
            has_video_input=bool(input_video_paths),
            output_t=num_frames,
            output_h=height,
            output_w=width,
        )
        row_data = {}
        row_data['inputs'] = inputs_structure
        if (images is not None and len(images) > 0) or image is not None or num_frames == 1:
            images = [image] if image is not None else list(images or [])
            if num_frames == 1: images.append("output_img_flag")
            images = [create_fake_image(height, width) if img == "output_img_flag" else img for img in images]
            images = [_as_pil_image(img) for img in images]
            row_data['image_embeds'] = []
            row_data['image_grid_thw'] = []
            for image_chunk in _chunked(images, reference_image_chunk_size):
                image_inputs = self.vit_processor.image_processor(
                    images=image_chunk, return_tensors="pt",
                    min_pixels=vit_min_pixels, max_pixels=vit_max_pixels
                )
                pixel_values = image_inputs['pixel_values']
                image_grid_thw = image_inputs['image_grid_thw']
                image_embeds = get_vit_features(mllm_model, pixel_values, image_grid_thw)
                row_data['image_embeds'].extend(
                    [tensor_to_bytes(embed.detach().cpu()) for embed in image_embeds]
                )
                row_data['image_grid_thw'].extend(image_grid_thw.numpy().tolist())
                del image_inputs, pixel_values, image_grid_thw, image_embeds
                _cleanup_cuda()
            # VAE
            image_tensors = [vae_transform(img) for img in images]
            image_vae_latents = []
            for img_tensor in image_tensors:
                latent = get_vae_features(vae_model, img_tensor)
                image_vae_latents.append(latent)
            row_data['image_vae_latents'] = image_vae_latents
            del image_tensors
            _cleanup_cuda()
        
        if video is not None or num_frames > 1:
            row_data["video_embeds"] = []
            row_data["video_grid_thw"] = []
            row_data["video_vae_latents"] = []
            video_meta = []
            if video is not None: 
                if isinstance(video, str):
                    video_meta.append(video)
                elif isinstance(video, list):
                    video_meta.extend(video)
            if num_frames > 1:
                if video is not None:
                    video_meta.append(video_meta[0])
                else:
                    video_meta.append('output_vid_flag')
            for video_path in video_meta:
                if video_path != "output_vid_flag":
                    duration = None
                    if row is not None and 'videos' in row:
                        for v_meta in row['videos']:
                            if v_meta.get('video_path') == video_path:
                                duration = v_meta.get('duration', None)
                                break
                    if duration is not None and max_duration is not None and duration > max_duration:
                        duration = max_duration
                    video_reader = PathVideoReader(video_path, duration=duration, crop_method='left')
                else:
                    video_reader = FakeVideoReader(
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        fps=vae_fps
                    )
                vit_idx = smart_video_nframes(
                    total_frames=video_reader.length, video_fps=video_reader.fps,
                    fps=vit_fps, frame_factor=2,
                    max_frames=num_frames, add_one=False
                )
                video_for_vit = video_reader.sample(vit_idx)
                video_inputs = self.vit_processor.video_processor(
                    videos=video_for_vit, return_tensors="pt",
                    size={'shortest_edge': vit_min_pixels, 'longest_edge': vit_max_pixels},
                )
                vid_pixel_values = video_inputs['pixel_values_videos']
                vid_grid_thw = video_inputs['video_grid_thw']
                video_embeds = get_vit_features(mllm_model, vid_pixel_values, vid_grid_thw)
                row_data['video_embeds'].extend([tensor_to_bytes(embed.detach().cpu()) for embed in video_embeds])
                row_data['video_grid_thw'].extend(vid_grid_thw.numpy().tolist())
                del video_inputs
                vae_idx = smart_video_nframes(
                    total_frames=video_reader.length, video_fps=video_reader.fps,
                    fps=vae_fps, frame_factor=4,
                    max_frames=num_frames, add_one=True
                )
                video_for_vae = video_reader.sample(vae_idx)
                video_tensor = torch.stack([vae_transform(frame) for frame in video_for_vae], dim=1)
                video_vae_latent = get_vae_features(vae_model, video_tensor)
                row_data['video_vae_latents'].append(video_vae_latent)
                del video_tensor

                _cleanup_cuda()
        
        return row_data

    def transform_inputs(
        self,
        sample,
        max_vae_frames: int = 81,
        task_name: str = "t2v",
        neg_prompt: Optional[str] = None,
        t5_neg_prompt: str = "",
        use_qwen_neg_prompt: bool = True,
    ):
        if neg_prompt is not None:
            t5_neg_prompt = neg_prompt
        rope = WanRotaryPosEmbed(
            128,
            (1, 2, 2),
            1024,
            use_src_id_rotary_emb=True,
        )
        mllm_config_path = self.config.mllm_config_path
        mllm_config_subfolder = getattr(self.config, "mllm_subfolder", None)
        vae_model_path = getattr(self.config, "vae_model_path", None)
        vae_subfolder = getattr(self.config, "vae_subfolder", None)
        vae_config_path = self.config.vae_config_path
        
        mllm_config = AutoConfig.from_pretrained(mllm_config_path, subfolder=mllm_config_subfolder)
        fake_model = SimpleNamespace(
            config=mllm_config,
            image_token_id=mllm_config.image_token_id,
            video_token_id=mllm_config.video_token_id,
        )
        position_id_func = partial(Qwen2_5_VLModel.get_rope_index, fake_model)
        
        processor = self.vit_processor
        chat_template = BerniniTemplate(
            processor.tokenizer,
            t5_tokenizer=self.t5_tokenizer
        )

        with open(vae_config_path, 'r') as f:
            vae_config = json.load(f)
        vae_latent_mean = torch.tensor(vae_config['latents_mean'], device="cpu")
        vae_latent_std = torch.tensor(vae_config['latents_std'], device="cpu")
        vae_latent_mean = vae_latent_mean.view(vae_config['z_dim'], 1, 1, 1)
        vae_latent_std = vae_latent_std.view(vae_config['z_dim'], 1, 1, 1)

        src_id_kwargs = dict(
            interpolate_src_id=getattr(self.config, "interpolate_src_id", True),
            max_trained_src_id=getattr(self.config, "max_trained_src_id", 5),
        )
        transform = partial(
            bernini_process_sample,
            processor=processor,
            chat_template=chat_template,
            position_id_func=position_id_func,
            vae_rope_func=rope,
            vae_latent_mean=vae_latent_mean,
            vae_latent_std=vae_latent_std,
            text_dropout_rate=0.0,
            img_dropout_rate=0.0,
            video_dropout_rate=0.0,
            max_vae_frames=max_vae_frames,
            source_name=task_name,
            **src_id_kwargs,
        )
        uncond_transform = partial(
            bernini_process_sample,
            processor=processor,
            chat_template=chat_template,
            position_id_func=position_id_func,
            vae_rope_func=rope,
            vae_latent_mean=vae_latent_mean,
            vae_latent_std=vae_latent_std,
            text_dropout_rate=1.0,
            img_dropout_rate=1.0,
            video_dropout_rate=1.0,
            max_vae_frames=max_vae_frames,
            source_name=task_name,
            **src_id_kwargs,
        )
        imgcond_transform = partial(
            bernini_process_sample,
            processor=processor,
            chat_template=chat_template,
            position_id_func=position_id_func,
            vae_rope_func=rope,
            vae_latent_mean=vae_latent_mean,
            vae_latent_std=vae_latent_std,
            text_dropout_rate=0.0,
            img_dropout_rate=1.0,
            video_dropout_rate=1.0,
            max_vae_frames=max_vae_frames,
            source_name=task_name,
            **src_id_kwargs,
        )
        def process_sample(sample, sample_idx):
            neg_prompt = sample.get('neg_prompt', t5_neg_prompt)
            tokenized_example = transform(sample)[0]
            imgcond_tokenized_example = imgcond_transform(sample)[0]
            uncond_tokenized_example = uncond_transform(sample, neg_prompt=neg_prompt if use_qwen_neg_prompt else "")[0]

            for k in tokenized_example:
                if isinstance(tokenized_example[k], torch.Tensor):
                    tokenized_example[k] = tokenized_example[k]
                    uncond_tokenized_example[k] = uncond_tokenized_example[k]
                    imgcond_tokenized_example[k] = imgcond_tokenized_example[k]
            
            sample.pop('inputs')
            return dict(
                uid=sample.pop('uid', f'{sample_idx:03d}').split('/')[-1].split('.')[0],
                edit_type=sample.pop('edit_type', 'unknown'),
                inputs=tokenized_example,
                uncond_inputs=uncond_tokenized_example,
                imgcond_inputs=imgcond_tokenized_example,
                **sample
            )
        
        return process_sample(sample, 0)

    @torch.no_grad()
    def sample_vit_embed(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        visual_output_token_mask=None,
        uncond_input_embeds: Optional[torch.FloatTensor] = None,
        uncond_position_ids: Optional[torch.Tensor] = None,
        uncond_attention_mask_4d: Optional[torch.Tensor] = None,
        uncond_visual_output_token_mask=None,
        imgcond_input_embeds: Optional[torch.FloatTensor] = None,
        imgcond_position_ids: Optional[torch.Tensor] = None,
        imgcond_attention_mask_4d: Optional[torch.Tensor] = None,
        imgcond_visual_output_token_mask=None,
        planning_step=3,
        vit_denoising_step=1,
        vit_txt_cfg=1.4,
        vit_img_cfg=1.2,
        progress_callback=None,
    ):  
        device = input_embeds.device
        mask_ratio_generator_infer = lambda s, totals: np.cos(math.pi / 2.0 * (s + 1) / totals)
        # Init and sample generation orders
        n_query_tokens = visual_output_token_mask.sum().detach().cpu().numpy()
        order = np.array(list(range(n_query_tokens)))
        np.random.shuffle(order)
        order = torch.Tensor(np.array(order)).to(device).long()
        mask = torch.ones(n_query_tokens).to(device)

        if position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(
                0, 1).contiguous()  # bs, dim, l -> dim, bs, l
        if uncond_position_ids.shape[1] == 3:
            uncond_position_ids = uncond_position_ids.transpose(
                0, 1).contiguous()  # bs, dim, l -> dim, bs, l
        if imgcond_position_ids.shape[1] == 3:
            imgcond_position_ids = imgcond_position_ids.transpose(
                0, 1).contiguous()  # bs, dim, l -> dim, bs, l

        def _planning_progress(current):
            if progress_callback is not None:
                progress_callback("planning", current, planning_step)

        if self.model.vit_decoder is not None:
            for step in tqdm(range(planning_step), desc=f"Sample FM+MAR clip in {planning_step} steps"):
                if self.connector is not None:
                    connector_param = next(self.connector.parameters())
                    if connector_param.device != input_embeds.device or connector_param.dtype != input_embeds.dtype:
                        self.connector.to(device=input_embeds.device, dtype=input_embeds.dtype)
                hidden_state = self.text_encoder(
                    inputs_embeds=input_embeds.clone(),
                    position_ids=position_ids.clone(),
                    attention_mask=attention_mask_4d.clone(),
                    output_hidden_states=True,
                ).hidden_states[-2]
                uncond_hidden_state = self.text_encoder(
                    inputs_embeds=uncond_input_embeds.clone(),
                    position_ids=uncond_position_ids.clone(),
                    attention_mask=uncond_attention_mask_4d.clone(),
                    output_hidden_states=True,
                ).hidden_states[-2]
                imgcond_hidden_state = self.text_encoder(
                    inputs_embeds=imgcond_input_embeds.clone(),
                    position_ids=imgcond_position_ids.clone(),
                    attention_mask=imgcond_attention_mask_4d.clone(),
                    output_hidden_states=True,
                ).hidden_states[-2]
                # extract feat from stageone to feed stagetwo
                cond_vit_embed = hidden_state[:, visual_output_token_mask, :]
                uncond_vit_embed = uncond_hidden_state[:, uncond_visual_output_token_mask, :]
                imgcond_vit_embed = imgcond_hidden_state[:, imgcond_visual_output_token_mask, :]
                pred_vit_embed_mllm = self.connector.for_vit(cond_vit_embed)
                uncond_pred_vit_embed_mllm = self.connector.for_vit(uncond_vit_embed)
                imgcond_pred_vit_embed_mllm = self.connector.for_vit(imgcond_vit_embed)

                # mask ratio for the next round, following MaskGIT and MAGE.
                mask_ratio = mask_ratio_generator_infer(step, planning_step)
                mask_len = torch.Tensor([np.floor(n_query_tokens * mask_ratio)]).to(device)
                # masks out at least one for the next iteration
                mask_len = torch.maximum(
                    torch.Tensor([1]).cuda(), torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len)
                )
                # get masking for next iteration
                mask_next = torch.zeros_like(mask)  # zero init
                mask_next = torch.scatter(
                    mask_next,
                    dim=-1,
                    index=order[: mask_len.long()],  
                    src=torch.ones_like(mask),
                ).bool()
                if step >= planning_step - 1:
                    mask_to_pred = mask.bool()  # Predict the left mask tokens
                else:
                    mask_to_pred = torch.logical_xor(mask.bool(), mask_next)
                mask = mask_next

                if mask_to_pred.nonzero(as_tuple=True)[0].sum() == 0:
                    _planning_progress(step + 1)
                    continue 
                cond_pred_vit_embed = pred_vit_embed_mllm[:, mask_to_pred.nonzero(as_tuple=True)[0]]
                uncond_pred_vit_embed = uncond_pred_vit_embed_mllm[:, mask_to_pred.nonzero(as_tuple=True)[0]]
                imgcond_pred_vit_embed = imgcond_pred_vit_embed_mllm[:, mask_to_pred.nonzero(as_tuple=True)[0]]
                cur_pred_vit_embed = self.sample_vit_decoder(
                    vit_embed=cond_pred_vit_embed,
                    uncond_vit_embed=uncond_pred_vit_embed,
                    imgcond_vit_embed=imgcond_pred_vit_embed,
                    vit_txt_cfg=vit_txt_cfg,
                    vit_img_cfg=vit_img_cfg,
                    sample_steps=vit_denoising_step,
                    verbose=False,
                )

                all_target_vit_embed = input_embeds[:, visual_output_token_mask, :]
                cur_pred_vit_embed = cur_pred_vit_embed.to(
                    device=all_target_vit_embed.device,
                    dtype=all_target_vit_embed.dtype,
                )
                all_target_vit_embed[:, mask_to_pred.nonzero(as_tuple=True)[0]] = cur_pred_vit_embed
                input_embeds[:, visual_output_token_mask] = all_target_vit_embed
                uncond_input_embeds[:, uncond_visual_output_token_mask] = all_target_vit_embed
                imgcond_input_embeds[:, imgcond_visual_output_token_mask] = all_target_vit_embed
                _planning_progress(step + 1)
        else:
            _planning_progress(planning_step)

        pred_vit_embed_diff = input_embeds[:, visual_output_token_mask, :]

        outputs = self.text_encoder(
            inputs_embeds=input_embeds.clone(),
            position_ids=position_ids.clone(),
            attention_mask=attention_mask_4d.clone(),
            output_hidden_states=True,
        )
        uncond_outputs = self.text_encoder(
            inputs_embeds=uncond_input_embeds.clone(),
            position_ids=uncond_position_ids.clone(),
            attention_mask=uncond_attention_mask_4d.clone(),
            output_hidden_states=True,
        )
        cond_outputs = self.model.feat_from_planner_to_renderer(
            hidden_states=outputs.hidden_states[-2],
            visual_output_mask=visual_output_token_mask, 
            tgt_vit_mask=None,
            inference=True
        )
        uncond_outputs = self.model.feat_from_planner_to_renderer(
            hidden_states=uncond_outputs.hidden_states[-2],
            visual_output_mask=uncond_visual_output_token_mask, 
            tgt_vit_mask=None,
            inference=True
        )

        if self.model.feature_type_from_stage_one in ["masked_tgt_embed_with_qwen_txt_tokens"]:
            cond_embeds_wotxt_wovit = uncond_outputs['diff_mllm_contexts']
            cond_embeds_wtxt_wvit = cond_outputs['diff_mllm_contexts']
            cond_embeds_wtxt_wovit = None
            cond_embeds_wotxt_wvit = None
        else:
            uncond_cond_embeds = uncond_outputs['diff_mllm_contexts']
            diff_mllm_context_txt_mask = uncond_outputs['diff_mllm_context_txt_mask']
            cond_embeds_wotxt_wovit = uncond_cond_embeds[:, diff_mllm_context_txt_mask] 
            diff_mllm_context_txt_mask = cond_outputs['diff_mllm_context_txt_mask']
            diff_mllm_context_vit_mask = cond_outputs['diff_mllm_context_vit_mask']
            cond_embeds_wtxt_wvit = cond_outputs['diff_mllm_contexts']
            cond_embeds_wtxt_wovit = cond_embeds_wtxt_wvit[:, diff_mllm_context_txt_mask]
            cond_embeds_wotxt_wvit = cond_embeds_wtxt_wvit[:, diff_mllm_context_vit_mask]

        return dict(
            cond_embeds_wtxt_wvit=cond_embeds_wtxt_wvit,
            cond_embeds_wtxt_wovit=cond_embeds_wtxt_wovit,
            cond_embeds_wotxt_wvit=cond_embeds_wotxt_wvit,
            cond_embeds_wotxt_wovit=cond_embeds_wotxt_wovit,
            pred_vit_embed=pred_vit_embed_diff
        )

    @torch.no_grad()
    def __call__(
        self,
        task_name: str,
        prompt: str,
        *,
        neg_prompt: str = "",
        num_frames: int = 81,
        max_image_size: int = 624,
        height: int = 480,
        width: int = 832,
        video=None,
        image=None,
        images=None,
        num_inference_steps: int = 40,
        guidance_mode: str = "rv2v",
        omega_vid: float = 3.0, 
        omega_img: float = 3.0, 
        omega_txt: float = 4.0, 
        omega_tgt: float = 4.0, 
        omega_scale: float = 0.75,
        planning_step: int = 25,
        vit_txt_cfg: float = 1.4,
        vit_img_cfg: float = 1.2,
        vit_denoising_step: int = 3,
        flow_shift: float = 5.0,
        seed: int = 42,
        fps: int = 16,
        eta: float = 0.5,
        norm_threshold=(50.0, 50.0),
        momentum: float = -0.5,
        system_prompt: str = "",
        output_path: str = "output.mp4",
        write_output: bool = True,
        use_truncate: bool = True,
        max_sequence_length: int = 512,
        sequential_unload: bool = True,
        vae_tiling: bool = True,
        vae_slicing: bool = True,
        reference_image_chunk_size: int = 4,
        unload_after_run: bool = False,
        progress_callback=None,
    ):
        """Generate one clip and write it to `output_path`.

        `video` drives video editing, `image` a single-image edit, `images` a
        list of reference images; the output size follows the source video or
        single image, otherwise `height` / `width`.

        With `write_output=False` the decode/save step is skipped (used by the
        redundant ranks of an Ulysses group) and ``None`` is returned.
        """
        device = self.device
        # NumPy's legacy seeding API only accepts uint32 seeds.
        seed = int(seed) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Resolve fps aliases: fps sets default for both vae_fps and vit_fps
        vae_fps = fps
        vit_fps = fps // 8  # default vit_fps is 1/8 of vae_fps
        raw_prompt = _prompt_clean(prompt)
        t5_prompt = _prompt_clean(system_prompt + raw_prompt)
        logger.info("prompt: %s", t5_prompt)
        # ---- encode visual conditions on the VAE ----
        self._load_vae_stage(enable_tiling=vae_tiling, enable_slicing=vae_slicing)
        self._load_planner_stage()
        vae_transform = VAEVideoTransform(
            max_image_size=max_image_size,
            min_image_size=240,
            image_stride=16,
        )
        sample = self.preprocess_inputs(
            raw_prompt,
            mllm_model=self.model.mllm,
            vae_model=self.vae,
            vae_transform=vae_transform,
            num_frames=num_frames,
            height=height,
            width=width,
            video=video,
            image=image,
            images=images,
            vit_fps=vit_fps,
            vae_fps=vae_fps,
            reference_image_chunk_size=reference_image_chunk_size,
        )
        if progress_callback is not None:
            progress_callback("preprocess", 1, 1)
        self._unload_vae_stage(aggressive=sequential_unload)
        input_dict = self.transform_inputs(
            sample,
            num_frames,
            task_name=task_name,
            neg_prompt=neg_prompt,
        )

        def _move_to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, dict):
                return {k: _move_to_device(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_move_to_device(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_move_to_device(v) for v in obj)
            return obj

        input_dict = _move_to_device(input_dict)
        inputs = input_dict['inputs']
        uncond_inputs = input_dict['uncond_inputs']
        imgcond_inputs = input_dict['imgcond_inputs']

        input_embeds = self.model.format_mllm_inputs_embeds(
            input_ids=inputs['input_ids'],
            visual_embeds=inputs['visual_embeds'],
            visual_input_mask=inputs['visual_input_token_mask'],
            visual_output_mask=inputs['visual_output_token_mask'],
        ).to(self.weight_dtype)
        uncond_input_embeds = self.model.format_mllm_inputs_embeds(
            input_ids=uncond_inputs['input_ids'],
            visual_embeds=uncond_inputs['visual_embeds'],
            visual_input_mask=uncond_inputs['visual_input_token_mask'],
            visual_output_mask=uncond_inputs['visual_output_token_mask'],
        ).to(self.weight_dtype)
        imgcond_input_embeds = self.model.format_mllm_inputs_embeds(
            input_ids=imgcond_inputs['input_ids'],
            visual_embeds=imgcond_inputs['visual_embeds'],
            visual_input_mask=imgcond_inputs['visual_input_token_mask'],
            visual_output_mask=imgcond_inputs['visual_output_token_mask'],
        ).to(self.weight_dtype)

        post_process_out = self.model.post_process_input_embeds(
                input_embeds.unsqueeze(0), 
                inputs['visual_output_token_mask'], 
                tgt_vit_mask=None,
                inference=True
            )
        inputs_embed = post_process_out['input_embeds']
        uncond_post_process_out = self.model.post_process_input_embeds(
                uncond_input_embeds.unsqueeze(0), 
                uncond_inputs['visual_output_token_mask'], 
                tgt_vit_mask=None,
                inference=True
            )
        uncond_inputs_embed = uncond_post_process_out['input_embeds']
        imgcond_post_process_out = self.model.post_process_input_embeds(
                imgcond_input_embeds.unsqueeze(0), 
                imgcond_inputs['visual_output_token_mask'], 
                tgt_vit_mask=None,
                inference=True
            )
        imgcond_inputs_embed = imgcond_post_process_out['input_embeds']
        ret = self.sample_vit_embed(
            input_embeds=inputs_embed,
            attention_mask_4d=inputs['attention_mask_4d'].unsqueeze(0),
            position_ids=inputs['position_ids'].unsqueeze(0),
            visual_output_token_mask=inputs['visual_output_token_mask'],
            uncond_input_embeds=uncond_inputs_embed,
            uncond_position_ids=uncond_inputs['position_ids'].unsqueeze(0),
            uncond_attention_mask_4d=uncond_inputs['attention_mask_4d'].unsqueeze(0),
            uncond_visual_output_token_mask=uncond_inputs['visual_output_token_mask'],
            imgcond_input_embeds=imgcond_inputs_embed,
            imgcond_position_ids=imgcond_inputs['position_ids'].unsqueeze(0),
            imgcond_attention_mask_4d=imgcond_inputs['attention_mask_4d'].unsqueeze(0),
            imgcond_visual_output_token_mask=imgcond_inputs['visual_output_token_mask'],
            planning_step=planning_step,
            vit_txt_cfg=vit_txt_cfg,
            vit_img_cfg=vit_img_cfg,
            vit_denoising_step=vit_denoising_step,
            progress_callback=progress_callback,
        )
        
        cond_embeds_wtxt_wvit = ret['cond_embeds_wtxt_wvit']
        cond_embeds_wtxt_wovit = ret['cond_embeds_wtxt_wovit']
        cond_embeds_wotxt_wvit = ret['cond_embeds_wotxt_wvit']
        cond_embeds_wotxt_wovit = ret['cond_embeds_wotxt_wovit']

        self._unload_planner_stage(aggressive=sequential_unload)

        self._load_t5_stage()
        t5_input_ids, t5_attention_mask = _get_t5_text_ids(
            t5_prompt, self.t5_tokenizer,
        )
        t5_embeds = self.model.get_t5_text_embeddings_sample(
            t5_input_ids.to(device), t5_attention_mask.to(device)
        )

        neg_prompt_ids, neg_prompt_attention_mask = _get_t5_text_ids(
            _prompt_clean(neg_prompt),
            self.t5_tokenizer,
        )
        neg_t5_embeds = self.model.get_t5_text_embeddings_sample(neg_prompt_ids.to(device), neg_prompt_attention_mask.to(device))
        if progress_callback is not None:
            progress_callback("text_encode", 1, 1)
        
        cond_embeds_wtxt_wvit = torch.cat([t5_embeds, cond_embeds_wtxt_wvit], dim=1)
        if cond_embeds_wtxt_wovit is not None:
            cond_embeds_wtxt_wovit = torch.cat([t5_embeds, cond_embeds_wtxt_wovit], dim=1)
        if cond_embeds_wotxt_wvit is not None:
            cond_embeds_wotxt_wvit = torch.cat([neg_t5_embeds, cond_embeds_wotxt_wvit], dim=1)
        cond_embeds_wotxt_wovit = torch.cat([neg_t5_embeds, cond_embeds_wotxt_wovit], dim=1)
        self._unload_t5_stage(aggressive=sequential_unload)

        def is_image_vae_shape(shape):
            shape = shape.tolist() if isinstance(shape, torch.Tensor) else shape
            return shape[0] == 1 or (len(shape) > 1 and shape[1] == 1)

        def pad_and_truncate_feat(feat, max_sequence_length=max_sequence_length, truncate=use_truncate):
            if feat is None:
                return None
            if feat.shape[1] < max_sequence_length:
                feat = torch.cat(
                    [feat, feat.new_zeros((1, max_sequence_length-feat.shape[1], feat.shape[-1]))],
                    dim=1
                )
            if truncate and feat.shape[1] > max_sequence_length:
                feat = feat[:, :max_sequence_length, :]
            return feat

        if max_sequence_length > 0:
            cond_embeds_wtxt_wvit = pad_and_truncate_feat(cond_embeds_wtxt_wvit)
            cond_embeds_wotxt_wovit = pad_and_truncate_feat(cond_embeds_wotxt_wovit)
            if cond_embeds_wtxt_wovit is not None:
                cond_embeds_wtxt_wovit = pad_and_truncate_feat(cond_embeds_wtxt_wovit)
            if cond_embeds_wotxt_wvit is not None:
                cond_embeds_wotxt_wvit = pad_and_truncate_feat(cond_embeds_wotxt_wvit)
        
        pos = 0
        all_vae_latents = inputs['input_vae_latents']
        all_vae_rope = inputs['input_vae_rope']
        src_image_vae_latents, src_image_vae_shapes, src_image_vae_rope = [], [], []
        src_video_vae_latents, src_video_vae_shapes, src_video_vae_rope = [], [], []
        for i, shape in enumerate(inputs['input_vae_shape'][:-1]):
            vae_len = shape[1] * shape[2] // 4 * shape[0]
            cur_vae_rope = all_vae_rope[pos:pos+vae_len]
            cur_vae_latent = all_vae_latents[pos:pos+vae_len]
            if is_image_vae_shape(shape):
                src_image_vae_latents.append(cur_vae_latent)
                src_image_vae_shapes.append(shape)
                src_image_vae_rope.append(cur_vae_rope)
            else:
                src_video_vae_latents.append(cur_vae_latent)
                src_video_vae_shapes.append(shape)
                src_video_vae_rope.append(cur_vae_rope)
            pos += vae_len

        # Diffusion dimensions are derived from the transformed target VAE shape
        target_vae_shape = inputs['input_vae_shape'][-1]
        target_t = int(target_vae_shape[0])
        num_frames = min(num_frames, 1 + (target_t - 1) * 4)
        if height is None or height <= 0:
            height = int(target_vae_shape[1]) * 8
        if width is None or width <= 0:
            width = int(target_vae_shape[2]) * 8

        _cleanup_cuda(aggressive=sequential_unload)
        latents = self.model.diff_dec.sample_bernini_wvitcfg(
            prompt_embeds_wtxt_wvit=cond_embeds_wtxt_wvit.to(self.weight_dtype),
            prompt_embeds_wtxt_wovit=cond_embeds_wtxt_wovit.to(self.weight_dtype) if cond_embeds_wtxt_wovit is not None else None,
            prompt_embeds_wotxt_wvit=cond_embeds_wotxt_wvit.to(self.weight_dtype) if cond_embeds_wotxt_wvit is not None else None,
            prompt_embeds_wotxt_wovit=cond_embeds_wotxt_wovit.to(self.weight_dtype),
            source_image_vae_latents=None if len(src_image_vae_latents) == 0 else torch.cat(src_image_vae_latents, dim=0),
            source_image_vae_rope=None if len(src_image_vae_rope) == 0 else torch.cat(src_image_vae_rope, dim=0),
            source_video_vae_latents=None if len(src_video_vae_latents) == 0 else torch.cat(src_video_vae_latents, dim=0),
            source_video_vae_rope=None if len(src_video_vae_rope) == 0 else torch.cat(src_video_vae_rope, dim=0),
            num_frames=num_frames,
            width=width,
            height=height,
            omega_txt=omega_txt,
            omega_img=omega_img,
            omega_vid=omega_vid,
            omega_tgt=omega_tgt,
            omega_scale=omega_scale,
            num_inference_steps=num_inference_steps,
            guidance_mode=guidance_mode,
            flow_shift=flow_shift,
            seed=seed,
            device=device,
            progress_callback=progress_callback,
        )

        if not write_output:
            if unload_after_run:
                self.destroy(aggressive=True)
            return None

        self._load_vae_stage(enable_tiling=vae_tiling, enable_slicing=vae_slicing)
        output = _vae_decode(self.vae, latents)
        if progress_callback is not None:
            progress_callback("decode", 1, 1)
        self._unload_vae_stage(aggressive=sequential_unload)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_output(output, output_path, fps=vae_fps)
        if progress_callback is not None:
            progress_callback("save", 1, 1)
        logger.info("saved -> %s  (%d frames, %dx%d)", output_path, output.shape[0], height, width)
        if unload_after_run:
            self.destroy(aggressive=True)
        return output_path
