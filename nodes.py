import json
import gc
import os
import re
import sys
import uuid
from typing import Iterable

try:
    import folder_paths
except Exception:  # pragma: no cover - allows import checks outside ComfyUI.
    folder_paths = None


QUALITY_PRESETS = ["smoke_test", "standard", "quality"]
MEMORY_MODES = ["keep_loaded_batch", "low_vram_unload"]
ACCELERATION_MODES = ["wan2.2_lightx2v_4step", "none"]
DEFAULT_MODEL_DIR = "Bernini-Diffusers-qint8"
MAX_NUMPY_SEED = 0xFFFFFFFF
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
_PIPELINE_CACHE = {}


def _prompt_widget(default: str = ""):
    return (
        "STRING",
        {
            "multiline": True,
            "default": default,
            "tooltip": "Text prompt or edit instruction.",
        },
    )


def _negative_prompt_widget():
    return (
        "STRING",
        {
            "multiline": True,
            "default": "low quality, blurry, distorted, text, watermark",
            "tooltip": "Negative prompt.",
        },
    )


def _size_widgets(default_height: int = 512, default_width: int = 512):
    return {
        "width": (
            "INT",
            {
                "default": default_width,
                "min": 64,
                "max": 2048,
                "step": 16,
                "tooltip": "Output width in pixels. Larger values increase VRAM use and runtime.",
            },
        ),
        "height": (
            "INT",
            {
                "default": default_height,
                "min": 64,
                "max": 2048,
                "step": 16,
                "tooltip": "Output height in pixels. Larger values increase VRAM use and runtime.",
            },
        ),
    }


def _video_size_widgets():
    return {
        "width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16, "tooltip": "0 follows the source video width."}),
        "height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16, "tooltip": "0 follows the source video height."}),
    }


def _common_widgets(default_steps: int = 4, default_seed: int = 42, default_quality_preset: str = "standard"):
    return {
        "steps": ("INT", {"default": default_steps, "min": 1, "max": 100, "step": 1, "tooltip": "Diffusion steps."}),
        "seed": (
            "INT",
            {
                "default": default_seed,
                "min": 0,
                "max": MAX_NUMPY_SEED,
                "step": 1,
                "control_after_generate": "randomize",
                "tooltip": "Random seed. Values are clamped to NumPy's valid 0 to 2**32-1 range before inference.",
            },
        ),
        "quality_preset": (
            QUALITY_PRESETS,
            {
                "default": default_quality_preset,
                "tooltip": "smoke_test is for connectivity tests; standard/quality spend more planning time for better output.",
            },
        ),
        "acceleration": (
            ACCELERATION_MODES,
            {
                "default": "wan2.2_lightx2v_4step",
                "tooltip": "Auto-load the Wan2.2 T2V LightX2V 4-step LoRA pair from ComfyUI/models/loras when available.",
            },
        ),
        "memory_mode": (
            MEMORY_MODES,
            {
                "default": "keep_loaded_batch",
                "tooltip": "keep_loaded_batch keeps the Bernini pipeline cached for faster consecutive jobs; low_vram_unload frees memory after each run.",
            },
        ),
    }


def _video_widgets(default_frames: int = 9):
    return {
        "num_frames": (
            "INT",
            {
                "default": default_frames,
                "min": 5,
                "max": 129,
                "step": 4,
                "tooltip": "Number of output frames. Higher values increase VRAM use and runtime.",
            },
        ),
        "fps": (
            "INT",
            {
                "default": 16,
                "min": 1,
                "max": 60,
                "step": 1,
                "tooltip": "Frame rate written into the returned video.",
            },
        ),
    }


def _memory_settings(memory_mode: str):
    keep_model_loaded = str(memory_mode or "keep_loaded_batch") != "low_vram_unload"
    return {
        "sequential_unload": True,
        "vae_tiling": True,
        "vae_slicing": True,
        "keep_model_loaded": keep_model_loaded,
    }


def _quality_settings(quality_preset: str):
    preset = str(quality_preset or "standard")
    if preset == "quality":
        return {"planning_step": 35, "vit_denoising_step": 5}
    if preset == "standard":
        return {"planning_step": 25, "vit_denoising_step": 5}
    return {"planning_step": 2, "vit_denoising_step": 1}


def _normalize_seed(seed) -> int:
    return int(seed) % (MAX_NUMPY_SEED + 1)


class _ComfyProgressTracker:
    _STAGE_RANGES = {
        "preprocess": (20, 30),
        "planning": (30, 55),
        "text_encode": (55, 60),
        "diffusion": (60, 90),
        "decode": (90, 98),
        "save": (98, 100),
    }

    def __init__(self):
        self._pbar = None
        self._last_value = 0
        try:
            from comfy.utils import ProgressBar

            self._pbar = ProgressBar(100)
        except Exception:
            self._pbar = None
        self.update_absolute(1)

    def update_absolute(self, value):
        if self._pbar is None:
            return
        value = max(0, min(100, int(round(value))))
        if value < self._last_value:
            return
        self._last_value = value
        self._pbar.update_absolute(value, total=100)

    def update_stage(self, stage, current=1, total=1):
        start, end = self._STAGE_RANGES.get(str(stage), (self._last_value, self._last_value))
        total = max(1, int(total or 1))
        current = max(0, min(total, int(current or 0)))
        span = end - start
        self.update_absolute(start + span * (current / total))

    def finish(self):
        self.update_absolute(100)


def _models_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        return folder_paths.models_dir
    return os.path.join(os.getcwd(), "models")


_T2V_LORA_CANDIDATES = {
    "high": [
        "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
        "Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
    ],
    "low": [
        "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors",
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
        "Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
    ],
}

def _find_lora_file(candidates: list[str]) -> str:
    loras_dir = os.path.join(_models_dir(), "loras")
    for name in candidates:
        candidate = os.path.join(loras_dir, name)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return ""


def _resolve_acceleration_loras(task_type: str, acceleration: str):
    if str(acceleration or "none") != "wan2.2_lightx2v_4step":
        return {"high_noise_lora_path": "", "low_noise_lora_path": ""}

    # Keep every Bernini task on the same acceleration adapter so T2I/I2I/V2V
    # share one cached pipeline instead of cold-loading per task family.
    candidates = _T2V_LORA_CANDIDATES
    high_path = _find_lora_file(candidates["high"])
    low_path = _find_lora_file(candidates["low"])
    if not high_path or not low_path:
        print(
            "[RH Bernini Full] Wan2.2 T2V LightX2V acceleration requested, but matching high/low LoRA files "
            "were not found under ComfyUI/models/loras. Running without acceleration."
        )
        return {"high_noise_lora_path": "", "low_noise_lora_path": ""}
    return {
        "high_noise_lora_path": high_path,
        "low_noise_lora_path": low_path,
        "high_lora_scale": 1.0,
        "low_lora_scale": 1.0,
    }


def _output_dir() -> str:
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.join(os.getcwd(), "output")


def _split_paths(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items: Iterable = value
    else:
        items = re.split(r"[\n,]+", str(value))
    return [str(item).strip() for item in items if str(item).strip()]


def _resolve_model_dir(model_dir: str) -> str:
    model_dir = str(model_dir or "").strip()
    candidates = []
    if model_dir:
        candidates.append(model_dir)
        if not os.path.isabs(model_dir):
            models_root = _models_dir()
            candidates.append(os.path.join(models_root, model_dir))
            candidates.append(os.path.join(models_root, "diffusers", model_dir))
    for candidate in candidates:
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isdir(candidate):
            _validate_bernini_model_dir(candidate)
            return candidate
    raise FileNotFoundError(
        "Bernini model directory not found. Expected a full Bernini-Diffusers "
        f"directory under ComfyUI/models/diffusers. Checked: {candidates}"
    )


def _validate_bernini_model_dir(model_dir: str):
    normalized = model_dir.replace("\\", "/").lower()
    if "bernini-r" in normalized or "bernini_r" in normalized:
        raise ValueError("Bernini-R paths are not allowed. Use ByteDance/Bernini-Diffusers only.")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing full Bernini config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "bernini":
        raise ValueError(
            f"Unsupported model_type={config.get('model_type')!r}. "
            "This plugin only loads the full Bernini-Diffusers model."
        )

    required = ["bernini", "mllm", "t5_text_encoder", "t5_tokenizer", "vae"]
    missing = [name for name in required if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        raise FileNotFoundError(f"Incomplete Bernini-Diffusers directory, missing: {', '.join(missing)}")


def _safe_prefix(prefix: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix or "bernini"))
    return prefix.strip("._") or "bernini"


def _temp_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "get_temp_directory"):
        root = folder_paths.get_temp_directory()
    else:
        root = os.path.join(_output_dir(), "bernini_temp")
    os.makedirs(root, exist_ok=True)
    return root


def _temp_path(prefix: str, ext: str) -> str:
    ext = ext if str(ext).startswith(".") else f".{ext}"
    return os.path.join(_temp_dir(), f"{_safe_prefix(prefix)}_{uuid.uuid4().hex}{ext}")


def _image_batch_to_paths(image, prefix: str) -> list[str]:
    if image is None:
        return []
    if isinstance(image, str):
        return _split_paths(image)

    import numpy as np
    import torch
    from PIL import Image

    if not torch.is_tensor(image):
        raise TypeError(f"Expected ComfyUI IMAGE tensor, got {type(image).__name__}")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected IMAGE tensor with shape [B,H,W,C], got {tuple(image.shape)}")

    paths = []
    for index, frame in enumerate(image):
        array = frame.detach().cpu().float().clamp(0, 1).numpy()
        if array.shape[-1] > 3:
            array = array[..., :3]
        array = np.rint(array * 255.0).astype(np.uint8)
        path = _temp_path(f"{prefix}_{index:02d}", ".png")
        Image.fromarray(array, mode="RGB").save(path)
        paths.append(path)
    return paths


def _image_from_path(path: str):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _video_to_path(video, prefix: str) -> str:
    if video is None:
        return ""
    if isinstance(video, str):
        return video

    source = None
    if hasattr(video, "get_stream_source"):
        source = video.get_stream_source()
        if isinstance(source, str) and os.path.exists(source):
            return source

    path = _temp_path(prefix, ".mp4")
    if hasattr(video, "save_to"):
        from comfy_api.latest import Types

        video.save_to(path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
        return path

    if source is not None and hasattr(source, "read"):
        with open(path, "wb") as f:
            f.write(source.read())
        return path

    raise TypeError(f"Expected ComfyUI VIDEO input, got {type(video).__name__}")


def _video_from_path(path: str):
    from comfy_api.input_impl import VideoFromFile

    return VideoFromFile(path)


def _get_pipeline(
    model_dir: str,
    device,
    use_unipc: bool,
    use_src_tgt_id: bool,
    high_noise_lora_path: str = "",
    low_noise_lora_path: str = "",
    high_lora_scale: float = 1.0,
    low_lora_scale: float = 1.0,
):
    _ensure_local_bernini_package()
    from bernini.pipeline import BerniniPipeline

    key = (
        model_dir,
        str(device),
        bool(use_unipc),
        bool(use_src_tgt_id),
        os.path.abspath(high_noise_lora_path) if high_noise_lora_path else "",
        os.path.abspath(low_noise_lora_path) if low_noise_lora_path else "",
        float(high_lora_scale),
        float(low_lora_scale),
    )
    pipeline = _PIPELINE_CACHE.get(key)
    if pipeline is None:
        pipeline = BerniniPipeline.from_pretrained(
            model_dir,
            device=device,
            use_unipc=bool(use_unipc),
            use_src_id_rotary_emb=bool(use_src_tgt_id),
        )
        if high_noise_lora_path or low_noise_lora_path:
            pipeline.load_wan_lightx2v_loras(
                high_noise_lora_path=high_noise_lora_path,
                low_noise_lora_path=low_noise_lora_path,
                high_lora_scale=high_lora_scale,
                low_lora_scale=low_lora_scale,
            )
        _PIPELINE_CACHE[key] = pipeline
    return key, pipeline


def _ensure_local_bernini_package():
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    existing = sys.modules.get("bernini")
    existing_file = os.path.abspath(getattr(existing, "__file__", "")) if existing is not None else ""
    if existing_file and not existing_file.startswith(plugin_dir):
        for name in list(sys.modules):
            if name == "bernini" or name.startswith("bernini."):
                del sys.modules[name]


def _run_bernini(
    task_type,
    prompt,
    num_frames,
    height,
    width,
    num_inference_steps,
    seed,
    output_prefix,
    source_video_path="",
    source_image_path="",
    reference_image_paths="",
    negative_prompt=DEFAULT_NEG_PROMPT,
    system_prompt="",
    guidance_mode="vae_txt_vit_wapg",
    max_image_size=848,
    fps=16,
    omega_txt=4.0,
    omega_img=1.0,
    omega_vid=1.0,
    omega_tgt=0.5,
    omega_scale=0.8,
    planning_step=25,
    vit_denoising_step=5,
    vit_txt_cfg=1.2,
    vit_img_cfg=1.0,
    flow_shift=5.0,
    reference_image_chunk_size=2,
    sequential_unload=True,
    vae_tiling=True,
    vae_slicing=True,
    keep_model_loaded=False,
    acceleration="wan2.2_lightx2v_4step",
    use_unipc=True,
    use_src_tgt_id=True,
):
    import torch

    if not str(prompt or "").strip():
        raise ValueError("prompt is required")

    progress = _ComfyProgressTracker()
    seed = _normalize_seed(seed)
    progress.update_absolute(3)
    resolved_model_dir = _resolve_model_dir(DEFAULT_MODEL_DIR)
    lora_settings = _resolve_acceleration_loras(task_type, acceleration)
    progress.update_absolute(8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    progress.update_absolute(10)
    cache_key, pipeline = _get_pipeline(
        resolved_model_dir,
        device=device,
        use_unipc=use_unipc,
        use_src_tgt_id=use_src_tgt_id,
        **lora_settings,
    )
    progress.update_absolute(20)

    videos = _split_paths(source_video_path)
    refs = _split_paths(reference_image_paths)
    source_image = _split_paths(source_image_path)
    image = source_image[0] if source_image else None
    video = videos if len(videos) > 1 else (videos[0] if videos else None)
    images = refs or None
    output_path = _temp_path(output_prefix, ".png" if int(num_frames) <= 1 else ".mp4")

    try:
        pipeline(
            task_type,
            prompt,
            neg_prompt=negative_prompt,
            num_frames=int(num_frames),
            max_image_size=int(max_image_size),
            height=int(height),
            width=int(width),
            video=video,
            image=image,
            images=images,
            num_inference_steps=int(num_inference_steps),
            guidance_mode=guidance_mode,
            omega_vid=float(omega_vid),
            omega_img=float(omega_img),
            omega_txt=float(omega_txt),
            omega_tgt=float(omega_tgt),
            omega_scale=float(omega_scale),
            planning_step=int(planning_step),
            vit_txt_cfg=float(vit_txt_cfg),
            vit_img_cfg=float(vit_img_cfg),
            vit_denoising_step=int(vit_denoising_step),
            flow_shift=float(flow_shift),
            seed=int(seed),
            fps=int(fps),
            system_prompt=system_prompt,
            output_path=output_path,
            sequential_unload=bool(sequential_unload),
            vae_tiling=bool(vae_tiling),
            vae_slicing=bool(vae_slicing),
            reference_image_chunk_size=int(reference_image_chunk_size),
            unload_after_run=not bool(keep_model_loaded),
            progress_callback=progress.update_stage,
        )
        progress.finish()
    finally:
        if not keep_model_loaded:
            cached = _PIPELINE_CACHE.pop(cache_key, None)
            target = cached if cached is not None else pipeline
            if target is not None:
                if hasattr(target, "destroy"):
                    target.destroy(aggressive=True)
                else:
                    target.unload_all(aggressive=True)
                del target
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            pipeline = None

    return output_path


class _RHBerniniCompactBase:
    CATEGORY = "RunningHub/Bernini"
    FUNCTION = "generate"
    OUTPUT_NODE = False


class RHBerniniFullTextToImage(_RHBerniniCompactBase):
    DESCRIPTION = "Full Bernini text-to-image using Bernini-Diffusers only."
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": _prompt_widget("A clean studio product image of a red ceramic teapot on a white table, soft daylight"),
            "negative_prompt": _negative_prompt_widget(),
        }
        required.update(_size_widgets())
        required.update(_common_widgets(default_steps=4, default_seed=1234))
        return {"required": required}

    def generate(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        seed,
        quality_preset,
        acceleration="wan2.2_lightx2v_4step",
        memory_mode="keep_loaded_batch",
    ):
        settings = _quality_settings(quality_preset)
        settings.update(_memory_settings(memory_mode))
        output_path = _run_bernini(
            task_type="t2i",
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=1,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed,
            output_prefix="rh_bernini_t2i",
            max_image_size=max(int(width), int(height), 512),
            guidance_mode="vae_txt_vit_wapg",
            omega_txt=4.0,
            omega_img=1.0,
            omega_vid=1.0,
            omega_tgt=0.5,
            omega_scale=0.8,
            acceleration=acceleration,
            **settings,
        )
        return (_image_from_path(output_path),)


class RHBerniniFullImageToImage(_RHBerniniCompactBase):
    DESCRIPTION = "Full Bernini image-to-image editing using Bernini-Diffusers only."
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE", {"tooltip": "Source image from a ComfyUI image output."}),
            "prompt": _prompt_widget("Turn the source image into a polished flat illustration."),
            "negative_prompt": _negative_prompt_widget(),
        }
        required.update(_size_widgets())
        required.update(_common_widgets(default_steps=4, default_seed=2345))
        return {"required": required}

    def generate(
        self,
        image,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        seed,
        quality_preset,
        acceleration="wan2.2_lightx2v_4step",
        memory_mode="keep_loaded_batch",
    ):
        settings = _quality_settings(quality_preset)
        settings.update(_memory_settings(memory_mode))
        source_paths = _image_batch_to_paths(image, "rh_bernini_i2i_source")
        if not source_paths:
            raise ValueError("image is required")
        output_path = _run_bernini(
            task_type="i2i",
            prompt=prompt,
            source_image_path=source_paths[0],
            negative_prompt=negative_prompt,
            num_frames=1,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed,
            output_prefix="rh_bernini_i2i",
            max_image_size=max(int(width), int(height), 512),
            guidance_mode="vae_txt_vit_wapg",
            omega_txt=4.0,
            omega_img=1.2,
            omega_vid=1.0,
            omega_tgt=0.5,
            omega_scale=0.8,
            acceleration=acceleration,
            **settings,
        )
        return (_image_from_path(output_path),)


class RHBerniniFullTextToVideo(_RHBerniniCompactBase):
    DESCRIPTION = "Full Bernini text-to-video using Bernini-Diffusers only."
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": _prompt_widget("A short smooth video of a red ceramic teapot rotating slowly on a white table."),
            "negative_prompt": _negative_prompt_widget(),
        }
        required.update(_video_widgets(default_frames=9))
        required.update(_size_widgets())
        required.update(_common_widgets(default_steps=4, default_seed=3456))
        return {"required": required}

    def generate(
        self,
        prompt,
        negative_prompt,
        num_frames,
        fps,
        width,
        height,
        steps,
        seed,
        quality_preset,
        acceleration="wan2.2_lightx2v_4step",
        memory_mode="keep_loaded_batch",
    ):
        settings = _quality_settings(quality_preset)
        settings.update(_memory_settings(memory_mode))
        output_path = _run_bernini(
            task_type="t2v",
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed,
            output_prefix="rh_bernini_t2v",
            max_image_size=max(int(width), int(height), 512),
            fps=fps,
            guidance_mode="vae_txt_vit_wapg",
            omega_txt=4.0,
            omega_img=1.0,
            omega_vid=1.0,
            omega_tgt=0.5,
            omega_scale=1.0,
            acceleration=acceleration,
            **settings,
        )
        return (_video_from_path(output_path),)


class RHBerniniFullVideoToVideo(_RHBerniniCompactBase):
    DESCRIPTION = "Full Bernini video-to-video editing using Bernini-Diffusers only."
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video": ("VIDEO", {"tooltip": "Source video from a ComfyUI video output."}),
            "prompt": _prompt_widget("Restyle the source video while preserving motion."),
            "negative_prompt": _negative_prompt_widget(),
        }
        required.update(_video_widgets(default_frames=9))
        required.update(_video_size_widgets())
        required.update(_common_widgets(default_steps=4, default_seed=4567))
        return {"required": required}

    def generate(
        self,
        video,
        prompt,
        negative_prompt,
        num_frames,
        fps,
        width,
        height,
        steps,
        seed,
        quality_preset,
        acceleration="wan2.2_lightx2v_4step",
        memory_mode="keep_loaded_batch",
    ):
        settings = _quality_settings(quality_preset)
        settings.update(_memory_settings(memory_mode))
        source_video_path = _video_to_path(video, "rh_bernini_v2v_source")
        if not source_video_path:
            raise ValueError("video is required")
        output_path = _run_bernini(
            task_type="v2v",
            prompt=prompt,
            source_video_path=source_video_path,
            negative_prompt=negative_prompt,
            system_prompt="You are a helpful assistant specialized in video editing.",
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed,
            output_prefix="rh_bernini_v2v",
            max_image_size=512,
            fps=fps,
            guidance_mode="vae_txt_vit_wapg",
            omega_txt=4.0,
            omega_img=1.25,
            omega_vid=1.25,
            omega_tgt=0.5,
            omega_scale=0.75,
            acceleration=acceleration,
            **settings,
        )
        return (_video_from_path(output_path),)


class RHBerniniFullReferenceToVideo(_RHBerniniCompactBase):
    DESCRIPTION = "Full Bernini reference-image-to-video using Bernini-Diffusers only."
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "reference_image": ("IMAGE", {"tooltip": "One or more reference images. Batched IMAGE inputs are passed as multiple references."}),
            "prompt": _prompt_widget("Animate the reference image into a short smooth video."),
            "negative_prompt": _negative_prompt_widget(),
        }
        required.update(_video_widgets(default_frames=9))
        required.update(_size_widgets())
        required.update(_common_widgets(default_steps=4, default_seed=5678))
        return {"required": required}

    def generate(
        self,
        reference_image,
        prompt,
        negative_prompt,
        num_frames,
        fps,
        width,
        height,
        steps,
        seed,
        quality_preset,
        acceleration="wan2.2_lightx2v_4step",
        memory_mode="keep_loaded_batch",
    ):
        settings = _quality_settings(quality_preset)
        settings.update(_memory_settings(memory_mode))
        reference_paths = _image_batch_to_paths(reference_image, "rh_bernini_r2v_reference")
        if not reference_paths:
            raise ValueError("reference_image is required")
        output_path = _run_bernini(
            task_type="r2v",
            prompt=prompt,
            reference_image_paths=reference_paths,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed,
            output_prefix="rh_bernini_r2v",
            max_image_size=max(int(width), int(height), 512),
            fps=fps,
            guidance_mode="vae_txt_vit_wapg",
            omega_txt=4.0,
            omega_img=4.5,
            omega_vid=1.25,
            omega_tgt=1.5,
            omega_scale=0.8,
            reference_image_chunk_size=1,
            acceleration=acceleration,
            **settings,
        )
        return (_video_from_path(output_path),)


class RHBerniniFullReferenceVideoToVideo(_RHBerniniCompactBase):
    DESCRIPTION = "Full Bernini reference-guided video editing using Bernini-Diffusers only."
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video": ("VIDEO", {"tooltip": "Source video from a ComfyUI video output."}),
            "reference_image": ("IMAGE", {"tooltip": "One or more reference images. Batched IMAGE inputs are passed as multiple references."}),
            "prompt": _prompt_widget("Restyle the source video using the reference image while preserving motion."),
            "negative_prompt": _negative_prompt_widget(),
        }
        required.update(_video_widgets(default_frames=9))
        required.update(_video_size_widgets())
        required.update(_common_widgets(default_steps=4, default_seed=6789))
        return {"required": required}

    def generate(
        self,
        video,
        reference_image,
        prompt,
        negative_prompt,
        num_frames,
        fps,
        width,
        height,
        steps,
        seed,
        quality_preset,
        acceleration="wan2.2_lightx2v_4step",
        memory_mode="keep_loaded_batch",
    ):
        settings = _quality_settings(quality_preset)
        settings.update(_memory_settings(memory_mode))
        source_video_path = _video_to_path(video, "rh_bernini_rv2v_source")
        reference_paths = _image_batch_to_paths(reference_image, "rh_bernini_rv2v_reference")
        if not source_video_path:
            raise ValueError("video is required")
        if not reference_paths:
            raise ValueError("reference_image is required")
        output_path = _run_bernini(
            task_type="rv2v",
            prompt=prompt,
            source_video_path=source_video_path,
            reference_image_paths=reference_paths,
            negative_prompt=negative_prompt,
            system_prompt="You are a helpful assistant specialized in video editing.",
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed,
            output_prefix="rh_bernini_rv2v",
            max_image_size=512,
            fps=fps,
            guidance_mode="rv2v_wapg",
            omega_txt=4.0,
            omega_img=3.0,
            omega_vid=0.75,
            omega_tgt=1.5,
            omega_scale=0.75,
            reference_image_chunk_size=1,
            acceleration=acceleration,
            **settings,
        )
        return (_video_from_path(output_path),)


NODE_CLASS_MAPPINGS = {
    "RHBerniniFullTextToImage": RHBerniniFullTextToImage,
    "RHBerniniFullImageToImage": RHBerniniFullImageToImage,
    "RHBerniniFullTextToVideo": RHBerniniFullTextToVideo,
    "RHBerniniFullVideoToVideo": RHBerniniFullVideoToVideo,
    "RHBerniniFullReferenceToVideo": RHBerniniFullReferenceToVideo,
    "RHBerniniFullReferenceVideoToVideo": RHBerniniFullReferenceVideoToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RHBerniniFullTextToImage": "Bernini Full Text to Image",
    "RHBerniniFullImageToImage": "Bernini Full Image to Image",
    "RHBerniniFullTextToVideo": "Bernini Full Text to Video",
    "RHBerniniFullVideoToVideo": "Bernini Full Video to Video",
    "RHBerniniFullReferenceToVideo": "Bernini Full Reference to Video",
    "RHBerniniFullReferenceVideoToVideo": "Bernini Full Reference + Video to Video",
}
