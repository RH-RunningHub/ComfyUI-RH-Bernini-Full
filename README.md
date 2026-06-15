[![RunningHub China](https://img.shields.io/badge/RunningHub-China%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.ai/?inviteCode=rh-v1367)

# ComfyUI-RH-Bernini-Full

ComfyUI/RunningHub wrapper for the full Bernini pipeline. This plugin always loads the full `ByteDance/Bernini-Diffusers` pipeline and intentionally does not load Bernini-R, `high_noise_ckpt`, or `low_noise_ckpt`. Optional Wan2.2 LightX2V LoRAs are used only as 4-step acceleration adapters for the Bernini diffusion transformers.

## Node

- `RHBerniniFullTextToImage`: text to image.
- `RHBerniniFullImageToImage`: image editing.
- `RHBerniniFullTextToVideo`: text to video.
- `RHBerniniFullVideoToVideo`: video editing.
- `RHBerniniFullReferenceToVideo`: reference image to video.
- `RHBerniniFullReferenceVideoToVideo`: reference image + video editing.

The old all-parameter node has been removed. The task-specific nodes use standard ComfyUI `IMAGE` and `VIDEO` sockets for media input and output. They do not expose filename/path inputs or `output_prefix`; connect them to `LoadImage`, `LoadVideo`, `SaveImage`, or `SaveVideo` as needed.

## Memory Controls

- `sequential_unload`: offloads VAE, MLLM/VIT, T5, and diffusion transformer stages after each phase.
- `vae_tiling` / `vae_slicing`: enables diffusers VAE spatial/batch chunking when available.
- `reference_image_chunk_size`: chunks Qwen-VL reference image encoding.
- `keep_model_loaded`: keeps or releases the cached pipeline after each run.
- `memory_mode`: compact-node preset; defaults to `keep_loaded_batch` for faster consecutive jobs, while `low_vram_unload` releases cached state after each run for low-memory machines.
- `acceleration`: defaults to `wan2.2_lightx2v_4step`. The nodes auto-detect the Wan2.2 T2V LightX2V high/low-noise LoRA pair under `ComfyUI/models/loras`; when it is absent, inference runs without LoRA.

## Wan2.2 LightX2V LoRA Acceleration

Supported filename priority:

```text
ComfyUI/models/loras/wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors
ComfyUI/models/loras/wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors
ComfyUI/models/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors
ComfyUI/models/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors
ComfyUI/models/loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors
ComfyUI/models/loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors
ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
```

All task types use the same T2V LoRA pair so consecutive T2I/I2I/V2V jobs can share the cached Bernini pipeline. The LoRA mainly speeds production sampling by enabling around `4` diffusion steps; it does not reduce the first-time Bernini qint8 model load.

## Model Layout

Default model directory:

```text
ComfyUI/models/diffusers/Bernini-Diffusers-qint8
```

Temporary unquantized model directory:

```text
ComfyUI/models/diffusers/Bernini-Diffusers
```

The node validates that `config.json` has `model_type: "bernini"` and rejects paths containing `Bernini-R`.

## Example Workflows

ComfyUI API prompt examples are available in `examples/`:

- `bernini_t2i_test.json`
- `bernini_i2i_test.json`
- `bernini_t2v_test.json`
- `bernini_v2v_test.json`
- `bernini_r2v_test.json`
- `bernini_rv2v_test.json`

The image/video examples use standard ComfyUI loaders. Place the example PNGs and your source video in the ComfyUI input directory before running:

```text
ComfyUI/input/bernini_i2i_source.png
ComfyUI/input/bernini_reference_tile.png
ComfyUI/input/bernini_source_video.mp4
```

See `examples/README_CN.md` for details.

## Links

- Upstream code: https://github.com/bytedance/Bernini
- ModelScope model: https://modelscope.cn/models/bytedance-community/Bernini-Diffusers
- Hugging Face model: https://huggingface.co/ByteDance/Bernini-Diffusers
- Project page: https://bernini-ai.github.io/
- Paper: https://arxiv.org/abs/2605.22344
- License: https://www.apache.org/licenses/LICENSE-2.0
