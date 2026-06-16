[![RunningHub China](https://img.shields.io/badge/RunningHub-China%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub International](https://img.shields.io/badge/RunningHub-International%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.ai/?inviteCode=rh-v1367)

# ComfyUI-RH-Bernini-Full

ComfyUI / RunningHub wrapper for the full Bernini pipeline. The plugin uses `ByteDance/Bernini-Diffusers` only and loads the local `Bernini-Diffusers-qint8` model directory by default. It does not load Bernini-R, `high_noise_ckpt`, or `low_noise_ckpt`.

The nodes use standard ComfyUI `IMAGE` and `VIDEO` sockets, so they can be connected directly to `LoadImage`, `LoadVideo`, `SaveImage`, and `SaveVideo`. The old all-parameter node has been removed; internal model paths and output prefixes are no longer exposed as node widgets.

## Features

- Text to image: `RHBerniniFullTextToImage`
- Image editing: `RHBerniniFullImageToImage`
- Text to video: `RHBerniniFullTextToVideo`
- Video editing: `RHBerniniFullVideoToVideo`
- Reference image to video: `RHBerniniFullReferenceToVideo`
- Reference image + video editing: `RHBerniniFullReferenceVideoToVideo`

## Installation

Install inside your ComfyUI Python environment. Do not use this plugin's dependencies to replace the torch / CUDA packages already provided by your ComfyUI runtime.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RH-RunningHub/ComfyUI-RH-Bernini-Full.git
cd ComfyUI-RH-Bernini-Full
pip install -r requirements.txt
pip install --no-deps git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10
```

## Model

The recommended runtime model is the pre-quantized qint8 package:

```bash
cd ComfyUI/models
modelscope download --model Gluttony10/Bernini-Diffusers-qint8 --local_dir diffusers/Bernini-Diffusers-qint8
```

The plugin looks for this directory by default:

```text
ComfyUI/models/diffusers/Bernini-Diffusers-qint8
```

The node validates that `config.json` has `model_type: "bernini"` and rejects paths containing `Bernini-R` or `Bernini_R`. The unquantized `bytedance-community/Bernini-Diffusers` package is useful for local conversion or debugging; to run it directly, keep the full Bernini-Diffusers structure and place it at the default path or create a same-name symlink.

## Common Parameters

- `quality_preset`: only `standard` and `quality` are available. `standard` is the recommended default; `quality` spends more planning time.
- `acceleration`: defaults to `wan2.2_lightx2v_4step`. If no matching Wan2.2 T2V LightX2V high/low-noise LoRA pair is found under `ComfyUI/models/loras`, the node falls back to inference without LoRA.
- `steps`: official reproduction workflows use the original script step counts; with LightX2V acceleration enabled, `4` steps is the expected fast setting.
- `memory_mode`: defaults to `keep_loaded_batch`, allowing consecutive t2i/i2i/v2v/r2v jobs to reuse the same cached Bernini pipeline. Use `low_vram_unload` on memory-constrained machines.
- `width` / `height`: image and text-to-video nodes control output size directly; video editing nodes follow the source video when both are set to `0`.

## Memory And Speed

- The model cache is kept by default, so repeated runs with the same model and LoRA configuration avoid another cold load.
- `low_vram_unload` releases VAE, MLLM/VIT, T5, and diffusion transformer stages in order after each run, then clears CUDA cache.
- VAE tiling / slicing is enabled internally. Multi-reference images are encoded in chunks to reduce peak VRAM.
- More VRAM mainly allows larger resolution, more frames, and less unloading. It does not remove the first qint8 model load cost.

## Wan2.2 LightX2V LoRA

The node only searches the currently adapted high/low-noise LoRA pair under `ComfyUI/models/loras`. Other similarly named LightX2V / I2V LoRAs are not auto-loaded because they are not a good fit for the full Bernini pipeline and can noticeably reduce quality. All task types use this T2V LoRA pair so they can share the cached Bernini pipeline.

```text
ComfyUI/models/loras/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors
ComfyUI/models/loras/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors
```

## Example Workflows

ComfyUI API prompt examples are available in `examples/`. Their `class_type` values use the plugin registration names and can be submitted to `/prompt` or test runners. Assets and parameters follow bytedance/Bernini `assets/testcases/*` and `scripts/bernini/run_*.sh`.

- `bernini_t2i_test_api.json`: text to image, upstream `t2i/t2i.json`.
- `bernini_i2i_test_api.json`: image editing, upstream `i2i/i2i.json`.
- `bernini_t2v_test_api.json`: text to video, upstream `t2v/t2v.json`.
- `bernini_v2v_test_api.json`: video editing, upstream `v2v/v2v_case1.json`.
- `bernini_r2v_test_api.json`: multi-reference image to video, upstream `r2v/r2v.json`.
- `bernini_rv2v_test_api.json`: reference image + video editing, upstream `rv2v/rv2v_case1.json`.

Before running media-input examples, copy the bundled assets to your ComfyUI input directory:

```bash
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_i2i_source.png input/bernini_i2i_source.png
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_v2v_source_case1.mp4 input/bernini_v2v_source_case1.mp4
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_rv2v_source_case1.mp4 input/bernini_rv2v_source_case1.mp4
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_rv2v_ref_case1.jpg input/bernini_rv2v_ref_case1.jpg
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_r2v_source_img0.png input/bernini_r2v_source_img0.png
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_r2v_source_img1.png input/bernini_r2v_source_img1.png
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_r2v_source_img2.png input/bernini_r2v_source_img2.png
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_r2v_source_img3.png input/bernini_r2v_source_img3.png
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/bernini_r2v_source_img4.png input/bernini_r2v_source_img4.png
```

See `examples/README_CN.md` for more details.

## Limitations

- Only full Bernini-Diffusers is supported; Bernini-R is intentionally unsupported.
- The plugin performs local inference and does not download large models or LoRAs automatically.
- Video jobs are slow and VRAM-heavy. For environment checks, start with the text-to-image node.

## Links

- Upstream code: https://github.com/bytedance/Bernini
- ModelScope full model: https://modelscope.cn/models/bytedance-community/Bernini-Diffusers
- qint8 model: https://modelscope.cn/models/Gluttony10/Bernini-Diffusers-qint8
- Hugging Face full model: https://huggingface.co/ByteDance/Bernini-Diffusers
- Project page: https://bernini-ai.github.io/
- Paper: https://arxiv.org/abs/2605.22344
- License: https://www.apache.org/licenses/LICENSE-2.0
