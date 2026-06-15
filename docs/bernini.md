# Bernini — full pipeline

[← Back to main README](../README.md)

The full Bernini pipeline combines an MLLM-based semantic planner (built on
Qwen2.5-VL) with the Wan2.2 DiT renderer. The planner decomposes complex
instructions and plans semantic changes in latent space before diffusion
rendering, which gives stronger instruction following on complex
generation/editing requests.

| Field | Value |
|-------|-------|
| Checkpoint | [`ByteDance/Bernini-Diffusers`](https://huggingface.co/ByteDance/Bernini-Diffusers) |
| Renderer base | Wan2.2-T2V-A14B |
| Planner base | Qwen2.5-VL-7B-Instruct |
| Benchmarks | See [main README](../README.md#-highlights) |

## Download weights

Full Bernini uses the packaged **Bernini-Diffusers** layout, which collects the
Bernini checkpoint, the Qwen2.5-VL planner assets, and the Wan2.2 diffusion
components under one directory:

```text
ByteDance/Bernini-Diffusers/
  bernini/                    # Bernini checkpoint
  mllm/                       # semantic planner (Qwen2.5-VL)
  t5_text_encoder/
  t5_tokenizer/
  vae/
  scheduler/
  transformer_config.json     # diffusion decoder configs (no Wan base
  transformer_2_config.json   # transformer weights needed)
```

Download it from Hugging Face:

```bash
pip install -U "huggingface_hub"
hf download ByteDance/Bernini-Diffusers \
    --local-dir ByteDance/Bernini-Diffusers
```

Pass the directory directly as `--config`. The run scripts default to
`ByteDance/Bernini-Diffusers`; to use another location, set:

```bash
export BERNINI_CONFIG=/path/to/Bernini-Diffusers
```

## Run

> Make sure the environment is set up first — see
> [Installation](../README.md#-installation), which includes the required
> VeOmni install.

For single-GPU image tasks, use `infer_single_gpu.py`; for video tasks, use
`infer_multi_gpu.py` with `torchrun` and `--ulysses` sequence parallelism:

```bash
# Single-GPU image editing
python infer_single_gpu.py --config ByteDance/Bernini-Diffusers \
    --case assets/testcases/i2i/i2i.json --num_frames 1

# Multi-GPU video editing
torchrun --nproc-per-node 8 infer_multi_gpu.py \
    --config ByteDance/Bernini-Diffusers --ulysses 8 \
    --case assets/testcases/v2v/v2v_case1.json
```

Inputs are described by case files under
[`assets/testcases/`](../assets/testcases/); see the
[case-file format](../assets/testcases/README.md).

### Run scripts

[`scripts/bernini/`](../scripts/bernini/) provides one ready-to-run script per
task, each with the recommended sampling hyperparameters
(`--guidance_mode vae_txt_vit_wapg`, `--omega_*`, `--vit_*`, ...) and a default
case file. Default output is 480p / 16 fps, 81 frames for video tasks.

```bash
bash scripts/bernini/run_t2i.sh    # text-to-image
bash scripts/bernini/run_i2i.sh    # image editing
bash scripts/bernini/run_t2v.sh    # text-to-video
bash scripts/bernini/run_v2v.sh    # video editing
bash scripts/bernini/run_rv2v.sh   # reference + video editing
bash scripts/bernini/run_r2v.sh    # reference-to-video
```

Each script reads these environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CASE_PATH` | a bundled example case | case JSON to run |
| `BERNINI_CONFIG` | `ByteDance/Bernini-Diffusers` | model directory |
| `NPROC_PER_NODE` | 8 | number of processes |
| `ULYSSES` | 8 | Ulysses sequence-parallel degree |

Example override:

```bash
CASE_PATH=assets/testcases/v2v/v2v_case2.json \
BERNINI_CONFIG=/path/to/Bernini-Diffusers \
NPROC_PER_NODE=8 ULYSSES=8 \
bash scripts/bernini/run_v2v.sh
```

## Gradio demo

```bash
# 8 GPUs, 8-way Ulysses sequence parallel
torchrun --nproc-per-node 8 gradio_demo.py --ulysses 8 \
    --config ByteDance/Bernini-Diffusers --port 7860 --share

# Or the script launcher (honors BERNINI_CONFIG)
bash scripts/bernini/run_gradio.sh
```

See the [Gradio demo notes](../README.md#gradio-demo) in the main README for
the UI behavior and prompt-enhancer setup.
