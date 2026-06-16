[![RunningHub 中国站](https://img.shields.io/badge/RunningHub-%E4%B8%AD%E5%9B%BD%E7%AB%99%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub 国际站](https://img.shields.io/badge/RunningHub-%E5%9B%BD%E9%99%85%E7%AB%99%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.ai/?inviteCode=rh-v1367)

# ComfyUI-RH-Bernini-Full

这是 full Bernini 的 ComfyUI/RunningHub 封装。当前插件固定加载 `ByteDance/Bernini-Diffusers` / ModelScope `bytedance-community/Bernini-Diffusers` 对应的 full pipeline，不加载 Bernini-R、`high_noise_ckpt` 或 `low_noise_ckpt`。可选的 Wan2.2 LightX2V LoRA 只作为 Bernini diffusion transformer 的 4-step 加速适配器使用。

## 当前节点

- `RHBerniniFullTextToImage`：文生图。
- `RHBerniniFullImageToImage`：图生图/图像编辑。
- `RHBerniniFullTextToVideo`：文生视频。
- `RHBerniniFullVideoToVideo`：视频编辑。
- `RHBerniniFullReferenceToVideo`：参考图生视频。
- `RHBerniniFullReferenceVideoToVideo`：参考图 + 视频编辑。

旧的全参数大节点已经移除。当前任务节点使用 ComfyUI 标准 `IMAGE` / `VIDEO` socket 作为媒体输入和输出，不再暴露文件路径输入或 `output_prefix`；需要加载或保存时，直接连接 `LoadImage`、`LoadVideo`、`SaveImage`、`SaveVideo`。

## 内存控制

- `sequential_unload`：阶段执行后顺序卸载 VAE、MLLM/VIT、T5 和 diffusion transformer，并清理 CUDA cache。
- `vae_tiling` / `vae_slicing`：启用 diffusers VAE 空间/批次分块能力。
- `reference_image_chunk_size`：多参考图按块送入 Qwen-VL 编码，降低峰值显存。
- `keep_model_loaded`：控制运行后是否保留 pipeline 缓存。
- `memory_mode`：新节点中的预设开关；默认 `keep_loaded_batch`，适合连续批量任务；低内存机器可手动切到 `low_vram_unload`，每次运行后释放缓存。
- `acceleration`：默认 `wan2.2_lightx2v_4step`。节点会从 `ComfyUI/models/loras` 自动寻找 Wan2.2 T2V LightX2V 高/低噪 LoRA；找不到时回退为不加载 LoRA。

## Wan2.2 LightX2V 加速 LoRA

支持的文件名优先级：

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

所有任务类型统一使用 T2V LoRA，这样连续执行 T2I/I2I/V2V 时可以共用同一个 Bernini pipeline 缓存。加速 LoRA 主要减少正式采样步数，建议把 `steps` 设为 `4` 左右；它不会缩短首次加载 Bernini qint8 模型的时间。

## 模型路径

默认模型目录为：

```text
ComfyUI/models/diffusers/Bernini-Diffusers-qint8
```

未量化模型可临时放在：

```text
ComfyUI/models/diffusers/Bernini-Diffusers
```

节点会检查 `config.json` 的 `model_type` 必须是 `bernini`，并拒绝包含 `Bernini-R` 的路径。

## 示例工作流

`examples/` 下提供 ComfyUI API prompt 示例。发布、本地测试和 RH 验收均使用 `_api.json` 文件。

- `bernini_t2i_test_api.json`
- `bernini_i2i_test_api.json`
- `bernini_t2v_test_api.json`
- `bernini_v2v_test_api.json`
- `bernini_r2v_test_api.json`
- `bernini_rv2v_test_api.json`

图像/视频示例均使用 ComfyUI 标准加载节点。运行前把示例 PNG 和源视频放到 ComfyUI input 目录：

```text
ComfyUI/input/bernini_i2i_source.png
ComfyUI/input/bernini_reference_tile.png
ComfyUI/input/bernini_source_video.mp4
```

更多说明见 `examples/README_CN.md`。

## 来源/相关链接

- 上游代码：https://github.com/bytedance/Bernini
- ModelScope 模型：https://modelscope.cn/models/bytedance-community/Bernini-Diffusers
- Hugging Face 模型：https://huggingface.co/ByteDance/Bernini-Diffusers
- 项目页：https://bernini-ai.github.io/
- 论文：https://arxiv.org/abs/2605.22344
- 代码协议：https://www.apache.org/licenses/LICENSE-2.0
