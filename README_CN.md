[![RunningHub 中国站](https://img.shields.io/badge/RunningHub-%E4%B8%AD%E5%9B%BD%E7%AB%99%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.cn/?inviteCode=rh-v1367)
[![RunningHub 国际站](https://img.shields.io/badge/RunningHub-%E5%9B%BD%E9%99%85%E7%AB%99%20Online%20Platform-2f80ed?labelColor=333333)](https://www.runninghub.ai/?inviteCode=rh-v1367)

# ComfyUI-RH-Bernini-Full

ComfyUI / RunningHub 的 full Bernini 封装。插件固定使用 `ByteDance/Bernini-Diffusers` full pipeline，默认读取本地 `Bernini-Diffusers-qint8` 模型目录；不会加载 Bernini-R、`high_noise_ckpt` 或 `low_noise_ckpt`。

节点使用 ComfyUI 标准 `IMAGE` / `VIDEO` 输入输出，可以直接连接 `LoadImage`、`LoadVideo`、`SaveImage`、`SaveVideo`。旧的全参数大节点已经移除，内部模型路径、输出前缀等运行细节不再暴露到节点 UI。

## 功能

- 文生图：`RHBerniniFullTextToImage`
- 图生图 / 图像编辑：`RHBerniniFullImageToImage`
- 文生视频：`RHBerniniFullTextToVideo`
- 视频编辑：`RHBerniniFullVideoToVideo`
- 参考图生视频：`RHBerniniFullReferenceToVideo`
- 参考图 + 视频编辑：`RHBerniniFullReferenceVideoToVideo`

## 安装

在 ComfyUI 环境中安装插件依赖。不要用 `requirements.txt` 替换 ComfyUI 容器里的 torch / CUDA 相关包。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RH-RunningHub/ComfyUI-RH-Bernini-Full.git
cd ComfyUI-RH-Bernini-Full
pip install -r requirements.txt
pip install --no-deps git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10
```

## 模型

推荐使用已经量化好的 qint8 模型，放到 ComfyUI 的 diffusers 模型目录：

```bash
cd ComfyUI/models
modelscope download --model Gluttony10/Bernini-Diffusers-qint8 --local_dir diffusers/Bernini-Diffusers-qint8
```

插件默认只查找：

```text
ComfyUI/models/diffusers/Bernini-Diffusers-qint8
```

节点会校验 `config.json` 的 `model_type` 必须为 `bernini`，并拒绝包含 `Bernini-R` / `Bernini_R` 的路径。未量化的 `bytedance-community/Bernini-Diffusers` 仅建议用于本地转换或调试；如果要直接运行，需要确保目录内容与 full Bernini-Diffusers 结构一致，并放到默认目录或建立同名软链。

## 常用参数

- `quality_preset`：只保留 `standard` 和 `quality`。`standard` 对齐默认推荐；`quality` 会增加 planning 开销。
- `acceleration`：默认 `wan2.2_lightx2v_4step`。如果 `ComfyUI/models/loras` 下找不到匹配的 Wan2.2 T2V LightX2V 高/低噪 LoRA，会自动回退为无 LoRA 推理。
- `steps`：官方复现工作流使用原始脚本步数；启用 LightX2V 加速时通常设为 `4`。
- `memory_mode`：默认 `keep_loaded_batch`，连续跑 t2i/i2i/v2v/r2v 时复用同一个 Bernini pipeline 缓存；显存紧张时改成 `low_vram_unload`。
- `width` / `height`：图像和文生视频节点直接控制输出尺寸；视频编辑节点设为 `0` 时跟随源视频。

## 显存与速度

- 默认保留模型缓存，第二次运行同一模型和同一 LoRA 配置时避免重复冷加载。
- `low_vram_unload` 会在每次运行后顺序释放 VAE、MLLM/VIT、T5、diffusion transformer，并清理 CUDA cache。
- 内部启用 VAE tiling / slicing，多参考图会分块送入 Qwen-VL 编码，降低峰值显存。
- 更大的显存主要允许更高分辨率、更多帧、更少卸载；首次加载 qint8 模型本身仍需要较长时间。

## Wan2.2 LightX2V LoRA

节点只会在 `ComfyUI/models/loras` 下查找当前适配的高/低噪 LoRA。其它同名 LightX2V / I2V LoRA 不会自动加载，因为实测不适配 Bernini full pipeline，容易明显降低质量。所有任务统一使用这组 T2V LoRA，以便共享 Bernini pipeline 缓存。

```text
ComfyUI/models/loras/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors
ComfyUI/models/loras/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors
```

## 示例工作流

`examples/` 下提供 ComfyUI API prompt 示例，`class_type` 使用插件注册名，可直接用于 `/prompt` 或测试工具。示例素材和参数对齐 bytedance/Bernini 的 `assets/testcases/*` 与 `scripts/bernini/run_*.sh`。

- `bernini_t2i_test_api.json`：文生图，官方 `t2i/t2i.json`。
- `bernini_i2i_test_api.json`：图像编辑，官方 `i2i/i2i.json`。
- `bernini_t2v_test_api.json`：文生视频，官方 `t2v/t2v.json`。
- `bernini_v2v_test_api.json`：视频编辑，官方 `v2v/v2v_case1.json`。
- `bernini_r2v_test_api.json`：多参考图生视频，官方 `r2v/r2v.json`。
- `bernini_rv2v_test_api.json`：参考图 + 视频编辑，官方 `rv2v/rv2v_case1.json`。

运行带输入媒体的示例前，把仓库里的素材复制到 ComfyUI input 目录：

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

更多细节见 `examples/README_CN.md`。

## 限制

- 只支持 full Bernini-Diffusers，不支持 Bernini-R。
- 这个插件是本地推理封装，不会自动下载大模型或 LoRA。
- 视频任务耗时和显存占用较高；如果只做连通性验证，建议先用文生图节点确认环境和模型可加载。

## 来源/相关链接

- 上游代码：https://github.com/bytedance/Bernini
- ModelScope full 模型：https://modelscope.cn/models/bytedance-community/Bernini-Diffusers
- qint8 模型：https://modelscope.cn/models/Gluttony10/Bernini-Diffusers-qint8
- Hugging Face full 模型：https://huggingface.co/ByteDance/Bernini-Diffusers
- 项目页：https://bernini-ai.github.io/
- 论文：https://arxiv.org/abs/2605.22344
- 代码协议：https://www.apache.org/licenses/LICENSE-2.0
