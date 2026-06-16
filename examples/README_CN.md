# Bernini 示例工作流

这些文件是 ComfyUI API prompt 格式，可通过 `/prompt` 接口提交，也可作为 RunningHub 测试流的样例。示例素材和 prompt 来自 bytedance/Bernini 的 `assets/testcases/`，生成参数对齐 `scripts/bernini/run_*.sh`。示例使用 ComfyUI 标准 `LoadImage` / `LoadVideo` / `SaveImage` / `SaveVideo` 节点连接 Bernini 节点。

## 文件

- `bernini_t2i_test_api.json`：文生图，对齐官方 `t2i/t2i.json` 和 `run_t2i.sh`。
- `bernini_i2i_test_api.json`：图生图，通过 `LoadImage` 读取 `bernini_i2i_source.png`，对齐官方 `i2i/i2i.json` 和 `run_i2i.sh`。
- `bernini_t2v_test_api.json`：文生视频，对齐官方 `t2v/t2v.json` 和 `run_t2v.sh`。
- `bernini_v2v_test_api.json`：视频编辑，通过 `LoadVideo` 读取 `bernini_v2v_source_case1.mp4`，对齐官方 `v2v/v2v_case1.json` 和 `run_v2v.sh`。
- `bernini_r2v_test_api.json`：多参考图生视频，通过 5 个 `LoadImage` + `ImageBatch` 读取官方 `r2v/source_img*.png`。
- `bernini_rv2v_test_api.json`：参考图 + 视频编辑，通过 `LoadVideo` 读取 `bernini_rv2v_source_case1.mp4`，并通过 `LoadImage` 读取 `bernini_rv2v_ref_case1.jpg`，对齐官方 `rv2v/rv2v_case1.json` 和 `run_rv2v.sh`。

## 注意

运行图生图、视频编辑和参考图示例前，先把仓库里的官方示例资产复制到 ComfyUI input 目录：

```text
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

参数摘要：

- 通用：`seed=42`、`fps=16`、`quality_preset=standard`、`acceleration=none`、`memory_mode=keep_loaded_batch`。
- `t2i`：`512x512`、`steps=50`。
- `i2i`：`512x512`、`steps=40`。
- `t2v`：`848x480`、`81` 帧、`steps=50`。
- `v2v`：宽高为 `0`，跟随源视频、`81` 帧、`steps=40`。
- `r2v`：`832x480`、`81` 帧、`steps=40`。
- `rv2v`：宽高为 `0`，跟随源视频、`81` 帧、`steps=40`。

这些示例是按官方 Bernini 样例复现参数写的，不是低步数连通性测试。内存紧张时把 `memory_mode` 改为 `low_vram_unload`。

在 RunningHub 远端测试机上直接调用 `/prompt` 接口时，提交体建议包含：

```json
{
  "prompt": {},
  "extra_data": {},
  "workflow_config": {}
}
```

否则部分 RH 执行器版本会在水印/元数据配置读取阶段把 `workflow_config=None` 当作字典使用并报错。
