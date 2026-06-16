# Bernini 示例工作流

这些文件是 ComfyUI API prompt 格式，可通过 `/prompt` 接口提交，也可作为 RunningHub 测试流的最小样例。示例使用 ComfyUI 标准 `LoadImage` / `LoadVideo` / `SaveImage` / `SaveVideo` 节点连接 Bernini 节点。

## 文件

- `bernini_t2i_test_api.json`：文生图，输出到标准 `SaveImage`。
- `bernini_i2i_test_api.json`：图生图，通过 `LoadImage` 读取 `bernini_i2i_source.png`，输出到标准 `SaveImage`。
- `bernini_t2v_test_api.json`：文生视频，短视频 smoke test，默认 9 帧。
- `bernini_v2v_test_api.json`：视频编辑，通过 `LoadVideo` 读取 `bernini_source_video.mp4`。
- `bernini_r2v_test_api.json`：参考图生视频，通过 `LoadImage` 读取 `bernini_reference_tile.png`。
- `bernini_rv2v_test_api.json`：参考图 + 视频编辑，通过 `LoadVideo` 读取 `bernini_source_video.mp4`，并通过 `LoadImage` 读取 `bernini_reference_tile.png`。

## 注意

运行图生图/参考图示例前，先把仓库里的 PNG 示例资产复制到 ComfyUI input 目录：

```text
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/i2i_source.png input/bernini_i2i_source.png
cp custom_nodes/ComfyUI-RH-Bernini-Full/examples/assets/reference_tile.png input/bernini_reference_tile.png
```

`v2v` 和 `rv2v` 示例还需要把源视频放到 `input/bernini_source_video.mp4`。

示例默认使用 `memory_mode=keep_loaded_batch` 保留模型，并开启 `acceleration=wan2.2_lightx2v_4step` 自动查找 Wan2.2 T2V LightX2V LoRA。所有示例默认使用 `4` diffusion steps；视频类示例仍保持 `9` 帧、`512x512`，用于插件连通性测试。正式视频出片时建议把 `num_frames` 调回 `81`。内存紧张时把 `memory_mode` 改为 `low_vram_unload`。

在 RunningHub 远端测试机上直接调用 `/prompt` 接口时，提交体建议包含：

```json
{
  "prompt": {},
  "extra_data": {},
  "workflow_config": {}
}
```

否则部分 RH 执行器版本会在水印/元数据配置读取阶段把 `workflow_config=None` 当作字典使用并报错。
