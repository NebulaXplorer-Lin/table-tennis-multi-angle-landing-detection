# 数据去重清理（2026-10-16）

清理完成：删除 144,788 个重复或可重建文件，实际可用空间增加约 49.58 GiB。保留原视频、标注、固定划分、原位置/事件数据版本、权重与实验图册。

## 当前保留

- `data/pp_train_data`、`data/pp_public_data`、`data/ttnet`：原视频及原始标注。
- `event_annotations/`：人工事件导出原件及来源记录。
- `data/cache/frames/31be21667d34f45b0c7f/`：PP 唯一一套原分辨率 RGB uint8 NPY，共 154 视频、23,690 帧，约 65.6 GiB。
- `data/cache/frames/d3299f59fa021c8c845e/`：OpenTTGames 唯一一套 NPY，共 12 视频、28,883 稀疏帧，约 167.3 GiB。
- `data/datasets/pp_ball_v2` 和 `pp_ball_events_v1` 共用同一份 PP 画面，只有轻量标签/划分描述不同；OpenTTGames 使用另一套源视频，两个帧缓存不是重复数据。
- 模型描述符缓存、历史标注与指标只占少量空间，保留其来源身份；旧 PNG/JPEG 训练画面已清理。

## 已清理

- 归档中的 PP 原帧 PNG、两份旧 Global 缩放 PNG。
- 归档 event25_v1/v2/v3 的 JPEG 训练帧。
- 归档 YOLO 数据集的训练/验证 JPG；标签与历史评估图册保留。
- `data/_archives` 中逐成员验证与现存原数据完全一致的两个 PP 压缩包。
- 上轮合并的重复 `first_build` 目录；其位置/事件/划分/窗口内容与最终版本已核对一致。

保留缓存的全部块文件已通过 SHA256 校验。兼容的历史 ball_v1/ball_v2、event25_v1/v3 入口新增只读 `frame_store.json` 引用，复用当前 NPY；旧帧路径仅保留为来源信息。YOLO 图片目录已清理，再跑旧 YOLO 训练需从原视频另行生成图片；不要把历史 `data.yaml` 当作仍有完整图片的数据集。

原视频共181个，未删除。122项受保护历史权重/运行记录在清理前后均通过校验。80项测试中79项通过；1项专门对照旧缩放PNG的历史测试因源缓存已清理而明确跳过。另检查了数组来源重迁移，确保不会依赖已删除图片。

## 清单与复核

- [清理结果](../out/results/20261016_data_cleanup/result.json)
- [删除文件清单](../out/results/20261016_data_cleanup/delete_manifest.jsonl)
- [保留缓存完整校验](../out/results/20261016_data_cleanup/cache_verification.json)
- [位置与事件合并验证](../out/results/20261016_data_cleanup/merged_data_verification.json)
- [模型读取与历史数值对齐](../out/results/20261016_data_cleanup/model_read_verification.json)

以后默认使用 `data/datasets/*` 和 `FrameStore`，保留这一套分块张量与原视频。需要迁移时，现有帧库可直接作为迁移输入；无需恢复旧散落图片。输入缩放和裁剪可在读取时进行，避免再次同时生成重复图片目录。
