# 组合定位错误检查

[打开错误图册和慢放视频](errors.html)

绿色 = GT；红色 = 最终 Pred；蓝色 = Local ROI；视频额外显示黄色 Global。所有帧均来自已保存原图，使用同一次 best_val 的预测和阈值，没有重新推理。

| 分组 | 错误帧数 |
|---|---:|
| GT 在 Local 裁剪区域外 | 636 |
| GT 在 ROI 内，Local 定位偏差 | 88 |
| GT 在 ROI 内，Local 无有效结果，回退 Global 仍有偏差 | 43 |
| Global 未通过阈值，无 Local 裁剪区域 | 42 |

图册覆盖全部评分错误；视频是各分组的一个示例，保留中心前后约 0.75 秒的真实帧，约 1/3 速度播放。

- GT 在 Local 裁剪区域外：[02_001，帧 8 图片](../../visualizations/best_val/02_001/000008_location_error.jpg) · [慢放视频](../../visualizations/error_clips/gt_outside_roi_02_001_000008.mp4)
- GT 在 ROI 内，Local 定位偏差：[02_001，帧 22 图片](../../visualizations/best_val/02_001/000022_location_error.jpg) · [慢放视频](../../visualizations/error_clips/local_error_02_001_000022.mp4)
- GT 在 ROI 内，Local 无有效结果，回退 Global 仍有偏差：[02_001，帧 46 图片](../../visualizations/best_val/02_001/000046_location_error.jpg) · [慢放视频](../../visualizations/error_clips/global_fallback_error_02_001_000046.mp4)
- Global 未通过阈值，无 Local 裁剪区域：[03_002，帧 8 图片](../../visualizations/best_val/03_002/000008_miss.jpg) · [慢放视频](../../visualizations/error_clips/no_global_roi_03_002_000008.mp4)
