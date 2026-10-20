# 四组冻结定位器事件训练：seed42

状态：GPU 后台队列运行中，四组真实样本 smoke（含 micro-batch8）已通过。实时状态以 [status.json](/home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42/status.json) 为准。

用户已确认“触台已标全，主要漏了挥拍／落地等负样本标签”。本轮未标出的 Bounce 作为负类，ignore 与空导出02_004保持 unknown，Net 屏蔽。原粗标注版本不改写；训练使用固定的 [标注快照](annotation_snapshot.json)。

| 集合 | 视频 | Bounce | 有球位置 GT 参考 | 仅时间评分 |
|---|---:|---:|---:|---:|
| train | 114 | 613 | 600 | 13 |
| val | 40 | 203 | 189 | 14 |

视角为图像复核后的 front/diagonal/side 粗分类，训练81/18/15视频，验证8/14/18视频；不声称精确标定角度。落点主指标使用事件帧原球心 GT 参考，缺坐标的事件仍参与连续触台时间评分，但不计入空间FN。

四组冻结现有三帧定位器，13帧事件窗口；AdamW3e-4、有效batch8、最多30轮、patience5，seed42。为提速，定位特征批量8、事件训练micro-batch8、验证batch128。A/C共用缓存，B/D共用缓存；B/D的缺帧历史从原视频流式解码，不生成第二套原帧缓存。

RTX5070Ti小样本定位提取吞吐约79.7→133.8 FPS；事件头同一有效batch8前后向约12.21→1.55ms。这是组件测速，完整训练另含读数据、验证和保存时间。AMP批处理最大描述符差异约1.03e-4，测得128帧中超过1px的几何变化为0。batch16/32/64均未比8更快。

队列顺序：四组smoke → A/C训练 → B/D训练 → 四组PP验证 → A0验证 → seed42对照报告。后续A与最佳新方案seed43/44，以及独立test/public泛化，待本轮结果后继续。
