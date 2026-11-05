# PP＋OpenTTGames定位器对照（2026-11-03）

本轮由用户明确要求在球定位训练中加入OpenTTGames，优先于旧SOP仅用PP的默认数据范围。两组均使用新混合数据、有效batch16、seed42；与历史PP单来源、有效batch8的成绩分开比较。

| 组别 | Global | Local |
|---|---|---|
| A | 原fixed3 Residual FPN | 原fixed3 Residual FPN |
| B | 全分辨率并行多尺度＋SE | 原fixed3 Residual FPN，使用B的Global预测训练 |

数据版本为`data/datasets/pp_opentt_ball_v1`。保留PP原114/40清单与标签；OpenTT原生120fps，game_1–4训练、game_5验证、test_1–7锁定后测试。原视频包PTS索引逐项检查单调性、数量和120fps间隔，每视频另验证9个实际解码时间戳。全部监督中心的三帧邻域按真实帧号读取，标注空隙不压缩成邻帧。

训练每轮PP 15,544个有效中心＋等量OpenTT，两个来源各50%；OpenTT共38,568个有效中心跨轮轮换。每个有效batch包含两来源各8个样本。采样与增强由seed、epoch和样本索引决定，独立于模型初始化及micro batch。每轮保存采样数量和清单哈希。

训练仍为AdamW lr=3e-4、wd=1e-4、恒定学习率、AMP、最多20轮。Local前10轮GT抖动、后10轮本组Global预测ROI；PP验证选择阈值与checkpoint，game_5单列，无球和未知标签分别处理。测试成绩不触发重训或调参。

新Global通道16/32/64/96，空间比例1/1、1/2、1/4、1/8；保留共享逐帧编码和四尺度时序融合，再做两轮多尺度交换和SE。最终输出仍为32通道半分辨率空间特征、单通道半分辨率热图和128维描述符。Global/Local拼接后的事件输入为`[B,13,256]`。

## 入口

```bash
# 首次构建新数据；发布后的数据目录拒绝覆盖。
.venv/bin/python -m ttnet.data_tools.prepare_mixed_ball

# 构建共享resize输入，并检查跨来源近似重复；有候选时先核实。
.venv/bin/python -m ttnet.data_tools.audit_mixed_ball \
  --dataset data/datasets/pp_opentt_ball_v1 \
  --output out/results/20261103_mixed_ball_source_audit

# 真实取数、反向和优化器更新校准；需要GPU可见的运行环境。
.venv/bin/python -m ttnet.training.calibrate_mixed \
  --output out/results/20261103_mixed_ball_calibration

# 两条Global→Local流水线；是否并行及micro batch来自实测。
.venv/bin/python -m ttnet.training.mixed_experiment \
  --task out/experiments/<新时间戳>_mixed_ball_comparison \
  --calibration out/results/20261103_mixed_ball_calibration/calibration.json \
  --audit out/results/20261103_mixed_ball_source_audit/audit.json
```

新任务先做小样本过拟合检查。两训练进程的保守显存预算小于总显存80%，并行固定步数总完工时间至少缩短5%才采用并发。Global缓存和GT裁剪缓存共享，预测裁剪缓存按Global SHA隔离；源视频不展开第二份原图缓存。正式推理测速在训练完成后独占GPU。

训练任务的`task.json`记录当前阶段与四个运行目录；每次命令的日志与退出状态位于任务`logs/`。恢复同一任务时使用同一`--task`，已成功的阶段跳过，未完成阶段沿用原配置恢复。最终目录含PP独立/关联、OpenTT原生验证/测试的分开评估、错误叠图、ONNX对齐及真实PP/event25特征梯度检查。全部证据齐备后才标记completed。

## 验证与解释

```bash
.venv/bin/python -m unittest discover -s ttnet/src/tests -q
```

近似重复筛查记录每五帧PP图像与全部OpenTT有效球中心的pHash/像素相似候选，原视频哈希另作精确去重。这不是对任意裁剪、覆盖字幕或变速版本零泄漏的数学证明；报告保留检查范围。测试图像只用于排查训练内容重叠，模型锁定前不计算测试精度。

特征尺寸一致只保证接口兼容。新特征缓存必须按新权重重建，后续事件头需要重新训练。B同时改变高分辨率分支和SE，本轮不分别归因。

## 本次运行

任务状态：[task.json](../out/experiments/20261103-140723_mixed_ball_comparison/task.json)；持久化启动与阶段日志：[driver.log](../out/launches/20261103-140723_mixed_ball_comparison/driver.log)。20轮完整中心帧清单保存在任务的`shared_sampling/`，四个训练阶段共同使用并在最终汇总时核对逐轮哈希。

最终模型实现复测的100次优化器更新：串行合计32.16秒，并发31.56秒，收益约1.9%，未达到约定5%，因此采用串行。固定A micro=8、B micro=4，两者effective batch=16。初次三个batch组合的原始记录保留，正式依据为[最终复测](../out/results/20261103_mixed_ball_calibration_final/calibration.json)。

[跨来源筛查](../out/results/20261103_mixed_ball_source_audit/audit.json)覆盖4805张PP抽样帧与52944个OpenTT有效中心，阈值内候选为0。预训练检查中的新Global ONNX在6个输入案例（含动态batch和实际PP帧）上通过2e-4容差，热图/特征最大误差约2.0e-6/9.5e-7；训练后的最终权重仍会重新导出与检查。

当前状态以任务和各阶段JSON为准；训练及完整验证全部结束前，不将预检或中间epoch指标称为最终结果。

四个阶段均已完成20轮，训练期间PP选出的Global/Local最佳F1分别为A的0.87575（50px）/0.87378（10px），B的0.90985（50px）/0.90456（10px）。这不是尚未完成的原生全流式对照报告。

最终导出首次在A Global描述符对齐处停止。训练后GroupNorm在ONNX大空间归约上的数值误差暴露于常量输入；仅在导出图使用逐维归约和同一2×2分箱的FP32均值，训练实现、权重和运行时GroupNorm不变。原`atol=rtol=2e-4`混合容差未放宽，四个最终模型各8例（真实PP、零输入、动态batch、中心帧有效、全缺帧）通过。原始失败、修复检查和恢复命令保存在任务的`recoveries/`及`out/results/20261103_mixed_ball_export_recovery/`。

最终A/B权重的真实PP与event25窗口均已通过`[1,13,256]`前向和事件头反传检查，Global与Local梯度均有限且非零；此检查不更新保存的定位器参数，也不构成正式事件头训练。
