# B＋旧MLP配对对照

冻结旧Global和新B Global，成对重新训练相同TableHead。固定seed42、相同初始权重与逐轮样本顺序，训练集全部有效框、无增强、batch128、AdamW 3e-4、L1+GIoU，最多60轮/patience10，按v1验证帧均IoU选best。

参考分支复用旧Global空间缓存并按v1重建targets；B重新提取4×4空间缓存。不改变已有CNN结果，不启动事件训练。B＋MLP与旧B CNN的训练配方不同，不直接宣称纯头结构因果。

实时状态以新任务task.json为准。
