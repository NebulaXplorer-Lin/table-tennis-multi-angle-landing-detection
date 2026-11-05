# 球桌两路线训练

2026-11-04用户修订：B共享分支继续，不再使用YOLO。排队YOLO任务已取消；独立分支替换为原生单帧多尺度特征编码器＋轻量CNN，从头训练。先完成现有B，再自动开始independent_cnn，不启动事件头训练。原YOLO配置作为retired_independent_yolo11s.json保留历史。

实时状态以任务JSON与各路线run.json为准；本文件及run.json为修订快照。
