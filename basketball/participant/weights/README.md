# participant/weights/

放官方 demo 的球员检测模型 **player.onnx**（由赛事方提供，不随代码仓分发）。

当前版本：**YOLO26n COCO 预训练** 导出（imgsz=960 静态、opset 13、FP32、端到端 NMS-free），
导出命令（容器外，Ultralytics 环境）：
```bash
yolo export model=yolo26n.pt format=onnx imgsz=960 opset=13
```

## ONNX 约定

| 项 | 要求 |
|---|---|
| 文件名 | `player.onnx` |
| 输入 | `(1, 3, H, W)` float32，**RGB**，值域 [0,1]（letterbox + /255 已在 demo 内做） |
| 输入尺寸 | 建议 **960×960** 或 1280×1280（640 下远端球员易漏）；静态尺寸最利于 TRT 固定 shape engine |
| 输出 | 三种布局 demo 首次推理自动识别（可用 `layout=` 显式指定）： ① YOLOv5 风格 `[1, N, 5+nc]`（cx,cy,w,h,obj,cls…，得分=obj*cls）② YOLOv8 风格 `[1, 4+nc, N]`（cx,cy,w,h,cls…）③ **端到端 NMS-free** `[1, K, 6]`（x1,y1,x2,y2,score,class_id，YOLO26/v10 等，图内已做 NMS，K 固定如 300、补齐行 score=0）。当前权重为 ③，solution.py 已显式 `layout="e2e"` |
| 类别 | COCO 预训练 person=0（在 solution.py 里 `CLASSES=[0]`）；自训单类模型 `CLASSES=None` |
| opset | 建议 12–17（TRT 8.6 兼容区间；当前 13） |

导出后建议先本地验证：
```bash
python3 -c "import onnx; m=onnx.load('player.onnx'); onnx.checker.check_model(m); print(m.graph.input[0])"
```
