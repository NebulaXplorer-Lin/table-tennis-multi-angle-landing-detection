# ball.onnx 算子层清单

由 inspect_ball_onnx.py 生成。序号为 ONNX 图中 0 基节点索引；shape 推断时固定 batch=1。

只列 Conv、MatMul、Resize；完整连接与其他算子见 ball_onnx_graph.json。MAC 不包括激活、偏置、插值及逐元素运算。

| 节点 | 名称 | 类型 | 权重形状 | 输出形状 | MAC |
| --- | --- | --- | --- | --- | ---: |
| 0 | `/model/conv1/Conv` | Conv | `[[64, 9, 3, 3], [64]]` | `[[1, 64, 288, 512]]` | 764,411,904 |
| 2 | `/model/conv2/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 288, 512]]` | 5,435,817,984 |
| 4 | `/model/layer1/layer1.0/conv1/Conv` | Conv | `[[32, 64, 1, 1], [32]]` | `[[1, 32, 288, 512]]` | 301,989,888 |
| 6 | `/model/layer1/layer1.0/conv2/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 288, 512]]` | 1,358,954,496 |
| 8 | `/model/layer1/layer1.0/conv3/Conv` | Conv | `[[128, 32, 1, 1], [128]]` | `[[1, 128, 288, 512]]` | 603,979,776 |
| 9 | `/model/layer1/layer1.0/downsample/downsample.0/Conv` | Conv | `[[128, 64, 1, 1], [128]]` | `[[1, 128, 288, 512]]` | 1,207,959,552 |
| 12 | `/model/transition1.0/transition1.0.0/Conv` | Conv | `[[16, 128, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 2,717,908,992 |
| 14 | `/model/transition1.1/transition1.1.0/transition1.1.0.0/Conv` | Conv | `[[32, 128, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 1,358,954,496 |
| 16 | `/model/stage2/stage2.0/branches.0/branches.0.0/conv1/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 18 | `/model/stage2/stage2.0/branches.0/branches.0.0/conv2/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 32 | `/model/stage2/stage2.0/branches.0/branches.0.0/se/fc/fc.0/MatMul` | MatMul | `[[16, 1]]` | `[[1, 1]]` | 16 |
| 34 | `/model/stage2/stage2.0/branches.0/branches.0.0/se/fc/fc.2/MatMul` | MatMul | `[[1, 16]]` | `[[1, 16]]` | 16 |
| 47 | `/model/stage2/stage2.0/branches.0/branches.0.1/conv1/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 49 | `/model/stage2/stage2.0/branches.0/branches.0.1/conv2/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 63 | `/model/stage2/stage2.0/branches.0/branches.0.1/se/fc/fc.0/MatMul` | MatMul | `[[16, 1]]` | `[[1, 1]]` | 16 |
| 65 | `/model/stage2/stage2.0/branches.0/branches.0.1/se/fc/fc.2/MatMul` | MatMul | `[[1, 16]]` | `[[1, 16]]` | 16 |
| 78 | `/model/stage2/stage2.0/branches.1/branches.1.0/conv1/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 80 | `/model/stage2/stage2.0/branches.1/branches.1.0/conv2/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 94 | `/model/stage2/stage2.0/branches.1/branches.1.0/se/fc/fc.0/MatMul` | MatMul | `[[32, 2]]` | `[[1, 2]]` | 64 |
| 96 | `/model/stage2/stage2.0/branches.1/branches.1.0/se/fc/fc.2/MatMul` | MatMul | `[[2, 32]]` | `[[1, 32]]` | 64 |
| 109 | `/model/stage2/stage2.0/branches.1/branches.1.1/conv1/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 111 | `/model/stage2/stage2.0/branches.1/branches.1.1/conv2/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 125 | `/model/stage2/stage2.0/branches.1/branches.1.1/se/fc/fc.0/MatMul` | MatMul | `[[32, 2]]` | `[[1, 2]]` | 64 |
| 127 | `/model/stage2/stage2.0/branches.1/branches.1.1/se/fc/fc.2/MatMul` | MatMul | `[[2, 32]]` | `[[1, 32]]` | 64 |
| 140 | `/model/stage2/stage2.0/fuse_layers.0.1/fuse_layers.0.1.0/Conv` | Conv | `[[16, 32, 1, 1], [16]]` | `[[1, 16, 144, 256]]` | 18,874,368 |
| 142 | `/model/stage2/stage2.0/fuse_layers.0.1/fuse_layers.0.1.2/Resize` | Resize | `[]` | `[[1, 16, 288, 512]]` | 0 |
| 145 | `/model/stage2/stage2.0/fuse_layers.1.0/fuse_layers.1.0.0/fuse_layers.1.0.0.0/Conv` | Conv | `[[32, 16, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 169,869,312 |
| 148 | `/model/transition2.2/transition2.2.0/transition2.2.0.0/Conv` | Conv | `[[64, 32, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 169,869,312 |
| 150 | `/model/stage3/stage3.0/branches.0/branches.0.0/conv1/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 152 | `/model/stage3/stage3.0/branches.0/branches.0.0/conv2/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 166 | `/model/stage3/stage3.0/branches.0/branches.0.0/se/fc/fc.0/MatMul` | MatMul | `[[16, 1]]` | `[[1, 1]]` | 16 |
| 168 | `/model/stage3/stage3.0/branches.0/branches.0.0/se/fc/fc.2/MatMul` | MatMul | `[[1, 16]]` | `[[1, 16]]` | 16 |
| 181 | `/model/stage3/stage3.0/branches.0/branches.0.1/conv1/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 183 | `/model/stage3/stage3.0/branches.0/branches.0.1/conv2/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 197 | `/model/stage3/stage3.0/branches.0/branches.0.1/se/fc/fc.0/MatMul` | MatMul | `[[16, 1]]` | `[[1, 1]]` | 16 |
| 199 | `/model/stage3/stage3.0/branches.0/branches.0.1/se/fc/fc.2/MatMul` | MatMul | `[[1, 16]]` | `[[1, 16]]` | 16 |
| 212 | `/model/stage3/stage3.0/branches.1/branches.1.0/conv1/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 214 | `/model/stage3/stage3.0/branches.1/branches.1.0/conv2/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 228 | `/model/stage3/stage3.0/branches.1/branches.1.0/se/fc/fc.0/MatMul` | MatMul | `[[32, 2]]` | `[[1, 2]]` | 64 |
| 230 | `/model/stage3/stage3.0/branches.1/branches.1.0/se/fc/fc.2/MatMul` | MatMul | `[[2, 32]]` | `[[1, 32]]` | 64 |
| 243 | `/model/stage3/stage3.0/branches.1/branches.1.1/conv1/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 245 | `/model/stage3/stage3.0/branches.1/branches.1.1/conv2/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 259 | `/model/stage3/stage3.0/branches.1/branches.1.1/se/fc/fc.0/MatMul` | MatMul | `[[32, 2]]` | `[[1, 2]]` | 64 |
| 261 | `/model/stage3/stage3.0/branches.1/branches.1.1/se/fc/fc.2/MatMul` | MatMul | `[[2, 32]]` | `[[1, 32]]` | 64 |
| 274 | `/model/stage3/stage3.0/branches.2/branches.2.0/conv1/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 276 | `/model/stage3/stage3.0/branches.2/branches.2.0/conv2/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 290 | `/model/stage3/stage3.0/branches.2/branches.2.0/se/fc/fc.0/MatMul` | MatMul | `[[64, 4]]` | `[[1, 4]]` | 256 |
| 292 | `/model/stage3/stage3.0/branches.2/branches.2.0/se/fc/fc.2/MatMul` | MatMul | `[[4, 64]]` | `[[1, 64]]` | 256 |
| 305 | `/model/stage3/stage3.0/branches.2/branches.2.1/conv1/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 307 | `/model/stage3/stage3.0/branches.2/branches.2.1/conv2/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 321 | `/model/stage3/stage3.0/branches.2/branches.2.1/se/fc/fc.0/MatMul` | MatMul | `[[64, 4]]` | `[[1, 4]]` | 256 |
| 323 | `/model/stage3/stage3.0/branches.2/branches.2.1/se/fc/fc.2/MatMul` | MatMul | `[[4, 64]]` | `[[1, 64]]` | 256 |
| 336 | `/model/stage3/stage3.0/fuse_layers.0.1/fuse_layers.0.1.0/Conv` | Conv | `[[16, 32, 1, 1], [16]]` | `[[1, 16, 144, 256]]` | 18,874,368 |
| 338 | `/model/stage3/stage3.0/fuse_layers.0.1/fuse_layers.0.1.2/Resize` | Resize | `[]` | `[[1, 16, 288, 512]]` | 0 |
| 340 | `/model/stage3/stage3.0/fuse_layers.0.2/fuse_layers.0.2.0/Conv` | Conv | `[[16, 64, 1, 1], [16]]` | `[[1, 16, 72, 128]]` | 9,437,184 |
| 342 | `/model/stage3/stage3.0/fuse_layers.0.2/fuse_layers.0.2.2/Resize` | Resize | `[]` | `[[1, 16, 288, 512]]` | 0 |
| 345 | `/model/stage3/stage3.0/fuse_layers.1.0/fuse_layers.1.0.0/fuse_layers.1.0.0.0/Conv` | Conv | `[[32, 16, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 169,869,312 |
| 347 | `/model/stage3/stage3.0/fuse_layers.1.2/fuse_layers.1.2.0/Conv` | Conv | `[[32, 64, 1, 1], [32]]` | `[[1, 32, 72, 128]]` | 18,874,368 |
| 349 | `/model/stage3/stage3.0/fuse_layers.1.2/fuse_layers.1.2.2/Resize` | Resize | `[]` | `[[1, 32, 144, 256]]` | 0 |
| 352 | `/model/stage3/stage3.0/fuse_layers.2.0/fuse_layers.2.0.0/fuse_layers.2.0.0.0/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 144, 256]]` | 84,934,656 |
| 354 | `/model/stage3/stage3.0/fuse_layers.2.0/fuse_layers.2.0.1/fuse_layers.2.0.1.0/Conv` | Conv | `[[64, 16, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 84,934,656 |
| 355 | `/model/stage3/stage3.0/fuse_layers.2.1/fuse_layers.2.1.0/fuse_layers.2.1.0.0/Conv` | Conv | `[[64, 32, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 169,869,312 |
| 359 | `/model/transition3.3/transition3.3.0/transition3.3.0.0/Conv` | Conv | `[[128, 64, 3, 3], [128]]` | `[[1, 128, 36, 64]]` | 169,869,312 |
| 361 | `/model/stage4/stage4.0/branches.0/branches.0.0/conv1/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 363 | `/model/stage4/stage4.0/branches.0/branches.0.0/conv2/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 377 | `/model/stage4/stage4.0/branches.0/branches.0.0/se/fc/fc.0/MatMul` | MatMul | `[[16, 1]]` | `[[1, 1]]` | 16 |
| 379 | `/model/stage4/stage4.0/branches.0/branches.0.0/se/fc/fc.2/MatMul` | MatMul | `[[1, 16]]` | `[[1, 16]]` | 16 |
| 392 | `/model/stage4/stage4.0/branches.0/branches.0.1/conv1/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 394 | `/model/stage4/stage4.0/branches.0/branches.0.1/conv2/Conv` | Conv | `[[16, 16, 3, 3], [16]]` | `[[1, 16, 288, 512]]` | 339,738,624 |
| 408 | `/model/stage4/stage4.0/branches.0/branches.0.1/se/fc/fc.0/MatMul` | MatMul | `[[16, 1]]` | `[[1, 1]]` | 16 |
| 410 | `/model/stage4/stage4.0/branches.0/branches.0.1/se/fc/fc.2/MatMul` | MatMul | `[[1, 16]]` | `[[1, 16]]` | 16 |
| 423 | `/model/stage4/stage4.0/branches.1/branches.1.0/conv1/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 425 | `/model/stage4/stage4.0/branches.1/branches.1.0/conv2/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 439 | `/model/stage4/stage4.0/branches.1/branches.1.0/se/fc/fc.0/MatMul` | MatMul | `[[32, 2]]` | `[[1, 2]]` | 64 |
| 441 | `/model/stage4/stage4.0/branches.1/branches.1.0/se/fc/fc.2/MatMul` | MatMul | `[[2, 32]]` | `[[1, 32]]` | 64 |
| 454 | `/model/stage4/stage4.0/branches.1/branches.1.1/conv1/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 456 | `/model/stage4/stage4.0/branches.1/branches.1.1/conv2/Conv` | Conv | `[[32, 32, 3, 3], [32]]` | `[[1, 32, 144, 256]]` | 339,738,624 |
| 470 | `/model/stage4/stage4.0/branches.1/branches.1.1/se/fc/fc.0/MatMul` | MatMul | `[[32, 2]]` | `[[1, 2]]` | 64 |
| 472 | `/model/stage4/stage4.0/branches.1/branches.1.1/se/fc/fc.2/MatMul` | MatMul | `[[2, 32]]` | `[[1, 32]]` | 64 |
| 485 | `/model/stage4/stage4.0/branches.2/branches.2.0/conv1/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 487 | `/model/stage4/stage4.0/branches.2/branches.2.0/conv2/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 501 | `/model/stage4/stage4.0/branches.2/branches.2.0/se/fc/fc.0/MatMul` | MatMul | `[[64, 4]]` | `[[1, 4]]` | 256 |
| 503 | `/model/stage4/stage4.0/branches.2/branches.2.0/se/fc/fc.2/MatMul` | MatMul | `[[4, 64]]` | `[[1, 64]]` | 256 |
| 516 | `/model/stage4/stage4.0/branches.2/branches.2.1/conv1/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 518 | `/model/stage4/stage4.0/branches.2/branches.2.1/conv2/Conv` | Conv | `[[64, 64, 3, 3], [64]]` | `[[1, 64, 72, 128]]` | 339,738,624 |
| 532 | `/model/stage4/stage4.0/branches.2/branches.2.1/se/fc/fc.0/MatMul` | MatMul | `[[64, 4]]` | `[[1, 4]]` | 256 |
| 534 | `/model/stage4/stage4.0/branches.2/branches.2.1/se/fc/fc.2/MatMul` | MatMul | `[[4, 64]]` | `[[1, 64]]` | 256 |
| 547 | `/model/stage4/stage4.0/branches.3/branches.3.0/conv1/Conv` | Conv | `[[128, 128, 3, 3], [128]]` | `[[1, 128, 36, 64]]` | 339,738,624 |
| 549 | `/model/stage4/stage4.0/branches.3/branches.3.0/conv2/Conv` | Conv | `[[128, 128, 3, 3], [128]]` | `[[1, 128, 36, 64]]` | 339,738,624 |
| 563 | `/model/stage4/stage4.0/branches.3/branches.3.0/se/fc/fc.0/MatMul` | MatMul | `[[128, 8]]` | `[[1, 8]]` | 1,024 |
| 565 | `/model/stage4/stage4.0/branches.3/branches.3.0/se/fc/fc.2/MatMul` | MatMul | `[[8, 128]]` | `[[1, 128]]` | 1,024 |
| 578 | `/model/stage4/stage4.0/branches.3/branches.3.1/conv1/Conv` | Conv | `[[128, 128, 3, 3], [128]]` | `[[1, 128, 36, 64]]` | 339,738,624 |
| 580 | `/model/stage4/stage4.0/branches.3/branches.3.1/conv2/Conv` | Conv | `[[128, 128, 3, 3], [128]]` | `[[1, 128, 36, 64]]` | 339,738,624 |
| 594 | `/model/stage4/stage4.0/branches.3/branches.3.1/se/fc/fc.0/MatMul` | MatMul | `[[128, 8]]` | `[[1, 8]]` | 1,024 |
| 596 | `/model/stage4/stage4.0/branches.3/branches.3.1/se/fc/fc.2/MatMul` | MatMul | `[[8, 128]]` | `[[1, 128]]` | 1,024 |
| 609 | `/model/stage4/stage4.0/fuse_layers.0.1/fuse_layers.0.1.0/Conv` | Conv | `[[16, 32, 1, 1], [16]]` | `[[1, 16, 144, 256]]` | 18,874,368 |
| 611 | `/model/stage4/stage4.0/fuse_layers.0.1/fuse_layers.0.1.2/Resize` | Resize | `[]` | `[[1, 16, 288, 512]]` | 0 |
| 613 | `/model/stage4/stage4.0/fuse_layers.0.2/fuse_layers.0.2.0/Conv` | Conv | `[[16, 64, 1, 1], [16]]` | `[[1, 16, 72, 128]]` | 9,437,184 |
| 615 | `/model/stage4/stage4.0/fuse_layers.0.2/fuse_layers.0.2.2/Resize` | Resize | `[]` | `[[1, 16, 288, 512]]` | 0 |
| 617 | `/model/stage4/stage4.0/fuse_layers.0.3/fuse_layers.0.3.0/Conv` | Conv | `[[16, 128, 1, 1], [16]]` | `[[1, 16, 36, 64]]` | 4,718,592 |
| 619 | `/model/stage4/stage4.0/fuse_layers.0.3/fuse_layers.0.3.2/Resize` | Resize | `[]` | `[[1, 16, 288, 512]]` | 0 |
| 622 | `/model/final_layers.0/Conv` | Conv | `[[3, 16, 1, 1], [3]]` | `[[1, 3, 288, 512]]` | 7,077,888 |
