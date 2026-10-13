# 乒乓球落点检测赛 · 比赛工程（含附加题：篮球球员跟踪）

主赛题为乒乓球落点检测，附加题为篮球球员跟踪（自愿选做，成绩不影响主赛排名）；
两题共用纯环境镜像 `sport-base:v1`（只有运行环境，不含本工程）。
本工程构建提交镜像时放到 `/participant`；本地自测也可直接挂载到 `/participant`。

## 目录结构

```text
participant/
├── run_all.sh           联合入口：依次执行两任务（评测一次 docker run 跑完）
├── pingpang/            乒乓球落点检测（框架独立自包含）
│   ├── core/              框架：runner/decoder/types/...（参赛者不改）
│   ├── run.py             单任务运行入口（run.sh/validate.py/integrity.py 为配套）
│   ├── scripts/           自检：check_env.py / check_solution.py
│   ├── public_data/       验证集（15 视频 + manifest + GT，开箱自测）
│   └── participant/       参赛者代码区（内置官方参考实现，可整体替换）
└── basketball/          篮球球员跟踪（结构同上；验证集 2 视频 main+switch）
```

两任务 core 同构独立，互不依赖——单任务出问题不波及另一个。

## 本地自测（在 participant/ 所在目录执行）

```bash
# 双任务联合（与评测形态一致；验证集模拟 /participant/input/{pingpang,basketball}）
docker run --rm --gpus all \
    -v $PWD/participant:/participant:ro \
    -v $PWD/participant/pingpang/public_data:/participant/input/pingpang:ro \
    -v $PWD/participant/basketball/public_data:/participant/input/basketball:ro \
    -v $PWD/out/joint_demo:/participant/output \
    sport-base:v1

# 单任务（以乒乓球为例，篮球换路径）
docker run --rm --gpus all \
    -v $PWD/participant:/participant -v $PWD/out:/participant/output \
    --entrypoint python3 sport-base:v1 \
    /participant/pingpang/run.py \
        --input /participant/pingpang/public_data --output /participant/output/pingpang_demo \
        --solution participant.solution:Solution --decoder gpu
```

## 提交形式与规则

提交物为基于 `sport-base:v1` 的镜像，`COPY participant/ /participant/`
（详见随包《作品形式&提交流程.md》）。每个任务在 `participant/<task>/participant/solution.py`
实现 `Solution` 类（接口见该任务 `core/types.py`），模型放 `participant/weights/`。
只做一个任务时，另一任务保持内置参考实现，评测按参考实现计分。

规则：只改各任务的 `participant/` 代码区；不改 `core/` 与框架脚本（完整性校验不过则无成绩）；
不升级 torch / numpy / onnxruntime / tensorrt（评测环境固定，无安装环节）。

任务细节（数据/指标/FAQ）见《赛题解析.md》。
