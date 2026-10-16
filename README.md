工作区已统一为共享代码与方案配置。研究入口见 [工作区说明](docs/workspace.md)、[实验索引](experiments/index.md) 和 [后续计划](docs/next_steps.md)。当前训练画面使用分块NPY；历史文件归档保留。

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

## 本地 Blackwell 开发环境

RTX 5070 Ti 可用 `uv sync --locked` 安装独立的 Python 3.10 / PyTorch cu128 环境。
本地推理使用 ONNX Runtime CUDA 入口；安装、自检和运行命令见
[本地环境说明](docs/local_environment.md)。官方提交仍使用下述固定镜像与原有入口。

## 数据准备与跨环境重建

将 `pp_train_data.zip`、`pp_public_data.zip` 放在项目根目录，执行
`python3 scripts/prepare_data.py`，会自动准备两份 PP 数据并下载 TTNet / OpenTTGames 到
根目录 `data/`。本机已有数据会复用，也可以用 `--export-pp data/_archives` 导出 PP 迁移包。
目录、SHA256 校验和后续训练输入生成方法见 [数据准备说明](docs/prepare_data.md)。

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
