# 重建本机原始数据

`scripts/prepare_data.py` 使用 Python 3.10+ 和系统 `curl`，默认与命令执行目录无关。
TTNet 使用 [OpenTTGames 官方下载源](https://lab.osai.ai/)，下载全部 5 个训练视频、
7 个测试视频及对应标注；支持断点续传、失败重试和逐比赛解压，避免不同比赛的同名标注相互覆盖。

## 本机准备与迁移

```bash
python3 scripts/prepare_data.py

# 导出本机两份 PP 数据；ZIP 不重新编码视频，保持原始字节
python3 scripts/prepare_data.py --export-pp data/_archives
```

在其他环境检出同一份代码，将导出的 `pp_train_data.zip`、`pp_public_data.zip`
放在项目根目录或 `data/_archives/`，然后执行：

```bash
python3 scripts/prepare_data.py
```

脚本会自动识别 PP 压缩包、校验并解压，同时从官网下载 TTNet。
只复制 Git 仓库无法恢复 PP 视频；目前仓库没有 PP 官方下载地址，必须携带数据包，
或通过下面的参数提供下载链接。缺少 PP 来源时会在下载大体积 TTNet 前报错。

```text
data/
├── ttnet/                      # TTNet 的原始 OpenTTGames 数据
│   ├── training/
│   │   ├── videos/game_1.mp4 … game_5.mp4
│   │   └── annotations/game_1.zip、game_1/ …
│   └── test/
│       ├── videos/test_1.mp4 … test_7.mp4
│       └── annotations/test_1.zip、test_1/ …
├── pp_train_data/              # 154 视频 + 154 JSON 标注
│   ├── rallies_videos/00/ … 24/
│   └── annotations/00/ … 24/
├── pp_public_data/             # 15 视频 + manifest.json + gt_public.jsonl
│   └── videos/0度/、45度/、90度/
└── _archives/                  # 可选的 PP 迁移/下载包
```

`pp_tarin_data` 按现有训练代码统一为 `pp_train_data`，也可识别这个拼写的目录/压缩包。
本机已有的 `data/openttgames`、`data/public_data` 会在同一文件系统内重命名，
并保留指向新目录的相对符号链接。不会复制几十 GB 的 TTNet 视频。
原始数据不会删除；历史平铺解压的标注可以继续保留，但新环境只生成逐比赛隔离的标注。

## 查找来源与手动指定

自动查找项目根目录、`data/`、`data/_archives/` 下同名 `.zip`、`.tar`、`.tar.gz`、
`.tgz` 和目录。PP 公开集还支持项目内 `pingpang/public_data/` 或 `participant.zip`。
工程包中的篮球公开集不会被识别为乒乓球公开集。支持压缩包外面再套目录。

```bash
# 指定已有目录或压缩包
python3 scripts/prepare_data.py \
  --train-source /path/to/pp_train_data.zip \
  --public-source /path/to/participant.zip

# 指定可直接下载 ZIP/TAR 的 URL；curl 代理环境变量同样适用
python3 scripts/prepare_data.py \
  --train-source 'https://your-host/pp_train_data.zip' \
  --public-source 'https://your-host/pp_public_data.zip'

# 只准备其中的数据
python3 scripts/prepare_data.py --datasets pp_train_data pp_public_data
python3 scripts/prepare_data.py --datasets ttnet

# 查看来源和 TTNet 的完整下载清单，不下载或写文件
python3 scripts/prepare_data.py --dry-run

# 只检查原始文件 SHA256 和已解压 TTNet 标注 CRC，不写文件
python3 scripts/prepare_data.py --check-only
```

`scripts/data_manifest.json` 保存本机的相对文件名、大小和 SHA256：TTNet 的 24 个
下载文件、PP 训练集的 308 个文件、PP 公开集的 17 个文件。每次准备都校验源文件身份，
不一致就报错，不会覆盖已有 PP 文件。成功下载的 TTNet ZIP 会按比赛重新解压；
`--check-only` 也会核对解压内容与 ZIP 中的 CRC，发现缺失或改动。
清单校验的是原始数据内容，不包含历史解压残留、训练缓存、模型或实验输出。
SHA256 会顺序读取约 36 GB 原始数据，需要一定时间；完整训练帧生成另需空间。

## 后续训练

现有 YOLO 和 TTNet 数据管线读取的 `data/pp_train_data` 路径保持可用。
PP 公开集是落点事件验证集，不会被当作球框训练数据。
本脚本下载的 OpenTTGames 是独立的原始数据，当前球定位管线仍使用 PP 固定划分，
不会自动将 OpenTTGames 混入训练。

在安装项目训练依赖后，可以用当前 Python 环境生成 TTNet 训练帧、标签和时序窗口：

```bash
uv sync --locked
.venv/bin/python scripts/prepare_data.py --prepare-training --workers 4

# 也可单独运行已有的数据管线
.venv/bin/python -m ttnet.data_tools.prepare --workers 4
.venv/bin/python -m ttnet.data_tools.audit --full-images
```

生成的数据沿用 TTNet 现有默认位置 `ttnet/data/datasets/ball_v2`，以及固定的
114 train / 40 val 视频清单。处理版本和抽帧依赖同样会被记录。
`--prepare-training` 只生成训练输入，不启动训练；训练方法见 [TTNet 说明](../ttnet/README.md)。
用 `data/pp_public_data` 跑赛事验证时，显式传给框架 `--input` 即可；
若新环境缺少 `pingpang/public_data`，脚本自动创建指向 `data/pp_public_data` 的相对链接，
供现有评估入口使用；本机已有的公开数据目录继续保留。

## 脚本验证

```bash
python3 -m unittest discover -s scripts/tests -v
```

测试不访问网络，用小型数据验证完整重建、重复执行、源哈希校验、目录兼容、
ZIP/TAR 与工程包来源，以及路径越界和符号链接拦截。
