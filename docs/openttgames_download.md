# 下载 TTNet / OpenTTGames 数据集

TTNet 论文使用的公开数据集为 [OpenTTGames](https://lab.osai.ai/)，包含
5 个训练视频、7 个测试视频，以及球坐标、事件和分割标注。官网列出的文件总量约 35 GB；
标注解压后还需额外空间。下载脚本使用官网链接，依赖 Python 3.10+ 和系统 `curl`，
无需安装 Python 包。

```bash
# 下载全部视频及标注，自动解压标注
python3 scripts/download_ttnet.py

# 仅训练集
python3 scripts/download_ttnet.py --split train

# 先下载一个较小的测试视频及标注
python3 scripts/download_ttnet.py --split test --games test_2

# 仅下载全部标注
python3 scripts/download_ttnet.py --only annotations

# 查看下载清单，不写文件
python3 scripts/download_ttnet.py --dry-run
```

默认保存到项目根目录的 `data/openttgames/`，与运行命令时的工作目录无关：

```text
data/openttgames/
├── training/
│   ├── videos/game_1.mp4 ... game_5.mp4
│   └── annotations/       # game_1.zip、game_1/ ...；每场比赛单独解压
└── test/
    ├── videos/test_1.mp4 ... test_7.mp4
    └── annotations/       # test_1.zip、test_1/ ...
```

下载期间使用 `.part` 文件，成功后才改为正式文件名。中断后重跑相同命令即可续传；
已完成文件会进行基础格式校验并跳过下载，ZIP 还会校验 CRC。MP4 文件头校验不等于
完整视频解码或官方哈希校验。标注再次解压会覆盖相同路径的文件，请在其他目录编辑标注。
脚本不会抽帧，也不会自动把此数据集加入项目现有 TTNet 训练管线。

可用 `--output-dir PATH` 更改目录，`--no-extract` 仅保留 ZIP，
`--base-url URL` 指定包含同名文件的下载源。代理使用 `curl` 的 `https_proxy` 等环境变量。
数据集授权为 CC BY-NC-SA 4.0，详见官网。
