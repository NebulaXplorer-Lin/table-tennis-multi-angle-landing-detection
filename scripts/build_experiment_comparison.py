"""Build a dated, evidence-backed comparison without changing historical runs."""
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]
RECORD = PROJECT / 'experiments/records/20261016_experiment_comparison'
DOCUMENT = PROJECT / 'docs/experiment_comparison_20261016.md'
STUDY = PROJECT / 'out/experiments/20261016-153213_event_comparison_b8_seed42'
TRAJECTORY = PROJECT / 'out/results/20261016-154803_trajectory_priority_seed42'
SOURCES = {}
LINKS = set()


def path(value):
    p = Path(value)
    return p if p.is_absolute() else PROJECT / p


def read(value):
    p = path(value)
    content = p.read_bytes()
    SOURCES[str(p.relative_to(PROJECT))] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def read_rows(value):
    p = path(value)
    content = p.read_bytes()
    SOURCES[str(p.relative_to(PROJECT))] = hashlib.sha256(content).hexdigest()
    return [json.loads(line) for line in content.decode().splitlines() if line.strip()]


def link(label, value):
    p = path(value)
    if not p.exists() and p not in (RECORD / 'inventory.json', RECORD / 'sources.json'):
        raise FileNotFoundError(p)
    LINKS.add(str(p))
    return f'[{label}](<{p}>)'


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def pct(value):
    return '—' if value in (None, '') else f'{float(value):.2%}'


def number(value, places=2):
    return '—' if value in (None, '') else f'{float(value):.{places}f}'


def dump(value, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    RECORD.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    pp = read('data/datasets/pp_ball_v2/metadata/dataset.json')
    pp_old = read('archive/20261015-102535_workspace/ttnet/data/datasets/ball_v1/metadata/dataset.json')
    ott = read('data/datasets/openttgames_event25_v3/dataset.json')
    split = read('data/datasets/openttgames_event25_v3/split.json')
    annotation = read('data/annotations/pp_bounce_confirmed_v1.json')
    public = read('data/pp_public_data/manifest.json')['videos']
    public_gt = read_rows('data/pp_public_data/gt_public.jsonl')
    pp_counts = {part: Counter(r['label_status'] for r in read_rows(f'data/datasets/pp_ball_v2/labels/{part}.jsonl')) for part in ('train', 'val')}
    event_counts = {}
    for part in ('train', 'val'):
        videos = [v for v in annotation['videos'] if v['split'] == part]
        events = [e for v in videos for e in v['events'] if e['kind'] == 'bounce']
        event_counts[part] = dict(videos=len(videos), bounces=len(events), spatial=sum(e.get('spatial_valid', True) for e in events),
                                  views=dict(Counter(v['view'] for v in videos)))
    ott_counts = {}
    for part in ('train', 'val', 'test'):
        samples = [s for vid in split[part] for s in read_rows(f'data/datasets/openttgames_event25_v3/{vid}/samples.jsonl')]
        ott_counts[part] = dict(videos=split[part], expanded_windows=len(samples),
            anchors=dict(Counter(s['group'] for s in samples if s['original'])))
    current = {k: read(STUDY.with_name(STUDY.name + '_' + k) / 'evaluations/continuous/metrics.json') for k in 'ABCD'}
    current['A0'] = read(STUDY / 'A0_continuous/metrics.json')
    best = {k: read(STUDY.with_name(STUDY.name + '_' + k) / 'best_selection.json') for k in 'ABCD'}
    states = {k: read(STUDY.with_name(STUDY.name + '_' + k) / 'run.json') for k in 'ABCD'}
    histories = {}
    for k in 'ABCD':
        run = STUDY.with_name(STUDY.name + '_' + k)
        histories[k] = [row for f in sorted((run / 'attempts').glob('*/epochs.jsonl')) for row in read_rows(f)]
    old_event = read('ttnet/runs/20261014-181231_event25_frozen/evaluations/best/metrics.json')
    joint = read('ttnet/runs/20261014-181231_event25_joint/evaluations/epoch002_public/public/metrics.json')
    yolo_public = read('experiments/records/20261013-215749_yolo11n_full1280_public/evaluations/epoch032_public_paused.json')
    official = read('experiments/records/20261013-212318_baseline_ball-position/evaluations/onnx_fp32_ball.json')
    ball_runs = ['20261013-234947_global_ball', '20261014-095202_global_fixed3_ball', '20261014-103011_local_fixed3_ball']
    balls = {name: read(f'ttnet/runs/{name}/evaluations/best_val/metrics.json') for name in ball_runs}
    ball_states = {name: read(f'ttnet/runs/{name}/run.json') for name in ball_runs}
    ball_bench = {name: read(f'ttnet/runs/{name}/evaluations/benchmark/metrics.json') for name in ball_runs}
    trajectory = read(TRAJECTORY / 'metrics.json')
    candidate = read('out/results/20261015_fixed3_candidates_full/summary.json')
    batch = read('out/results/20261016_training_preflight/batch_benchmark.json')
    warm_runtime = read('out/results/20261015_comparison_cuda_runtime_native/metrics.json')
    trajectory_runtime = read(TRAJECTORY / 'runtime.json')
    trajectory_strata = read(TRAJECTORY / 'stratified.json')
    evidence = read('out/results/20261016_transformer_diagnosis/evidence.json')
    temporal_diag = read('out/results/20261016_transformer_diagnosis/temporal_tolerance_diagnostic.json')
    fields = ['experiment_id', 'eval_id', 'section', 'scheme', 'status', 'training_dataset', 'evaluation_dataset', 'split',
              'scope', 'protocol', 'seed', 'epoch', 'threshold', 'precision', 'recall', 'f1', 'macro_f1', 'tp', 'fp', 'fn',
              'bounce_f1', 'net_f1', 'e2e_fps', 'component_fps', 'component_ms', 'roi_any_coverage', 'roi_selected_coverage',
              'peak_memory_mib', 'checkpoint_sha256', 'source', 'notes']
    rows = []

    def add(**values):
        unknown = set(values) - set(fields)
        if unknown:raise ValueError(unknown)
        rows.append({key: values.get(key, '') for key in fields})

    legacy_file = PROJECT / 'experiments/comparison.csv'
    SOURCES[str(legacy_file.relative_to(PROJECT))] = hashlib.sha256(legacy_file.read_bytes()).hexdigest()
    legacy = list(csv.DictReader(legacy_file.open()))
    for row in legacy:
        add(experiment_id=row['experiment_id'], eval_id=row['eval_id'], section='historical', scheme=row['model'],
            status=row['status'], training_dataset='PP全量154视频/20846有球帧' if 'full1280' in row['experiment_id'] else ('COCO预训练+PP采样3162帧' if row['model']=='YOLO11n' else '官方权重训练来源未知'),
            evaluation_dataset='PP public 15视频/92有效落点' if row['task']=='landing_detection' else 'PP固定val 40视频/5302有球帧',
            split='public' if row['task']=='landing_detection' else 'val', scope=row['task'], protocol=row['protocol'],
            seed=row['seed'], epoch=32 if 'full1280' in row['experiment_id'] else row['best_epoch'],
            precision=row['center_p'], recall=row['center_r'], f1=row['center_f1'], macro_f1=row['touch_f1'],
            e2e_fps=row['e2e_fps'], peak_memory_mib=row['peak_memory_mib'], checkpoint_sha256=row['checkpoint_sha256'],
            source=row['evaluation'] or 'experiments/records/'+row['experiment_id']+'/summary.md', notes=row['notes'])
    add(experiment_id='20261013-212318_baseline_ball-position',eval_id='onnx_fp32_ball_50px',section='ball',scheme='官方三帧ONNX',
        status='completed',training_dataset='训练来源未知',evaluation_dataset='PP固定val 40视频/5302有球帧',split='val',scope='ball_center',protocol='50原图px',
        **{k:official['center']['50'][k] for k in ('precision','recall','f1','tp','fp','fn')},source='experiments/records/20261013-212318_baseline_ball-position/evaluations/onnx_fp32_ball.json')
    for name in ball_runs:
        for tolerance in ('10','50'):
            metric=balls[name]['center'][tolerance]
            add(experiment_id=name,eval_id='best_val_'+tolerance+'px',section='ball',scheme=name.split('_',1)[1],status='completed',
                training_dataset='PP固定train 114视频/15544有球帧',evaluation_dataset='PP固定val 40视频/5302有球帧',split='val',scope='ball_center',
                protocol=tolerance+'原图px; unknown不评分',seed=42,epoch=ball_states[name]['best_epoch'],
                **{k:metric[k] for k in ('precision','recall','f1','tp','fp','fn')},e2e_fps=ball_bench[name]['e2e_fps'],
                source=f'ttnet/runs/{name}/evaluations/best_val/metrics.json',notes='FPS为纯定位流水线，未包含事件头')
    for part in ('val','test'):
        metric=old_event[part]
        add(experiment_id='20261014-181231_event25_frozen',eval_id=part+'_sparse',section='sparse',scheme='A0历史',status='completed',
            training_dataset='OpenTTGames game_1–4',evaluation_dataset='OpenTTGames '+('game_5' if part=='val' else 'test_1–7'),split=part,
            scope='sparse_event_classification',protocol='标注anchor分类; 阈值来自OTT val',seed=42,epoch=12,threshold='0.1/0.3',
            bounce_f1=metric['bounce']['f1'],net_f1=metric['net']['f1'],macro_f1=metric['macro_f1'],
            source='ttnet/runs/20261014-181231_event25_frozen/evaluations/best/metrics.json')
    add(experiment_id='20261014-181231_event25_frozen',eval_id='public_transfer',section='public',scheme='A0历史',status='completed',
        training_dataset='OpenTTGames game_1–4; PP定位器train114视频',evaluation_dataset='PP public 15视频/92有效落点',split='public',scope='landing',
        protocol=old_event['public']['protocol'],seed=42,epoch=12,threshold=.1,macro_f1=old_event['public']['macro_f1'],
        tp=68,fp=157,fn=24,e2e_fps=old_event['public']['e2e_fps'],source='ttnet/runs/20261014-181231_event25_frozen/evaluations/best/metrics.json')
    jt={key:sum(v[key] for v in joint['by_angle'].values()) for key in ('tp','fp','fn')}
    add(experiment_id='20261014-181231_event25_joint',eval_id='epoch002_public',section='public',scheme='历史联合TCN',status='intermediate_paused',
        training_dataset='OpenTTGames game_1–4; 定位器共同微调',evaluation_dataset='PP public 15视频/92有效落点',split='public',scope='landing',
        protocol=joint['protocol'],seed=42,epoch=2,macro_f1=joint['macro_f1'],**jt,e2e_fps=joint['e2e_fps'],
        source='ttnet/runs/20261014-181231_event25_joint/evaluations/epoch002_public/public/metrics.json',notes='中间结果; 第3轮step400暂停; 测速与训练共享GPU')
    for k in ('A0','A','B','C','D'):
        m=current[k];source=m['config'].get('scheme','')
        result_path=(STUDY/'A0_continuous/metrics.json') if k=='A0' else STUDY.with_name(STUDY.name+'_'+k)/'evaluations/continuous/metrics.json'
        training='事件头仅OpenTTGames game_1–4; 定位器PP train114视频' if k=='A0' else 'OpenTTGames game_1–4 + PP train114视频613触台; 50/50采样'
        for scope in ('landing','continuous_bounce'):
            metric=m['metrics'][scope]
            add(experiment_id=STUDY.name+('_'+k if k!='A0' else ''),eval_id=k+'_'+scope,section='current',scheme=k,status='evaluated_seed42',
                training_dataset=training,evaluation_dataset='PP内部val40视频/6292帧/'+('189空间GT' if scope=='landing' else '203触台GT'),
                split='val',scope=scope,protocol='原帧±1; '+('原图50px; ' if scope=='landing' else '')+'最大基数一对一; unknown屏蔽',
                seed=42,epoch=m['epoch'],threshold=m['metrics']['threshold'],**{a:metric[a] for a in ('precision','recall','f1','tp','fp','fn')},
                macro_f1=m['metrics']['macro_f1'] if scope=='landing' else '',e2e_fps=m['e2e_fps'],
                peak_memory_mib=max(v['peak_cuda_bytes'] for v in m['performance'])/2**20,checkpoint_sha256=m['checkpoint_sha256'],source=str(result_path.relative_to(PROJECT)),
                notes='空间GT来自事件帧原球位置标注; 非人工接触坐标; 非独立测试成绩')
        if k!='A0':
            s=best[k]['val']
            add(experiment_id=STUDY.name+'_'+k,eval_id='best_pp_selected_ott_val_anchors',section='sparse',scheme=k,status='training_validation_snapshot',
                training_dataset=training,evaluation_dataset='OpenTTGames game_5 / 450原始anchor',split='val',scope='sparse_event_classification',
                protocol='PP落点选checkpoint和Bounce阈值; Net阈值0.5; 缓存验证',seed=42,epoch=m['epoch'],threshold=str(best[k]['thresholds']),
                bounce_f1=s['bounce']['f1'],net_f1=s['net']['f1'],macro_f1=s['macro_f1'],source=str((STUDY.with_name(STUDY.name+'_'+k)/'best_selection.json').relative_to(PROJECT)),
                notes='不是当前四组的OTT独立test; 与A0历史阈值选择方式不同')
    for k in 'AB':
        for tolerance in ('10','50'):
            s=trajectory['schemes'][k];m=s['localization'][tolerance]
            add(experiment_id=TRAJECTORY.name,eval_id=k+'_'+tolerance+'px',section='trajectory',scheme=k,status='completed_locator_diagnostic',
                training_dataset='固定已训练Global/Local; 轨迹组件无额外训练',evaluation_dataset='PP固定val40视频/5302有球帧',split='val',scope='ball_center',
                protocol=tolerance+'原图px; 同CUDA AMP',**{a:m[a] for a in ('precision','recall','f1','tp','fp','fn')},
                roi_any_coverage=s['roi_any_coverage'],roi_selected_coverage=s['roi_selected_coverage'],
                e2e_fps=trajectory_runtime['schemes'][k]['e2e_fps'],source=str((TRAJECTORY/'metrics.json').relative_to(PROJECT)),
                notes='FPS来自另跑40原视频+相同未训练TCN; 不采用含画图编码的audit耗时')
    for item in batch['feature']:
        add(experiment_id='20261016_training_preflight',eval_id='feature_batch_'+str(item['batch']),section='batch',scheme='independent',status='component_benchmark',
            evaluation_dataset='PP val 22_001前128帧',split='val_subset',scope='feature_extraction',protocol='CUDA AMP; batch='+str(item['batch']),
            component_fps=item['fps'],peak_memory_mib=item['peak_cuda_bytes']/2**20,source='out/results/20261016_training_preflight/batch_benchmark.json',notes='组件吞吐，不是完整视频FPS')
    for item in batch['event']:
        add(experiment_id='20261016_training_preflight',eval_id='head_microbatch_'+str(item['micro_batch']),section='batch',scheme='TCN',status='component_benchmark',
            evaluation_dataset='真实缓存特征窗口',scope='head_forward_backward',protocol='effective_batch8; micro='+str(item['micro_batch']),
            component_ms=item['optimizer_group_ms'],peak_memory_mib=item['peak_cuda_bytes']/2**20,
            source='out/results/20261016_training_preflight/batch_benchmark.json',notes='每有效batch8前后向，不含optimizer.step，不是epoch耗时')
    for k in 'ABCD':
        add(experiment_id='20261015_comparison_cuda_runtime_native',eval_id=k+'_two_video_runtime',section='runtime',scheme=k,status='initialized_head_runtime_only',
            evaluation_dataset='PP val两个720p原视频/762帧',split='val_subset',scope='runtime',protocol='原视频解码至JSON; 未训练事件头',
            e2e_fps=warm_runtime['schemes'][k]['e2e_fps'],source='out/results/20261015_comparison_cuda_runtime_native/metrics.json',notes='只测速，无事件准确率')
    csv_path=RECORD/'comparison.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    # Tables below are rendered from the persisted summary source where possible.
    flat=list(csv.DictReader(csv_path.open()))
    def find(section,scheme,scope=None,eval_id=None):
        selected=[r for r in flat if r['section']==section and r['scheme']==scheme and (scope is None or r['scope']==scope) and (eval_id is None or r['eval_id']==eval_id)]
        if len(selected)!=1:raise ValueError((section,scheme,scope,eval_id,len(selected)))
        return selected[0]

    dataset_summary=dict(pp_ball_dataset_id=pp['dataset_id'],historical_pp_ball_v1_dataset_id=pp_old['dataset_id'],pp_split_hashes=pp['split_hashes'],pp_ball_label_counts=pp_counts,
        pp_events=event_counts,pp_event_annotation_id=current['A']['annotation_id'],combined_training_dataset_id=current['A']['dataset_id'],
        openttgames_dataset_id=ott['dataset_id'],openttgames_split=split,openttgames_counts=ott_counts,
        public=dict(videos=len(public),frames=sum(v['n_frames'] for v in public),valid_events=sum(v.get('valid',True) for v in public_gt),ignored_events=sum(not v.get('valid',True) for v in public_gt)))
    dump(dataset_summary,RECORD/'datasets.json')
    text=['# 乒乓球定位与落点检测：已完成实验对比', '', f'截至 {timestamp}（北京时间）。汇总已有实际产物，本次不启动新训练、不重新选择阈值。', '',
        '当前结论：本轮同一 PP 内部验证集上，B（多候选轨迹关联＋TCN）最好，落点三视角宏平均 F1 为 **77.96%**，完整视频 **86.54 FPS**。这是一轮 seed42 验证成绩；多种子与当前四组的外部泛化评估尚未完成。', '',
        '历史官方 ONNX 定位器在 PP 相同验证视频上的10px球心F1为97.76%，仍高于自训练定位器；其训练来源未知，不能据此证明独立泛化。球定位、稀疏事件分类、连续触台和落点成绩分别列出。', '',
        '机器可读汇总：'+link('comparison.csv',csv_path)+'；数据清单与完整身份：'+link('datasets.json',RECORD/'datasets.json')+'。表中百分数由0～1原始值格式化，空白/“—”表示未测，不表示0。', '',
        '## 1. 数据集与划分', '',
        table(['编号','来源与版本','训练部分','验证/测试部分','标注与用途'],[
            ['D1','PP 球定位；pp_ball_v1/v2；同一固定视频划分','114视频；17398帧；15544有球GT','val 40视频；6292帧；5302有球GT、990 unknown','球心/框；无可靠无球负类，unknown不评分'],
            ['D2','PP 新增连续事件；pp_bounce_confirmed_v1','D1的114视频；613触台；600有空间参考','D1的40视频；203触台；189有空间参考','按用户确认触台已标全；ignore/空导出保持unknown；Net屏蔽'],
            ['D3','OpenTTGames event25_v3','game_1–4；4场；3352原始anchor','val game_5：450 anchor；test_1–7：464 anchor','25 FPS，13帧窗口；Bounce/Net/empty；稀疏分类'],
            ['D4','PP public：pingpang_landing_v1_public','未加入本轮事件训练','15视频；'+str(sum(v['n_frames'] for v in public))+'帧；92有效落点、12忽略标签','0°/45°/90°各5视频；历史迁移/规则落点评估'],
            ['D5','历史YOLO全量训练版本','PP全部154视频，20846有球标注帧','评估用D4；没有按D1留出内部val训练','与D1内部验证训练边界不同，不能并入同一公平训练排名']]), '',
        'D1本地训练与留出评估复用固定114/40清单；本轮未重分。官方权重仅在val评估，训练来源未知。v1/v2的数据身份不同，但train/val清单哈希一致。早期YOLO640只从训练部分按每5帧采样，实际3162张，val为5302张有球帧。', '',
        'D2的空间参考是**触台事件帧的原球心GT**，不是新增人工精标接触坐标；13个训练事件、14个验证事件缺球心，仍用于时间监督/评分，空间评分排除。`02_004`空标注导出整段保持unknown；显式ignore优先。front/diagonal/side为粗视觉分组，不等同D4精确角度名称。', '',
        table(['PP集合','front视频','diagonal视频','side视频','触台事件','空间可评分事件'],[[p,event_counts[p]['views'].get('front',0),event_counts[p]['views'].get('diagonal',0),event_counts[p]['views'].get('side',0),event_counts[p]['bounces'],event_counts[p]['spatial']] for p in ('train','val')]), '',
        'D3原始anchor计数：训练Bounce1289 / Net986 / empty1077；验证248 / 184 / 18；测试240 / 180 / 44。位移软标签扩展后训练窗口12452、验证2178、测试2144；扩展窗口不是独立事件数量。每个epoch平衡/混合采样，当前四组实际使用7734个窗口，两个来源各50%。', '',
        '数据入口：'+link('PP train清单','data/datasets/pp_ball_v2/splits/train.jsonl')+'；'+link('PP val清单','data/datasets/pp_ball_v2/splits/val.jsonl')+'；'+link('连续触台标签','data/annotations/pp_bounce_confirmed_v1.json')+'；'+link('TTGames场次划分','data/datasets/openttgames_event25_v3/split.json')+'；'+link('public清单','data/pp_public_data/manifest.json')+'。', '',
        '**public的历史使用情况：**它已被历史YOLO、冻结TCN和联合训练中间评估使用，也已据此分析过错误。本轮A/B/C/D尚未评估public，且本轮不使用其标签训练或选阈值；但不能把它描述为整个项目从未查看过的盲测集。', '',
        '## 2. 球定位：历史模型与三帧定位器', '',
        '以下精度均来自D1固定40个验证视频的已知有球标签；未知帧不当作无球。球心阈值为原图10px/50px。官方预训练来源未知；不同模型解码、边界处理与后端不同，表格用于历史对照。', '']
    ball_table=[['YOLO11n 640，conf .25','COCO＋D1采样3162帧','6 / 1','0.00%','—','严重漏检，主评估无有效框'],
                ['YOLO11n 640，conf .001事后诊断','同一权重','同上','2.35%','—','只69个输出/63个10px TP，非新训练'],
                ['官方三帧ONNX','来源未知；本地未重训','—',pct(official['center']['10']['f1']),pct(official['center']['50']['f1']),'原始官方解码，非本轮定位骨干']]
    names=['自训练单帧Global','自训练三帧Global','自训练三帧Global＋Local']
    for name,label in zip(ball_runs,names):
        r=ball_states[name]
        ball_table.append([label,'D1 train114视频',str(r['epochs_completed'])+' / '+str(r['best_epoch']),pct(balls[name]['center']['10']['f1']),pct(balls[name]['center']['50']['f1']),'固定工作阈值0.7'])
    text += [table(['方案','训练数据','完成轮/最佳轮','球心F1@10px','球心F1@50px','说明'],ball_table), '',
        'YOLO640框架记录AP50=20.85%、AP50–95=6.69%，但默认阈值球心召回为0；事后本地101点AP诊断为0.99%/0.43%，协议不同。原报告已指出低召回下的库PR尾部插值问题，这些AP数值不能替代实际球心命中率。热图模型没有bbox，AP留空。', '',
        '三帧Global到Local的10px F1提升约0.74个百分点、50px提升约0.59个百分点；增加Local只带来有限总体提升，候选与搜索区域错误仍是重要来源。', '',
        '来源：'+link('YOLO首轮与AP审计','experiments/records/20261013-163117_yolo11n_ball-640/summary.md')+'；'+link('官方定位完整报告','experiments/records/20261013-212318_baseline_ball-position/summary.md')+'；'+link('单帧Global','ttnet/output/20261013-234947_global_ball/summary.md')+'；'+link('三帧Global','ttnet/output/20261014-095202_global_fixed3_ball/summary.md')+'；'+link('三帧Global/Local','ttnet/output/20261014-103011_local_fixed3_ball/summary.md')+'。', '',
        '## 3. 多候选轨迹关联：不训练事件头的A/B定位对照', '',
        'D1同40视频、6292帧，固定同一Global/Local权重、CUDA AMP和工作阈值。B是规则关联器：Global Top5、最多8条假设、短历史代价、最多2个Local ROI；组件本身不额外训练。GT只用于诊断，不用于选候选。', '']
    def trrow(label, getter, percent=False):
        vals=[getter(trajectory['schemes'][k]) for k in 'AB']
        return [label]+[pct(v) if percent else v for v in vals]
    text += [table(['指标','A 独立解码','B 多候选关联'],[
        trrow('球心F1@10px',lambda r:r['localization']['10']['f1'],True),trrow('球心F1@50px',lambda r:r['localization']['50']['f1'],True),
        trrow('任一Local ROI覆盖',lambda r:r['roi_any_coverage'],True),trrow('最终所选ROI覆盖',lambda r:r['roi_selected_coverage'],True),
        trrow('触台±1帧球心F1@50px',lambda r:r['bounce_neighborhood']['localization']['50']['f1'],True),
        trrow('错误跳转次数',lambda r:r['erroneous_jumps']['count']),trrow('最长定位漏检段@50px（帧）',lambda r:r['consecutive_missed']['50']['max']),
        trrow('定位漏检段P90（帧）',lambda r:r['consecutive_missed']['50']['p90']),trrow('最长无有效观测段（帧）',lambda r:r['consecutive_no_observation']['max'])]), '',
        '50px评分中恢复299帧、退化114帧；错误跳转按相邻已知GT的预测位移偏差>50px统计，不把真实快速运动直接算成错误。三个粗视角和720p/1080p均提升。最长漏检仍22帧，剩余漏检段P90由3变5，说明仍有持续跟错问题。', '',
        '候选oracle诊断（历史CPU FP32，D1正标签5302帧），Global阈值0.7：', '',
        table(['候选数','10px候选命中','50px候选命中','候选ROI覆盖上限'],[[r['k'],pct(r['coverage_10px']),pct(r['coverage_50px']),pct(r['roi_coverage'])] for r in candidate['candidate']['threshold_tables']['0.7']]), '',
        'Top5 oracle可恢复Top1的341/848个10px漏命中（40.21%），这是GT诊断上限，不是实际关联增益。历史CPU诊断与当前CUDA AMP有4帧有效坐标差异>1px，因此两次A错误计数808/809不可当作同后端数值对齐。', '',
        link('A/B完整报告',TRAJECTORY/'summary.md')+'；'+link('恢复与退化逐帧清单',TRAJECTORY/'paired_changes.jsonl')+'；'+link('A全部错误视频',TRAJECTORY/'A/index.html')+'；'+link('B全部错误视频',TRAJECTORY/'B/index.html')+'；'+link('候选oracle完整产物','out/results/20261015_fixed3_candidates_full/index.html')+'。', '',
        '## 4. 本轮A0/A/B/C/D：训练设置与轮数', '',
        table(['方案','定位/轨迹','事件头','事件训练数据','实际完成/最佳轮'],[
            ['A0','独立解码','历史TCN','仅D3；本轮不重训','17 / 12（历史）'],
            *[[k,'轨迹关联' if k in 'BD' else '独立解码','Transformer' if k in 'CD' else 'TCN','D3＋D2，来源各50%',str(len(histories[k]))+' / '+str(best[k]['epoch'])] for k in 'ABCD']]), '',
        '四组均冻结现有三帧Global/Local；只随机初始化并训练事件头。TCN约17.56万参数；Transformer为2层Encoder、128维、4头、FFN256、dropout0.1，约29.98万参数。输入均为13帧、256维视觉描述符＋9维运动/有效性/真实时间信息。', '',
        'AdamW，lr=3e-4，weight_decay=1e-4，AMP，seed42，有效batch8、micro_batch8；最多30轮，patience5。PP完整验证的三视角落点宏F1选择checkpoint，同分优先FP更少；Bounce阈值网格0.1～0.9，Net默认0.5且PP不监督Net。A/C共用定位特征，B/D共用关联特征，窗口不重置轨迹状态。', '',
        '每轮7734个窗口、约967次更新；定位器特征提前缓存，因此事件头训练和缓存验证每轮约3～4秒。累计逐轮耗时A约98秒、B约53秒、C约57秒、D约93秒，不含特征准备、初始化和训练后原视频完整评估。', '',
        '本轮best checkpoint均已保存并完成完整视频评估。部分旧run.json仍写trained_pending_eval；此文依据实际metrics.json和队列DONE结果判定“已评估”，不将旧状态字段误当成未完成。', '',
        '## 5. 本轮主结果：PP内部验证集完整落点检测', '',
        '评估数据为D2 val的40完整视频/6292帧；189个空间GT。原视频帧号±1帧、原图距离≤50px、最大基数一对一匹配，unknown不计FP。主指标为front/diagonal/side各自F1的宏平均；表中P/R及TP/FP/FN为合并计数，因此与三视角宏F1不是同一个聚合方式。A0仅在当前PP val选择阈值，不更新权重。', '']
    event_table=[]
    for k in ('A0','A','B','C','D'):
        r=find('current',k,'landing')
        event_table.append([k,pct(r['macro_f1']),pct(r['precision']),pct(r['recall']),r['tp'],r['fp'],r['fn'],r['threshold']])
    text += [table(['方案','落点三视角宏F1','合并P','合并R','TP','FP','FN','Bounce阈值'],event_table), '',
        'B相对A：宏F1增加 **5.52个百分点**，FP由57降至41，FN由55降至41。A相对A0增加 **21.58个百分点**，包含新增PP监督和重新训练协议的收益，不能全部归于架构。D相对C提升9.78个百分点，关联在两种事件头上均有收益；C/D均未超过对应TCN。', '',
        table(['方案','front落点F1','diagonal落点F1','side落点F1','720p合并落点F1','1080p合并落点F1'],[[k,*[pct(current[k]['metrics']['by_view'][v]['f1']) for v in ('front','diagonal','side')],*[pct(current[k]['metrics']['by_resolution'][v]['f1']) for v in ('1280x720','1920x1080')]] for k in ('A0','A','B','C','D')]), '',
        'B相对A三个视角均提升。1080p空间GT仅28个，720p为161个，分辨率子组应结合样本量解释。最终落点使用事件时刻预测球心，不包含额外人工接触点回归。', '',
        '## 6. 连续触台与TTGames稀疏分类分别看', '',
        '连续触台只判断时间，203个GT；不要求球坐标存在。数据仍为D2 PP val，不能与D3稀疏分类混为同一种F1。', '',
        table(['方案','连续触台P','连续触台R','连续触台F1（合并）','TP / FP / FN'],[[k,pct(current[k]['metrics']['continuous_bounce']['precision']),pct(current[k]['metrics']['continuous_bounce']['recall']),pct(current[k]['metrics']['continuous_bounce']['f1']), ' / '.join(str(current[k]['metrics']['continuous_bounce'][a]) for a in ('tp','fp','fn'))] for k in ('A0','A','B','C','D')]), '',
        'TTGames分类：只统计原始已标注anchor。历史A0在TTGames val选BCE和阈值；当前A/B/C/D在PP val选checkpoint与Bounce阈值、Net固定0.5。两批阈值/选模协议不同，下面不构成严格的同协议架构排名。', '',
        table(['模型','评估数据','样本数','Bounce F1','Net F1','两类宏F1','记录性质'],[
            ['A0历史','D3 game_5 val',450,pct(old_event['val']['bounce']['f1']),pct(old_event['val']['net']['f1']),pct(old_event['val']['macro_f1']),'历史独立val选模'],
            ['A0历史','D3 test_1–7 test',464,pct(old_event['test']['bounce']['f1']),pct(old_event['test']['net']['f1']),pct(old_event['test']['macro_f1']),'历史独立测试'],
            *[[k,'D3 game_5 val',450,pct(best[k]['val']['bounce']['f1']),pct(best[k]['val']['net']['f1']),pct(best[k]['val']['macro_f1']),'当前最佳PP模型的缓存验证快照'] for k in 'ABCD']]), '',
        '当前A/B/C/D未完成D3 test_1–7的独立测试；不要把A0历史test成绩归给新模型。C的TTGames分类较好，但PP连续触台漏检106/203；时间诊断放宽到±5帧仍只命中97个，并非大量预测仅偏一两帧。', '',
        'Transformer原因尚未证实：当前TCN分别投影视觉256→96与运动9→32；Transformer直接拼接265维投影至128再加时间编码，因此本轮比较的是两套完整事件头，不能将差距只归因于attention。'+link('Transformer诊断','out/results/20261016_transformer_diagnosis/summary.md')+'。', '',
        '## 7. 历史PP public落点评估（单独列示）', '',
        '数据为D4：15完整视频、三个角度各5个，92有效落点、12忽略标签。训练来源、事件规则、抽帧和忽略匹配实现存在差异，不能与第5节PP内部验证成绩直接排名。', '',
        table(['方案','训练数据/阶段','0° F1','45° F1','90° F1','角度宏F1','TP / FP / FN'],[
            ['YOLO11n 1280＋原启发式','D5全量154视频；第32/80轮暂停',*[pct(yolo_public['per_angle'][v]['f1']) for v in ('0度','45度','90度')],pct(yolo_public['macro_f1']),'43 / 81 / 49'],
            ['历史冻结TCN A0','D3事件训练＋PP定位预训练；best12',*[pct(old_event['public']['by_angle'][v]['f1']) for v in ('0度','45度','90度')],pct(old_event['public']['macro_f1']),'68 / 157 / 24'],
            ['历史联合微调TCN','D3；仅第2轮中间评估，后暂停',*[pct(joint['by_angle'][v]['f1']) for v in ('0度','45度','90度')],pct(joint['macro_f1']),' / '.join(str(jt[a]) for a in ('tp','fp','fn'))]]), '',
        'YOLO记录采用“先忽略再匹配”本地近似评分；历史TCN采用先匹配有效GT、再由忽略GT吸收剩余预测。联合模型仅中间结果，不能据15.47%断言联合训练最终无效。YOLO全量训练使用了D1内部val对应视频，因此其public结果不代表D1固定留出协议。', '',
        '历史A0 public 43.20%与当前A0内部val 50.86%是**同一历史权重在两个不同数据集/阈值协议下的成绩**；不能把差值当成模型训练收益。当前新四组尚未运行public迁移评估。', '',
        link('YOLO public原始记录','experiments/records/20261013-215749_yolo11n_full1280_public/evaluations/epoch032_public_paused.json')+'；'+link('历史冻结TCN完整报告','ttnet/output/events/20261014-181231_event25_frozen/summary.md')+'；'+link('联合版第2轮public指标','ttnet/runs/20261014-181231_event25_joint/evaluations/epoch002_public/public/metrics.json')+'。', '',
        '## 8. 速度与batch测试：按范围区分', '',
        '均为本机RTX 5070 Ti；不同精度、后端、视频集合和是否包含事件头不可直接当作同一基准。', '',
        table(['模型/测试','数据范围','后端/精度','完整视频FPS','包含内容'],[
            ['YOLO640，conf .25','D1六段代表视频','PyTorch FP32','316.73','解码到写盘；几乎无检测输出'],
            ['官方三帧ONNX','D1六段代表视频','ORT CUDA FP32/TF32','111.23','解码、定位、写盘'],
            *[[label,'D1固定val，按各自benchmark记录','PyTorch AMP',number(ball_bench[name]['e2e_fps']),'纯球定位，不含事件头'] for name,label in zip(ball_runs,names)],
            *[[k,'D2全部40原视频/6292帧','PyTorch AMP',number(current[k]['e2e_fps']),'解码＋定位/轨迹＋事件头＋JSON序列化'] for k in ('A0','A','B','C','D')],
            ['历史YOLO1280＋启发式','D4全部15视频','ORT CUDA FP32','66.20','历史完整落点管线'],
            ['历史A0 public','D4全部15视频','PyTorch AMP',number(old_event['public']['e2e_fps']),'历史完整落点管线'],
            ['历史joint第2轮','D4全部15视频','PyTorch AMP',number(joint['e2e_fps']),'与继续训练共享GPU，不作独立速度排名']]), '',
        '本轮A/B/C/D全部超过50FPS；B相对A约降低13.92%。PyTorch分配器记录的完整推理峰值约48～49MiB，不是整张显卡总占用，也不包含桌面/驱动/其他进程。轨迹额外未来等待0帧；原三帧定位需1帧未来，中心13帧事件窗口需6帧，组合理论等待7帧；执行器仍是离线整视频。', '',
        '特征提取batch实测：D1 val `22_001`前128帧，CUDA AMP，只计特征提取组件：', '',
        table(['batch','组件FPS','PyTorch峰值MiB','相比batch1坐标变化>1px'],[[r['batch'],number(r['fps']),number(r['peak_cuda_bytes']/2**20,1),r['geometry_changes_over_1px']] for r in batch['feature']]), '',
        '采用feature_batch8、micro_batch8、validation_batch128；有效batch始终为8、学习率不变。事件头同一有效batch8的前后向耗时micro1/2/4/8分别为12.21/5.26/2.68/1.55ms，不含optimizer.step。AMP描述符最大差异约1.03e-4；128帧中超过1px的坐标变化为0。', '',
        '早期两视频762帧的初始化事件头运行检查A/B/C/D为98.31/89.99/90.63/86.46FPS，只是运行检查；第5节当前已训练模型的40视频完整评估为正式本轮速度记录。'+link('batch原始记录','out/results/20261016_training_preflight/batch_benchmark.json')+'。', '',
        '## 9. 工程验证、数据合并与中断记录', '',
        table(['验证项','数据/对象','已有结果','证据'],[
            ['自动测试','模型/缓存/连续评分/轨迹边界等','最近完整记录86项：85通过、1项旧图片测试跳过',link('测试日志','out/experiments/20261016-154803_trajectory_priority_seed42/tests.log')],
            ['四组真实样本检查','PP val 02_001真实帧，CPU与CUDA','前向、梯度、恢复、mask、因果状态与A历史兼容通过；不是精度成绩',link('CUDA真实样本','out/results/20261015_comparison_code_checks/real_samples_cuda.json')],
            ['候选诊断','D1 val全40视频','旧CPU诊断808错误帧均有完整叠加视频',link('候选视频索引','out/results/20261015_fixed3_candidates_full/index.html')],
            ['A/B诊断视频核验','D1 val每方案40视频','80视频帧数/末帧解码/错误引用通过；A809、B658个10px错误',link('核验记录',TRAJECTORY/'overlay_verification.json')],
            ['新事件标注合并','D1＋event_annotations全部154视频','固定114/40划分；原粗标记层和确认后的监督层分开',link('合并记录','experiments/records/20261016-150030_pp_event_merge/summary.md')],
            ['重复数据清理/缓存读取','PP与TTGames原视频、共享NPY','清理144788文件，释放约49.6GiB；122项历史保护产物核对通过',link('数据清理报告','docs/data_cleanup.md')]]), '',
        '中断/未完成实验不作为成功模型：', '',
        table(['实验','实际状态','本次处理'],[
            ['官方ONNX第一次启动 20261013-212251','失败；随后212318成功重跑','不重复统计精度'],
            ['单帧Local 20261014-001042','完成2轮后用户暂停','没有完整可比较验证成绩，留空'],
            ['YOLO1280 20261013-215749','计划80轮，完成32轮后暂停','保留第32轮public中间评估'],
            ['联合TCN 20261014-181231_joint','完成2轮，第3轮step400暂停','只列第2轮public中间评估'],
            ['20261016-152348 的A','在特征准备阶段被batch优化替代','没有事件训练精度，不计额外A结果'],
            ['20261016-153213 的A/B恢复','缓存准备/进程恢复，保留attempt和检查点','本轮最终结果按实际best checkpoint计一次'],
            ['多次smoke与pipeline封装','小规模检查或调度器，不是独立模型对照','检查通过列入工程验证，不冒充新精度实验']]), '',
        '完整可发现运行目录的状态见 '+link('运行清单',RECORD/'inventory.json')+'。', '',
        '## 10. 当前结论与未完成项', '',
        '1. 自训练三帧定位明显好于自训练单帧Global，但仍未超过官方ONNX在相同PP视频上的定位验证成绩；官方权重来源、后端和边界处理差异需要保留。',
        '2. 本轮新增PP事件监督后，A相对A0的内部落点宏F1由50.86%到72.44%；它体现新增监督＋重新训练的综合收益。',
        '3. 在本轮固定协议、seed42下，轨迹关联在两种事件头上均带来配对提升：A→B、C→D均提高。B的77.96%/86.54FPS是当前验证候选最佳，尚未证明多种子稳定性。',
        '4. 当前Transformer两组不如对应TCN，主要问题包含连续触台漏检；输入融合和时间编码也与TCN不同，原因待受控消融，不能推广成Transformer普遍不适合。',
        '5. 尚未完成：A/B的seed43、44；当前新四组在OpenTTGames test_1–7、PP public上的最终泛化/迁移；最终逐视角、多种子选型。当前保留A为默认，B为待复验候选，不宣布部署替换。', '',
        '本轮边界仍是冻结定位器、13帧事件窗口；新骨干、联合微调、桌面几何、时间偏移回归及部署集成为后续阶段。', '',
        '## 11. 本轮模型与结果入口', '']
    entrances=[]
    for k in 'ABCD':
        run=STUDY.with_name(STUDY.name+'_'+k)
        entrances.append([k,link('配置',run/'config.yaml'),link('best权重',run/'checkpoints/best.pt'),link('完整指标',run/'evaluations/continuous/metrics.json'),link('错误视频',run/'evaluations/continuous/index.html'),current[k]['checkpoint_sha256'][:16]])
    text += [table(['方案','配置','权重','指标','可视化','checkpoint SHA256前16位'],entrances), '',
        'A0当前内部评估：'+link('指标',STUDY/'A0_continuous/metrics.json')+'；'+link('错误视频',STUDY/'A0_continuous/index.html')+'。完整权重SHA256、数据和来源均保存于CSV及JSON，避免仅凭实验名称认模型。', '',
        '完整身份：', '',
        table(['对象','身份'],[['PP球定位v2 dataset_id',pp['dataset_id']],['历史PP球定位v1 dataset_id',pp_old['dataset_id']],['PP train清单SHA256',pp['split_hashes']['train']],['PP val清单SHA256',pp['split_hashes']['val']],['TTGames dataset_id',ott['dataset_id']],['TTGames split_id',split['split_id']],['确认版PP事件annotation_id',current['A']['annotation_id']],['本轮合并训练dataset_id',current['A']['dataset_id']]]), '',
        '本报告生成器：'+link('build_experiment_comparison.py',Path(__file__))+'。来源清单与内容哈希保存于 '+link('sources.json',RECORD/'sources.json')+'。']

    # Inventory includes failed/paused/replaced runs but never invents scores.
    inventory=[];seen=set()
    for root in (PROJECT/'ttnet/runs', PROJECT/'out/experiments'):
        for run in sorted(root.glob('*')):
            if not run.is_dir() or run.resolve() in seen:continue
            seen.add(run.resolve());status=run/'run.json'
            if not status.exists():continue
            state=read(status)
            has_evaluation=(run/'evaluations/continuous/metrics.json').exists()
            inventory.append(dict(experiment_id=run.name,run=str(run),recorded_status=state.get('status'),
                derived_status='evaluated' if has_evaluation else state.get('status'),epochs_completed=state.get('epochs_completed'),
                best_epoch=state.get('best_epoch'),continuous_metrics=str(run/'evaluations/continuous/metrics.json') if has_evaluation else None))
    for row in legacy:
        if row['experiment_id'] not in {r['experiment_id'] for r in inventory}:
            inventory.append(dict(experiment_id=row['experiment_id'],recorded_status=row['status'],source='experiments/comparison.csv'))
    dump(inventory,RECORD/'inventory.json')
    dump(dict(generated_at=timestamp,hash_algorithm='SHA256',sources=SOURCES),RECORD/'sources.json')
    # Deferred links exist now; validate all local links before publishing.
    for p in LINKS:
        if not Path(p).exists():raise FileNotFoundError(p)
    keys=[(r['experiment_id'],r['eval_id']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate experiment/evaluation key')
    for row in rows:
        if not path(row['source']).is_file():raise FileNotFoundError(row['source'])
        for key in ('precision','recall','f1','macro_f1','bounce_f1','net_f1'):
            if row[key] != '' and not 0 <= float(row[key]) <= 1:raise ValueError((key,row[key]))
    for part in ('train','val'):
        actual=hashlib.sha256((PROJECT/f'data/datasets/pp_ball_v2/splits/{part}.jsonl').read_bytes()).hexdigest()
        if actual != pp['split_hashes'][part] or actual != pp_old['split_hashes'][part]:raise ValueError('split hash mismatch')
    if len({r['annotation_id'] for r in current.values()})!=1:raise ValueError('incomparable annotation versions')
    for result in current.values():
        metrics=result['metrics'];landing=metrics['landing'];temporal=metrics['continuous_bounce']
        if landing['tp']+landing['fn'] != event_counts['val']['spatial']:raise ValueError('spatial count mismatch')
        if temporal['tp']+temporal['fn'] != event_counts['val']['bounces']:raise ValueError('temporal count mismatch')
        mean=sum(v['f1'] for v in metrics['by_view'].values())/len(metrics['by_view'])
        if abs(mean-metrics['macro_f1'])>1e-12:raise ValueError('macro F1 mismatch')
    for k in 'ABCD':
        checkpoint=STUDY.with_name(STUDY.name+'_'+k)/'checkpoints/best.pt'
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=current[k]['checkpoint_sha256']:
            raise ValueError('checkpoint hash mismatch '+k)
    document='\n'.join(text)+'\n'
    DOCUMENT.write_text(document)
    (RECORD/'comparison.md').write_text(document)
    dump(dict(status='completed_documentation_snapshot',generated_at=timestamp,main_document=str(DOCUMENT),
        comparison_rows=len(rows),dataset_registry=str(RECORD/'datasets.json'),sources=len(SOURCES),
        verified_local_links=len(LINKS),checkpoint_hash_checks='A/B/C/D passed',
        incomplete_experiments=['A/B seeds43/44','current A/B/C/D OpenTTGames independent test','current A/B/C/D PP public final transfer']),RECORD/'verification.json')
    print(json.dumps(dict(document=str(DOCUMENT),rows=len(rows),sources=len(SOURCES),links=len(LINKS)),ensure_ascii=False))


if __name__=='__main__':main()
