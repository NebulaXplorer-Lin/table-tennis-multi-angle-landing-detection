"""Run ordered S1-S3 stages; optionally continue from an already-running Global run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from ttnet.common import DATA,PROJECT,ROOT,jsonl,now,save,sha


def new_run(name):
    return ROOT/'runs'/(datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')+'_'+name)


def command(module,*args):
    cmd=[sys.executable,'-m',module,*map(str,args)]
    print('Running '+' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=PROJECT,check=True)


def finalize(run):
    state=json.loads((run/'run.json').read_text())
    metric=json.loads((run/'evaluations/best_val/metrics.json').read_text())
    bench=json.loads((run/'evaluations/benchmark/metrics.json').read_text())
    exports=json.loads((run/'exports/manifest.json').read_text())
    for filename in ['checkpoints/best.pt','checkpoints/last.pt','predictions/best_val.jsonl',
                     'evaluations/best_val/errors.json','evaluations/best_val/index.html','feature_cache/best_val/identity.json']:
        if not (run/filename).exists():raise RuntimeError(f'missing artifact {filename}')
    for model in exports['models']:
        if not model['parity']['passed']:raise RuntimeError('export parity incomplete')
    source=run/'evaluations/best_val/errors.json'
    errors=json.loads(source.read_text())
    for error in errors:
        if not (run/error['visualization']).is_file():raise RuntimeError('missing error visualization')
    state.update(status='completed',completed=now(),validation=metric['center'],e2e_fps=bench['e2e_fps'],
                 export_parity='passed',target_tensorrt='not tested; no competition solution integration in stage 1')
    save(run/'run.json',state)
    config=json.loads((run/'config.yaml').read_text())
    dataset=Path(config['data'])
    dm=json.loads((dataset/'metadata/dataset.json').read_text())
    config.update(dataset_id=dm['dataset_id'],split_id='ttnet_ball_v1_frozen_yolo_pilot',split_hashes=dm['split_hashes'],
                  train_videos=dm['train_videos'],val_videos=dm['val_videos'],
                  train_frames=len(jsonl(dataset/'labels/train.jsonl')),val_frames=len(jsonl(dataset/'labels/val.jsonl')),
                  evaluated_best_checkpoint_sha256=state['best_sha256'],best_epoch=state['best_epoch'])
    save(run/'config.yaml',config)
    per=metric['center'];g=metric['global_center']
    summary=f'''# {state['stage']} 定位结果

状态：completed；数据集 {state['dataset_id']}；固定留出 40 视频、{metric['frames']} 帧。

| 项目 | 数值 |
|---|---:|
| best epoch | {state['best_epoch']} |
| Global 阈值 | {metric['global_threshold']} |
| Local 阈值 | {metric['local_threshold']} |
| 最终球心 F1，10 px | {per['10']['f1']:.6f} |
| 最终球心 F1，50 px | {per['50']['f1']:.6f} |
| 匹配误差中位数 / P90，10 px | {per['10']['error_median_px']} / {per['10']['error_p90_px']} |
| 最长连续漏检，10 px | {per['10']['max_consecutive_missed']} |
| ROI 覆盖率 | {metric['roi_coverage']} |
| 原始视频完整定位 FPS | {bench['e2e_fps']:.2f} |
| 峰值 GPU 显存 bytes | {bench['peak_cuda_bytes']} |
| 参数量，Global | {bench['global_compute']['parameters']} |
| Global GFLOPs（Conv2d） | {bench['global_compute']['flops']/1e9:.4f} |

性能为本机 PyTorch {bench['precision']}，含 PyAV 解码、预处理、定位和 JSON 序列化；不含磁盘写入与事件模型。GPU 可能与其他任务共享，不能代表赛事硬件结果。AP 不适用于球心热图。

历史基线 F1@10px：{metric['baseline']['reference_center_f1_10px']:.6f}；当前差值：{metric['baseline']['delta_f1_10px']:.6f}。视频划分相同，历史基线的解码与推理精度不同；这是历史对照，不是本次重新运行基线。单帧双分支与历史三帧定位器的输入也不同。

[逐视频指标](evaluations/best_val/per_video.json) · [完整指标](evaluations/best_val/metrics.json) · [错误索引](evaluations/best_val/errors.json) · [pred / label 预览](evaluations/best_val/index.html) · [测速](evaluations/benchmark/metrics.json) · [ONNX 对齐](exports/manifest.json)

未知标签不计入 P/R/F1，不把缺标判为背景。所有评分错误都有原图 pred / label 叠加图。未来事件接口已缓存验证；本阶段不识别触台。
'''
    (run/'summary.md').write_text(summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--global-run',type=Path);p.add_argument('--dataset',type=Path,default=DATA)
    p.add_argument('--wait-global',action='store_true');args=p.parse_args()
    pipeline=new_run('pipeline');pipeline.mkdir(parents=True)
    save(pipeline/'run.json',dict(status='running',started=now(),dataset=str(args.dataset)))
    global_run=args.global_run or new_run('global_ball')
    local_run=None
    try:
        if args.global_run and args.wait_global:
            while True:
                state=json.loads((global_run/'run.json').read_text()) if (global_run/'run.json').exists() else {}
                if state.get('status') in ['trained_pending_eval','evaluated_pending_export','completed']:break
                if state.get('status')=='failed':raise RuntimeError(f'Global failed: {state}')
                time.sleep(10)
        elif not args.global_run:
            smoke=new_run('global_smoke')
            command('ttnet.training.train','--stage','global','--smoke','--run',smoke,'--dataset',args.dataset)
            command('ttnet.training.train','--stage','global','--run',global_run,'--dataset',args.dataset)
        g=global_run/'checkpoints/best.pt'
        command('ttnet.evaluation.evaluate','--run',global_run,'--global-checkpoint',g,'--dataset',args.dataset,'--amp')
        command('ttnet.evaluation.export','--run',global_run,'--global-checkpoint',g,'--dataset',args.dataset)
        command('ttnet.evaluation.benchmark','--run',global_run,'--global-checkpoint',g,'--dataset',args.dataset,'--amp')
        finalize(global_run)
        smoke=new_run('local_smoke')
        command('ttnet.training.train','--stage','local','--smoke','--run',smoke,'--global-checkpoint',g,'--dataset',args.dataset)
        local_run=new_run('local_ball')
        command('ttnet.training.train','--stage','local','--run',local_run,'--global-checkpoint',g,'--dataset',args.dataset)
        l=local_run/'checkpoints/best.pt'
        command('ttnet.evaluation.evaluate','--run',local_run,'--global-checkpoint',g,'--local-checkpoint',l,'--dataset',args.dataset,'--amp')
        command('ttnet.evaluation.export','--run',local_run,'--global-checkpoint',g,'--local-checkpoint',l,'--dataset',args.dataset)
        command('ttnet.evaluation.benchmark','--run',local_run,'--global-checkpoint',g,'--local-checkpoint',l,'--dataset',args.dataset,'--amp')
        finalize(local_run)
        save(pipeline/'run.json',dict(status='completed',finished=now(),global_run=str(global_run.relative_to(ROOT)),local_run=str(local_run.relative_to(ROOT))))
        save(ROOT/'runs/latest.json',dict(global_run=str(global_run.relative_to(ROOT)),local_run=str(local_run.relative_to(ROOT)),pipeline_run=str(pipeline.relative_to(ROOT))))
        (pipeline/'summary.md').write_text(f'# 第一阶段完成\n\n[Global](../{global_run.name}/summary.md)\n\n[Global / Local](../{local_run.name}/summary.md)\n')
    except BaseException as e:
        save(pipeline/'run.json',dict(status='failed',finished=now(),error=repr(e),global_run=str(global_run),local_run=str(local_run) if local_run else None))
        raise


if __name__=='__main__':main()
