"""Attach post-hoc diagnostic results without replacing the prespecified primary row."""
import csv
import io
import json
import shutil
import sys
import numpy as np
from run_yolo_ball import ROOT, save, match, sha


def main():
    exp=sys.argv[1];record=ROOT/'experiments/records'/exp;out=ROOT/'out/experiments'/exp
    main=json.loads((record/'evaluations/best_fp32_640.json').read_text())
    diag=json.loads((record/'evaluations/score_diagnostic.json').read_text())
    audit=json.loads((record/'evaluations/ap_audit.json').read_text())
    errors=[]
    for line in (out/'eval/score_diagnostic/predictions.jsonl').read_text().splitlines():
        r=json.loads(line)
        errors.extend(t[2] for t in match(np.array(r['boxes']),np.array(r['predicted_boxes']).reshape(-1,4),10))
    diag['center_error_median_px']=float(np.median(errors)) if errors else None
    diag['center_error_p90_px']=float(np.percentile(errors,90)) if errors else None
    diag['ap_audit']=audit
    diag['checkpoint']=main['checkpoint'];diag['checkpoint_sha256']=main['checkpoint_sha256']
    save(record/'evaluations/score_diagnostic.json',diag)
    csvpath=ROOT/'experiments/comparison.csv';rows=list(csv.DictReader(csvpath.open()))
    primary=next(r for r in rows if r['experiment_id']==exp and r['eval_id']=='best_fp32_640')
    primary['notes']+='; library AP affected by linear interpolation beyond achieved recall; consult AP audit; benchmark at conf .25 with almost no detections'
    d=dict(primary);d.update(eval_id='score001_ap101_diagnostic',protocol='ball_v1_posthoc_score001_ap101',shape='1x3x384x640',
        ap50=audit['ap50'],ap50_95=audit['ap50_95'],ap_precision='',ap_recall='',center_p=diag['per_confidence']['0.001']['precision'],
        center_r=diag['per_confidence']['0.001']['recall'],center_f1=diag['per_confidence']['0.001']['f1'],
        center_median_px=diag['center_error_median_px'],center_p90_px=diag['center_error_p90_px'],model_p50_ms='',model_p95_ms='',e2e_fps='',peak_memory_mib='',
        params_m='',gmac='',gflops='',compute_tool='',compute_scope='',evaluation=str((record/'evaluations/score_diagnostic.json').relative_to(ROOT)),
        predictions=str((out/'eval/score_diagnostic/predictions.jsonl').relative_to(ROOT)),log=str((out/'attempts/001/diagnostic.log').relative_to(ROOT)),
        notes='post-hoc diagnostic, not selected operating threshold; AP101 zero beyond achieved recall, not full COCO; no timing for this conf; all source ratios letterbox to 384x640')
    rows=[r for r in rows if not(r['experiment_id']==exp and r['eval_id']==d['eval_id'])]+[d]
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(primary));w.writeheader();w.writerows(rows);csvpath.write_text(buf.getvalue())
    keys=['experiment_id','eval_id','status','ap50','ap50_95','center_f1','model_p50_ms','e2e_fps','params_m','gflops']
    (ROOT/'experiments/comparison.md').write_text('# 球定位实验\n\nAP / P / R / F1 均为 0–1。主行保留 Ultralytics AP；诊断行采用不同 AP 积分协议，**不能直接横比**。框架 AP 的尾部插值在本次低召回情况下明显偏高，详见实验 summary。\n\n| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(str(r[k]) or '—' for k in keys)+' |' for r in rows)+'\n\n触桌指标：本轮不执行。\n')
    b=main['benchmark'];c=diag['per_confidence']['0.001']
    (record/'summary.md').write_text(f'''# YOLO11n 球定位首轮测试\n\n**结果：本轮权重尚不能用作可靠定位模型。只测试球定位，没有运行触桌判断。**\n\n## 配置与完成情况\n\nYOLO11n COCO 预训练 → ball 单类别，输入 640，FP32，AdamW，seed=42。3,162 张训练帧（原视频每 5 帧采样），5,302 张验证帧。先完成 64/32 样本的一轮 smoke，再执行上限 15 epoch 的正式实验。因连续 5 轮 mAP50–95 没有提升，第 6 轮 early stopping；best 来自第 1 轮。流程已完成，但这不表示模型充分收敛。\n\n## 实际定位结果\n\n| 指标 | 主评估 conf=.25 | 事后诊断 conf=.001 |\n|---|---:|---:|\n| 10px 球心 TP | 0 | 63 |\n| FP | 0 | 6 |\n| FN | 5,302 | 5,239 |\n| Precision | 0（分母为零时的约定） | {c['precision']:.6f} |\n| Recall | 0 | {c['recall']:.6f} |\n| F1 | 0 | {c['f1']:.6f} |\n| 匹配误差中位数 / P90（原图 px） | — | {diag['center_error_median_px']:.3f} / {diag['center_error_p90_px']:.3f} |\n\nconf=.001 仍有 5,238 帧完全无输出，总计仅 69 个框；最高置信度为 {diag['top_score_percentiles']['max']:.6f}。降低阈值无法消除严重漏检。误差仅对 63 个匹配框统计，覆盖率只有 1.19%，不能凭小误差认为总体定位良好。conf=.25 下有标注连续区间最长漏检 144 帧。\n\n## AP 核对\n\n框架报告 AP50={main['bbox_ap']['metrics/mAP50(B)']:.6f}，AP50–95={main['bbox_ap']['metrics/mAP50-95(B)']:.6f}，已保留原值。检查实际安装的 Ultralytics 8.3.225 `compute_ap` 后发现，它将 PR 曲线补到 (recall=1, precision=0) 并线性插值积分。在本轮极低召回下，未覆盖的召回区域也贡献了面积，不能把框架 AP50=20.85% 理解成能定位约两成球。\n\n对事后单帧预测单独采用 101 个召回点、超过实际最大召回时精度置零的本地 AP 协议，AP50={audit['ap50']:.6f}，AP50–95={audit['ap50_95']:.6f}。这是另一评估记录，不是完整 pycocotools COCO 评分；单帧预测与 batch 验证也有数值差异，不能把两者的全部差值都归因于积分方式。同一份单帧预测使用库的线性插值可得到 AP50={audit['ultralytics_linear_endpoint_ap_by_iou'][0]:.6f}，支持上述尾部积分问题。\n\n## 速度与计算量\n\n- RTX 5070 Ti / PyTorch CUDA FP32，batch=1，固定 1×3×640×640，前向 p50={b['model_p50_ms']:.3f} ms、p95={b['model_p95_ms']:.3f} ms。warmup 50，3×200 次 CUDA events。\n- 六段完整验证视频，CPU 解码至 JSONL 写盘 E2E={b['e2e_fps']:.2f} FPS，实际预测为 stride 对齐的矩形输入。conf=.25 基本无检测框，后处理负担很轻，该速度不代表准确检测时的性能。\n- 未融合模型 {b['params_m']:.6f} M 参数，THOP {b['gmac']:.6f} GMAC / {b['gflops']:.6f} GFLOPs；不保证覆盖 functional attention 等操作。前向模型经过融合。\n\n## 限制与下一步\n\n只使用有球标注帧；2,844 个无球标注帧语义未知，未作为负样本，无法报告确定无球帧误报率。按场景保守分组，但缺少原始比赛来源元数据，不能证明完全独立。val 已用于 best checkpoint 选择，没有独立测试集。没有相同协议的热图对照，不能声称 YOLO 普遍无效或优于其他模型。\n\n下一轮应先修正 best / early-stop 选择指标对低召回的敏感性，验证训练能稳定学习后再测试更高输入分辨率。此次没有继续启动新训练或自动调整部署阈值。未导出 ONNX/TensorRT；本机结果不代表赛事 4090。官方 pingpang 完整性校验通过。\n\n## 产物\n\n- best / last：`{out.relative_to(ROOT)}/attempts/001/checkpoints/`\n- 原始日志、每 epoch CSV、显存记录：`{out.relative_to(ROOT)}/attempts/001/`\n- 逐帧预测、逐视频指标、绿色 GT / 红色预测错误图：`{out.relative_to(ROOT)}/eval/`\n- 测速原始样本：`{out.relative_to(ROOT)}/benchmark/`\n- 完整评估 JSON：本目录 `evaluations/`；best/last 已被框架 strip 优化器，不能保证精确续训。\n''')
    state=json.loads((record/'run.json').read_text());state['posthoc_diagnostics']=['score_diagnostic.json','ap_audit.json'];state['source_sha256']={p.name:sha(p) for p in (ROOT/'scripts').glob('*yolo*.py')}
    save(record/'run.json',state)
    for p in (ROOT/'scripts').glob('*yolo*.py'):
        target=out/'attempts/001/posthoc_sources'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    keys=[(r['experiment_id'],r['eval_id']) for r in rows];assert len(keys)==len(set(keys))
    for r in rows:
        for k in ['config','log','evaluation','predictions','checkpoint']:
            if r[k]:assert (ROOT/r[k]).exists(),(k,r[k])
    print('Final records and references verified')

if __name__=='__main__':main()
