"""Training curves and epoch records for result reports."""
import json
import os
from ttnet.common import ROOT, identity, save
from ttnet.publication import read

def epoch_records(run):
    records = {}
    for path in sorted((run/'logs').glob('attempt_*/epochs.jsonl')):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a running process may be appending the last line
            records[record['epoch']] = record
    return [records[epoch] for epoch in sorted(records)]


def training_curve(run, records):
    if not records:
        return
    cfg = read(run/'config.yaml')
    curve_identity = identity(dict(records=records, config=cfg))
    if read(run/'evaluations/training_curve_identity.json').get('identity') == curve_identity and (run/'evaluations/training_curve.svg').is_file():
        return
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/cache/matplotlib'))
    try:
        import matplotlib
    except ModuleNotFoundError:
        return  # result reporting must not interrupt training if plotting is unavailable
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
    axes[0].plot(epochs, [r['train_loss'] for r in records], marker='o', markersize=3)
    axes[0].set(title='Mean training loss', xlabel='Epoch', ylabel='Heatmap MSE')
    axes[1].plot(epochs, [r['val']['f1']*100 for r in records], marker='o', markersize=3, color='#d65d0e')
    tolerance = cfg['global_best_tolerance_px'] if cfg['stage']=='global' else cfg['local_best_tolerance_px']
    axes[1].set(title=f'Validation F1 @ {tolerance}px', xlabel='Epoch', ylabel='F1 (%)')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.set_xticks(epochs[::max(1,len(epochs)//10)])
    if cfg['stage']=='local' and epochs[-1]>cfg['gt_crop_epochs']:
        for axis in axes:
            axis.axvline(cfg['gt_crop_epochs']+.5,color='gray',linestyle='--',label='Predicted ROI phase')
        axes[1].legend()
    dest=run/'evaluations/training_curve.svg'
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(f'training_curve.{os.getpid()}.tmp.svg')
    fig.savefig(temp)
    preview=dest.with_suffix('.png')
    preview_temp=preview.with_name(f'training_curve.{os.getpid()}.tmp.png')
    fig.savefig(preview_temp,dpi=120)
    preview_temp.replace(preview)
    plt.close(fig)
    temp.replace(dest)
    save(run/'evaluations/training_curve_identity.json', dict(identity=curve_identity))


