"""Split AR candidate accuracy into positive/negative frames on fixed val cache.

Runs the trained CandidateGRU in pure prediction mode (no teacher forcing) over
the cached validation candidate sets and reports where the ball is dropped,
wrongly picked, or false ball invented. CPU-only; evidence for the AR-vs-rule gap.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ttnet.common import save
from ttnet.tracking.train_ar import load_sequences
from ttnet.tracking.autoregressive import CandidateGRU, INPUT_DIM

NULL = 7  # 8 candidate slots: up to 7 real candidates + 1 null


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--ar', type=Path, required=True)
    p.add_argument('--no-table', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cpu')
    seqs = load_sequences(a.cache, 'val', device, not a.no_table)

    state = torch.load(a.ar, map_location='cpu', weights_only=False)
    model = CandidateGRU().to(device)
    model.load_state_dict(state['model'])
    model.eval()

    agg = dict(supervised=0, positive=0, negative=0,
               pos_correct=0, pos_null=0, pos_wrong=0,
               neg_correct=0, neg_false_ball=0)
    per_video = []

    with torch.no_grad():
        for s in seqs:
            vid = s['video']
            x, mask, target, nearest = s['x'], s['mask'], s['target'], s['nearest']
            h = torch.zeros(1, 128, device=device)
            v = dict(video=vid, supervised=0, positive=0, negative=0,
                     pos_correct=0, pos_null=0, pos_wrong=0,
                     neg_correct=0, neg_false_ball=0)
            n = len(x)
            for t in range(n):
                logp, emb = model.distribution(x[t:t+1], mask[t:t+1], h)
                chosen = int(logp.argmax(-1).item())
                h = model.advance(emb, h, torch.tensor([chosen], device=device))
                if not bool(target[t].any()):
                    continue  # Unknown labels mask metrics, never recurrent state updates.
                is_positive = bool(target[t, :-1].any())
                agg['supervised'] += 1
                v['supervised'] += 1
                if is_positive:
                    agg['positive'] += 1
                    v['positive'] += 1
                    if bool(target[t, chosen]):
                        agg['pos_correct'] += 1
                        v['pos_correct'] += 1
                    elif chosen == NULL:
                        agg['pos_null'] += 1
                        v['pos_null'] += 1
                    else:
                        agg['pos_wrong'] += 1
                        v['pos_wrong'] += 1
                else:
                    agg['negative'] += 1
                    v['negative'] += 1
                    if chosen == NULL:
                        agg['neg_correct'] += 1
                        v['neg_correct'] += 1
                    else:
                        agg['neg_false_ball'] += 1
                        v['neg_false_ball'] += 1
            per_video.append(v)

    def rates(d, num, den):
        return round(float(d) / max(1, den), 4)

    summary = dict(
        supervised=agg['supervised'],
        overall_accuracy=round((agg['pos_correct'] + agg['neg_correct']) / max(1, agg['supervised']), 4),
        positive=agg['positive'],
        pos_accuracy=rates(agg['pos_correct'], 0, agg['positive']),
        pos_drop_to_null=rates(agg['pos_null'], 0, agg['positive']),
        pos_wrong_candidate=rates(agg['pos_wrong'], 0, agg['positive']),
        negative=agg['negative'],
        neg_accuracy=rates(agg['neg_correct'], 0, agg['negative']),
        neg_false_ball=rates(agg['neg_false_ball'], 0, agg['negative']),
        null_precision=agg['neg_correct']/max(1,agg['neg_correct']+agg['pos_null']),
        null_recall=agg['neg_correct']/max(1,agg['negative']),
        null_f1=2*agg['neg_correct']/max(1,2*agg['neg_correct']+agg['pos_null']+agg['neg_false_ball']),
        per_video=per_video,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save(a.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_video'}, ensure_ascii=False, indent=1))
    worst = sorted(per_video, key=lambda r: r['pos_wrong'] / max(1, r['positive']), reverse=True)[:8]
    print('\nWorst positive-misselect videos (wrong candidate / positive):')
    for r in worst:
        print(f"  {r['video']}: pos={r['positive']} wrong={r['pos_wrong']} "
              f"drop_null={r['pos_null']} neg_false_ball={r['neg_false_ball']}")


if __name__ == '__main__':
    main()
