"""Prepare PP-only candidate training tensors from immutable rule observations."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from ttnet.common import DATA,save,save_jsonl,jsonl,sha
from ttnet.scene.table import TableHead,decode_table
from ttnet.data_tools.geometry import letterbox_transform
from ttnet.evaluation.ball_corpus import BallCorpus


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    source=Path('data/cache/features/35af6f8dbee933512d8c');dest=a.root/'ar_training_cache';dest.mkdir(exist_ok=True)
    state=torch.load(a.root/'table/best.pt',map_location='cpu',weights_only=False);model=TableHead().eval();model.load_state_dict(state['head'])
    save(dest/'identity.json',dict(base=json.loads((source/'identity.json').read_text()),table_sha256=sha(a.root/'table/best.pt'),source='fixed_rule_candidates',match_px=10))
    for part in ('train','val'):
        corpus=BallCorpus(DATA,part)
        for vid in corpus.split[part]:
            meta=corpus.videos[vid]['meta'];t=letterbox_transform(meta['width'],meta['height'])
            with np.load(a.root/f'table/features/{vid}.npz') as z:
                with torch.inference_mode():boxes=model(torch.from_numpy(z['features'])).numpy()
                mapping=dict(zip(z['frame_ids'].tolist(),boxes))
            records=jsonl(source/f'pp_{vid}.jsonl')
            for r in records:r['table']=decode_table(mapping[r['frame_id']],t,meta)
            shutil.copyfile(source/f'pp_{vid}.npz',dest/f'{vid}.npz');save_jsonl(dest/f'{vid}.jsonl',records)
    print(dest)

if __name__=='__main__':main()
