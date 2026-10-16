"""Re-identify a completed prediction cache after verified initialization-only edits."""
import argparse,json,shutil
from pathlib import Path
import torch
from ttnet.common import EVENT_DATA,save
from ttnet.events.data import EventCorpus
from ttnet.events.continuous import attach_pp
from ttnet.events.engine import EventSystem,cache_identity,feature_cache_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    proof=json.loads((a.root/'cache_parity.json').read_text())
    if proof['status']!='passed':raise ValueError('parity not established')
    old=Path(json.loads((a.root/'table_cache.json').read_text())['cache'])
    cfg=json.loads((a.root/'configs/table_relative.json').read_text());corpus=EventCorpus(EVENT_DATA);attach_pp(corpus,cfg['pp_annotations']);system=EventSystem(cfg,torch.device('cpu'))
    expected=cache_identity(system,corpus);previous=json.loads((old/'identity.json').read_text())
    differences=[k for k in expected if previous.get(k)!=expected[k]]
    # Only the RNG isolation and oracle video-id alias were changed in engine.py.
    # Frozen numerical inference was checked against the original cache on three views.
    if set(differences)-{'code_sha256'}:raise ValueError(f'unreviewed cache changes {differences}')
    new=feature_cache_path(system,corpus);new.mkdir(parents=True,exist_ok=True)
    for part in ('train','val'):
        for vid in corpus.split[part]:
            for suffix in ('.npz','.jsonl'):
                src=old/(vid+suffix)
                if not src.exists():raise ValueError(f'incomplete source {src}')
                if old!=new:shutil.copyfile(src,new/src.name)
    save(new/'identity.json',expected);save(new/'derivation.json',dict(source=str(old),previous_identity=previous,changed_keys=differences,parity=proof,note='Engine-only changes: auxiliary initialization RNG isolation, pp_ alias for oracle. Prediction geometry unchanged.'))
    save(a.root/'table_cache.json',dict(cache=str(new)));print(new)

if __name__=='__main__':main()
