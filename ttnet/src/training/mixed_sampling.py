"""Epoch-addressable 50/50 sampling independent of model and micro-batch RNG."""
from collections import Counter
import numpy as np
from torch.utils.data import Sampler
from ttnet.common import identity


class MixedSourceSampler(Sampler):
    def __init__(self, rows, seed=42, effective_batch=16):
        self.rows,self.seed,self.effective_batch=rows,seed,effective_batch
        self.groups={source:[i for i,r in enumerate(rows) if r.get('source_domain','pp')==source]
                     for source in ('pp','opentt')}
        if not all(self.groups.values()) or effective_batch%2:
            raise ValueError('mixed sampling needs both sources and an even effective batch')
        self.epoch=0
        self.indices=[]
        self.set_epoch(0)

    def set_epoch(self,epoch):
        self.epoch=epoch
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,1701]))
        pp=np.array(self.groups['pp']);pp=pp[rng.permutation(len(pp))]
        count=len(pp);ot=self.groups['opentt'];selected=[]
        start=epoch*count
        while len(selected)<count:
            cycle,offset=divmod(start+len(selected),len(ot))
            order=np.random.default_rng(np.random.SeedSequence([self.seed,cycle,1702])).permutation(len(ot))
            take=min(count-len(selected),len(ot)-offset)
            selected.extend(ot[int(i)] for i in order[offset:offset+take])
        half=self.effective_batch//2
        self.indices=[]
        for start in range(0,count,half):
            block=pp[start:start+half].tolist()+selected[start:start+half]
            self.indices.extend(block[int(i)] for i in rng.permutation(len(block)))

    @property
    def manifest(self):
        rows=[self.rows[i] for i in self.indices]
        return dict(epoch=self.epoch+1,count=len(rows),sources=dict(Counter(r.get('source_domain','pp') for r in rows)),
                    sample_sha256=identity([(r['video_id'],r['frame_id']) for r in rows]))

    def __iter__(self):return iter(self.indices)
    def __len__(self):return len(self.indices)
