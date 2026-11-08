"""Exhaustive channel values, boundary/layout checks and CPU timing."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
import numpy as np
from participant.public_fast_normalize import normalize,reference_normalize


def main():
    rng=np.random.default_rng(43)
    all_values=np.repeat(np.arange(256,dtype=np.uint8).reshape(16,16,1),3,axis=2)
    cases=[all_values,np.zeros((256,256,3),np.uint8),np.full((256,256,3),255,np.uint8),
           rng.integers(0,256,(256,256,3),dtype=np.uint8),rng.integers(0,256,(288,512,3),dtype=np.uint8)]
    cases.extend([cases[3][::2,::2],cases[4][:,::-1],cases[3].astype(np.float32)])
    for x in cases:
        before=x.copy();expected=reference_normalize(x);actual=normalize(x)
        assert np.array_equal(actual,expected)
        assert actual.dtype==np.float32 and actual.flags.c_contiguous
        assert np.array_equal(x,before)
    result=dict(cases=len(cases),exact=True,benchmarks={})
    for size in [(256,256),(288,512)]:
        x=rng.integers(0,256,(*size,3),dtype=np.uint8)
        rows={}
        for name,fn in [('reference',reference_normalize),('lookup',normalize)]:
            for _ in range(30):fn(x)
            times=[]
            for _ in range(600):
                start=time.perf_counter();fn(x);times.append((time.perf_counter()-start)*1000)
            rows[name]=dict(p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),samples_ms=times)
        result['benchmarks'][str(size)]=rows
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(cases=result['cases'],exact=True,timings={k:{n:v['p50_ms'] for n,v in row.items()} for k,row in result['benchmarks'].items()}),indent=2))


if __name__=='__main__':main()
