#!/usr/bin/env python3
from __future__ import annotations
import json, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
M=ROOT/'results/measurements'
EXPECTED={
 'primary': dict(base=310342.0131826742,candidate=242750.24105461393,paired=0.2165502300220319,favor=8,noise=0.04236515633029331,threshold=0.08473031266058662),
 'world1': dict(base=435846.3704379562,candidate=375199.3600973236,paired=0.12944674473390172,favor=8,noise=0.058469144896320435,threshold=0.11693828979264087),
 'two-rank': dict(base=5043063.320652174,candidate=5067636.5869565215,paired=0.003848528575462456,favor=4,noise=0.04409642221849111,threshold=0.08819284443698222),
}

def load(path): return json.loads(path.read_text())
def close(a,b,tol=1e-12): return abs(a-b) <= tol*max(1.0,abs(a),abs(b))

def reduce_path(name):
    d=M/name
    by={}
    for p in sorted(d.glob('main-*.json')):
        if '-rank' in p.name: continue
        o=load(p); by.setdefault(o['block'],{})[o['arm']]=o['block_median_ns']
    assert sorted(by)==list(range(8))
    base=[by[b]['base'] for b in range(8)]
    cand=[by[b]['candidate'] for b in range(8)]
    improvements=[(by[b]['base']-by[b]['candidate'])/by[b]['base'] for b in range(8)]
    pair_abs=[]; phase_values={}
    for phase in ('pre','post'):
        pairs={}; vals=[]
        for p in sorted(d.glob(f'null-{phase}-*.json')):
            if '-rank' in p.name: continue
            o=load(p); vals.append(o['block_median_ns']); pairs.setdefault(o['block'],{})[o['logical_arm']]=o['block_median_ns']
        phase_values[phase]=vals
        for block in sorted(pairs):
            x=pairs[block]; pair_abs.append(abs(x['base_a']-x['base_b'])/x['base_a'])
    pre=statistics.median(phase_values['pre']); post=statistics.median(phase_values['post'])
    pair_noise=statistics.median(pair_abs)
    drift=abs(post-pre)/pre
    noise=max(pair_noise,drift)
    threshold=max(0.01,2*noise)
    out=dict(
      base_median_ns=statistics.median(base),
      candidate_median_ns=statistics.median(cand),
      paired_median_improvement=statistics.median(improvements),
      favor_candidate=sum(x>0 for x in improvements),
      total_pairs=8,
      median_pair_abs_relative=pair_noise,
      pre_post_drift=drift,
      noise_floor=noise,
      decision_threshold=threshold,
      positive=(sum(x>0 for x in improvements)>=6 and statistics.median(improvements)>threshold),
    )
    exp=EXPECTED[name]
    assert close(out['base_median_ns'],exp['base'])
    assert close(out['candidate_median_ns'],exp['candidate'])
    assert close(out['paired_median_improvement'],exp['paired'])
    assert out['favor_candidate']==exp['favor']
    assert close(out['noise_floor'],exp['noise'])
    assert close(out['decision_threshold'],exp['threshold'])
    return out

result={name:reduce_path(name) for name in ('primary','world1','two-rank')}
print(json.dumps(result,indent=2,sort_keys=True))
