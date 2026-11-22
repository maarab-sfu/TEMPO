"""M5.3: select temporally diverse object evidence from M5.2 trajectory.

Uses trajectory metadata only; does not hallucinate visual semantics.
"""
import argparse
import json
import math
from pathlib import Path


def _features(frame):
    x1,y1,x2,y2 = frame['box']
    return ((x1+x2)/2, (y1+y2)/2, math.sqrt(max(0,(x2-x1)*(y2-y1))))


def select_evidence(data, budget=8, min_gap=0.35):
    if budget < 1 or min_gap < 0:
        raise ValueError('budget must be >= 1 and min_gap >= 0')
    frames = sorted(data['frames'], key=lambda f:f['timestamp'])
    valid = [f for f in frames if f.get('box') is not None and f.get('mask_pixels',0)>0]
    if not valid:
        return {'schema_version':1,'status':'no_visible_object','object_phrase':data.get('object_phrase'), 'evidence':[], 'missing_intervals':_missing_intervals(frames), 'selection_method':'trajectory_diversity_v1'}
    # Prefer the longest uninterrupted visible run to avoid isolated tracking artifacts.
    runs=[]; run=[]
    for f in frames:
        if f.get('box') is not None and f.get('mask_pixels',0)>0:
            run.append(f)
        elif run:
            runs.append(run);run=[]
    if run:runs.append(run)
    main=max(runs,key=lambda r:(len(r),sum(f['mask_pixels'] for f in r)))
    # First/last give temporal coverage; greedy farthest-point samples add changes in location/scale.
    selected=[main[0]]
    if len(main)>1 and budget>1: selected.append(main[-1])
    xs=[_features(f) for f in main]
    # Normalize geometric distances by observed ranges; time by segment duration.
    ranges=[max(v[k] for v in xs)-min(v[k] for v in xs) for k in range(3)]
    duration=max(main[-1]['timestamp']-main[0]['timestamp'],1e-6)
    def dist(a,b):
        va,vb=_features(a),_features(b)
        return math.sqrt(sum(((va[k]-vb[k])/max(ranges[k],1))**2 for k in range(3)) + ((a['timestamp']-b['timestamp'])/duration)**2)
    while len(selected)<min(budget,len(main)):
        candidates=[f for f in main if f not in selected and all(abs(f['timestamp']-s['timestamp'])>=min_gap for s in selected)]
        if not candidates:break
        best=max(candidates,key=lambda f:(min(dist(f,s) for s in selected),f['mask_pixels']))
        selected.append(best)
    selected.sort(key=lambda f:f['timestamp'])
    return {'schema_version':1,'status':'selected','object_phrase':data.get('object_phrase'),
            'source_video':data.get('video'),'seed_timestamp':data.get('seed',{}).get('timestamp'),
            'selection_method':'trajectory_diversity_v1','requested_budget':budget,
            'visible_frame_count':len(valid),'primary_run_frame_count':len(main),
            'primary_run_interval':[main[0]['timestamp'],main[-1]['timestamp']],
            'evidence':[{'timestamp':f['timestamp'],'frame_index':f['frame_index'],'box':f['box'],'mask_pixels':f['mask_pixels'],'role':'object_visible'} for f in selected],
            'missing_intervals':_missing_intervals(frames),
            'limitations':['Object presence and geometry only; no action recognition','Selection is not ground-truth evidence quality','Missing masks do not establish object exit']}


def _missing_intervals(frames):
    intervals=[]; start=None;end=None;count=0
    for f in frames:
        missing=f.get('box') is None or f.get('mask_pixels',0)==0
        if missing:
            if start is None:start=f['timestamp'];count=0
            end=f['timestamp'];count+=1
        elif start is not None:
            intervals.append({'start':start,'end':end,'sample_count':count});start=None
    if start is not None:intervals.append({'start':start,'end':end,'sample_count':count})
    return intervals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',required=True,type=Path)
    p.add_argument('--output',default='outputs/m53_object_evidence/evidence.json',type=Path)
    p.add_argument('--budget',type=int,default=8)
    p.add_argument('--min-gap',type=float,default=0.35)
    a=p.parse_args()
    result=select_evidence(json.loads(a.trajectory.read_text(encoding='utf-8')),a.budget,a.min_gap)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'evidence_count':len(result['evidence']),'output':str(a.output)},indent=2))

if __name__=='__main__':main()
