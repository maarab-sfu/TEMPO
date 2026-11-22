"""M4.13 paired DAVIS GT-box vs detector-box SAM2 error analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['sequence', 'frame', 'object_id']

def analyze(gt_csv, detected_csv, detections_json, out_dir):
    gt = pd.read_csv(gt_csv)
    det = pd.read_csv(detected_csv)
    for label, df in [('GT', gt), ('detected', det)]:
        if df.duplicated(KEYS).any():
            raise ValueError(f'{label} CSV has duplicate frame/object keys')
        if not {'J','F','JF',*KEYS}.issubset(df.columns):
            raise ValueError(f'{label} CSV lacks required columns')
    joined = gt.merge(det, on=KEYS, how='outer', suffixes=('_gt','_det'), indicator=True, validate='one_to_one')
    if (joined['_merge'] != 'both').any():
        raise ValueError('Frame/object keys differ across runs; paired comparison is invalid')
    if not np.isfinite(joined[['J_gt','F_gt','JF_gt','J_det','F_det','JF_det']].to_numpy()).all():
        raise ValueError('Non-finite metric values; check DAVIS evaluator handling')
    detections = json.loads(Path(detections_json).read_text(encoding='utf-8'))
    if isinstance(detections, dict):
        detections = detections.get('sequences', [])
    lookup = {}
    for seq in detections:
        name = seq['sequence']
        if name in lookup:
            raise ValueError(f'Duplicate detection sequence: {name}')
        lookup[name] = seq
    if set(joined.sequence.unique()) != set(lookup):
        raise ValueError('Detection metadata sequences do not match paired CSV sequences')
    rows=[]
    for (seq, obj), sub in joined.groupby(['sequence','object_id'], sort=True):
        sub=sub.sort_values('frame')
        meta=lookup[seq]
        missed=int(obj) in set(map(int,meta['missed_object_ids']))
        matched_id=str(int(obj)) in meta['matched_boxes']
        if missed == matched_id:
            raise ValueError(f'Inconsistent match metadata: {seq} object {obj}')
        n=len(sub); k=max(1,n//3)
        early=sub.iloc[:k]; late=sub.iloc[-k:]
        gtjf=float(sub.JF_gt.mean()); detjf=float(sub.JF_det.mean())
        delta=detjf-gtjf
        early_delta=float((early.JF_det-early.JF_gt).mean())
        late_delta=float((late.JF_det-late.JF_gt).mean())
        drift=late_delta-early_delta
        # Flags are diagnostic heuristics, NOT definitive failure causes.
        flags=[]
        if missed: flags.append('missed_initialization')
        else:
            if early_delta <= -0.15: flags.append('early_quality_gap')
            if drift <= -0.15: flags.append('worsening_relative_gap')
            if detjf < 0.5: flags.append('low_detected_JF')
        rows.append(dict(sequence=seq, object_id=int(obj), frames=n, initialized=not missed,
                         first_frame_box_iou=float(meta['first_frame_match_iou'].get(str(int(obj)),0)),
                         gt_J=float(sub.J_gt.mean()), gt_F=float(sub.F_gt.mean()), gt_JF=gtjf,
                         detected_J=float(sub.J_det.mean()), detected_F=float(sub.F_det.mean()), detected_JF=detjf,
                         delta_JF=delta, early_delta_JF=early_delta, late_delta_JF=late_delta,
                         relative_gap_change=drift, flags=';'.join(flags)))
    objects=pd.DataFrame(rows)
    if len(objects) != sum(int(s['gt_objects']) for s in detections):
        raise ValueError('Object count does not match detector metadata')
    # Macro-average objects, same aggregation used by both benchmarks.
    def metrics(df):
        return {m:float(df[m].mean()) for m in ['gt_J','gt_F','gt_JF','detected_J','detected_F','detected_JF','delta_JF']}
    matched=objects[objects.initialized]
    missed=objects[~objects.initialized]
    summary={
        'sequences':len(lookup),'objects':len(objects),'frame_object_pairs':len(joined),
        'matched_objects':len(matched),'missed_objects':len(missed),
        'initialization_recall':float(len(matched)/len(objects)),
        'all_objects':metrics(objects),
        'matched_objects_only':metrics(matched),
        'missed_objects_only':metrics(missed) if len(missed) else None,
        'macro_delta_contribution_matched':float(matched.delta_JF.sum()/len(objects)),
        'macro_delta_contribution_missed':float(missed.delta_JF.sum()/len(objects)),
        'flag_counts':{flag:int(objects['flags'].str.split(';').apply(lambda x: flag in x).sum()) for flag in
                       ['missed_initialization','early_quality_gap','worsening_relative_gap','low_detected_JF']},
        'caveats':[
            'First-frame detector boxes are assigned to GT object identities by an oracle matcher.',
            'Missed objects are assigned empty masks; missing initialization is directly observed.',
            'Early quality gap and worsening relative gap are heuristic indicators, not verified causal diagnoses.',
            'Frame-level GT-box SAM2 and detector-box SAM2 results must use identical evaluation frame/object keys.',
            'No ground-truth mask pixels are required to analyze provided J/F scores, but overlays are needed to verify qualitative failure modes.'
        ]}
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    objects.sort_values('delta_JF').to_csv(out/'per_object.csv',index=False)
    joined.drop(columns='_merge').to_csv(out/'paired_per_frame.csv',index=False)
    with (out/'summary.json').open('w',encoding='utf-8') as f:json.dump(summary,f,indent=2)
    seq=objects.groupby('sequence',as_index=False).agg(objects=('object_id','size'),matched=('initialized','sum'),gt_JF=('gt_JF','mean'),detected_JF=('detected_JF','mean'),delta_JF=('delta_JF','mean'))
    seq.sort_values('delta_JF').to_csv(out/'per_sequence.csv',index=False)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt-csv',default='outputs/m410_davis_sam2/per_frame.csv')
    p.add_argument('--detected-csv',default='outputs/m411_davis_detector_sam2/per_frame.csv')
    p.add_argument('--detections',default='outputs/m411_davis_detector_sam2/detections.json')
    p.add_argument('--output',default='outputs/m413_davis_failure_analysis')
    a=p.parse_args()
    result=analyze(a.gt_csv,a.detected_csv,a.detections,a.output)
    print(json.dumps(result,indent=2))
    print(f'Wrote reports to {a.output}')
if __name__=='__main__':main()
