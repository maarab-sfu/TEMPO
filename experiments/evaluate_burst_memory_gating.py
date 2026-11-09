"""Stage 12E: causal confidence-gated appearance-memory updates.

Place in experiments/evaluate_burst_memory_gating.py and run:
  python -m experiments.evaluate_burst_memory_gating --sequences all

Requires existing Stage 12C modules and identity_hard data. GT IDs are used
ONLY for evaluation; geometry and category are still oracle BURST inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, write_csv, pairwise_metrics


def stable_uniform(seq, index, seed):
    b = f'{seed}|{seq}|{index}'.encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'big') / 2**64


def run_sequence(seq, indices, obs, emb, centers, args, mode, random_keep=1.0):
    by_frame = defaultdict(list)
    for i in indices:
        by_frame[int(obs[i]['frame_position'])].append(i)
    memories = {}
    next_id = 0
    rows = []
    for frame in sorted(by_frame):
        frame_ids = sorted(by_frame[frame]); pred_ids = sorted(memories)
        n, m = len(frame_ids), len(pred_ids)
        utility = np.full((n, m+n), -1e6, dtype=float)
        for r in range(n):
            utility[r,m+r] = 0.0
        edges = {}
        for r,i in enumerate(frame_ids):
            category = str(obs[i]['category_id'])
            for c,pid in enumerate(pred_ids):
                mem = memories[pid]
                if mem['category'] != category:
                    continue
                appearance = float(np.max(np.asarray(mem['vectors']) @ emb[i]))
                gap = frame - mem['last_frame']
                if gap <= 0:
                    raise AssertionError('Noncausal association')
                distance = float(np.linalg.norm(centers[i]-mem['last_center']))
                st = math.exp(-0.5*(distance/args.spatial_sigma)**2)*math.exp(-(gap-1)/args.temporal_tau)
                adv = args.appearance_weight*(appearance-args.threshold)+(1-args.appearance_weight)*(st-args.spatial_threshold)
                edges[r,c] = (appearance,st,adv,gap,distance)
                if adv >= 0:
                    utility[r,c] = adv + 1e-10
        rr,cc = linear_sum_assignment(utility,maximize=True)
        assignments = dict(zip(rr.tolist(),cc.tolist()))
        pending = []
        for r,i in enumerate(frame_ids):
            c = assignments[r]
            matched = c < m and utility[r,c]>=0
            candidates = sorted((v[2] for (row,col),v in edges.items() if row==r),reverse=True)
            gap_score = candidates[0]-candidates[1] if len(candidates)>=2 else None
            if matched:
                pid=pred_ids[c]
                app,st,adv,frame_gap,dist=edges[r,c]
                if mode=='baseline':
                    keep=True
                elif mode=='gated':
                    # One candidate is not automatically reliable; treat it as
                    # insufficient competition evidence and allow an update.
                    keep = gap_score is None or gap_score >= args.margin
                elif mode=='random':
                    keep=stable_uniform(seq,i,args.seed)<random_keep
                else:
                    raise ValueError(mode)
                decision='match'
            else:
                pid=f'p{next_id:04d}';next_id+=1
                app=st=adv=frame_gap=dist=None
                keep=True  # Every new identity needs a seed embedding.
                decision='new'
            pending.append((i,pid,decision,keep,gap_score,app,st,adv,frame_gap,dist))
        # All matches are decided using pre-frame state, then committed.
        for i,pid,decision,keep,gap_score,app,st,adv,frame_gap,dist in pending:
            if pid not in memories:
                memories[pid]={'category':str(obs[i]['category_id']), 'vectors':[],
                               'last_frame':frame,'last_center':centers[i].copy()}
            mem=memories[pid]
            if keep:
                mem['vectors'].append(emb[i])
                if args.memory_size and len(mem['vectors'])>args.memory_size:
                    mem['vectors']=mem['vectors'][-args.memory_size:]
            # Spatial state always advances, even if appearance update is gated.
            mem['last_frame']=frame
            mem['last_center']=centers[i].copy()
            rows.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                         'category_id':str(obs[i]['category_id']),
                         'gt_track_id':str(obs[i]['track_id']),
                         'predicted_id':pid,'decision':decision,
                         'appearance_memory_updated':int(keep),
                         'top_two_candidate_gap':gap_score if gap_score is not None else '',
                         'selected_appearance':app if app is not None else '',
                         'selected_spatiotemporal':st if st is not None else '',
                         'selected_advantage':adv if adv is not None else '',
                         'selected_frame_gap':frame_gap if frame_gap is not None else '',
                         'selected_center_distance':dist if dist is not None else ''})
    return rows


def summarize(rows):
    matches=[r for r in rows if r['decision']=='match']
    return {'metrics':pairwise_metrics(rows),
            'matched_updates':sum(r['appearance_memory_updated'] for r in matches),
            'matched_decisions':len(matches),
            'matched_update_rate':sum(r['appearance_memory_updated'] for r in matches)/len(matches) if matches else None,
            'new_identity_seeds':sum(r['decision']=='new' for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--output',type=Path,default=Path('results/burst/memory_gating'))
    p.add_argument('--reference',type=Path,default=Path('results/burst/temporal_association/combined/assignments.csv'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--spatial-threshold',type=float,default=.50)
    p.add_argument('--appearance-weight',type=float,default=.75)
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.0)
    p.add_argument('--memory-size',type=int,default=0)
    p.add_argument('--margin',type=float,default=.05,help='Exploratory fixed margin; not calibrated')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if not (args.spatial_sigma>0 and args.temporal_tau>0 and args.memory_size>=0 and args.margin>=0 and 0<=args.appearance_weight<=1):
        p.error('Invalid parameter')
    obs=read_csv(args.input/'observations.csv')
    with np.load(args.input/'embeddings.npz') as z:
        emb=np.asarray(z['embeddings'],dtype=np.float32)
    if len(obs)!=len(emb):
        raise ValueError('Embedding/observation length mismatch')
    emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    groups=defaultdict(list)
    for i,o in enumerate(obs):
        groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences!=['all']:
        groups={k:v for k,v in groups.items() if k in args.sequences or k.split('/',1)[1] in args.sequences}
    if not groups:
        raise ValueError('No sequences selected')
    centers=load_geometry(obs,args.annotations,args.input/'normalized_mask_centers.npz')
    args.output.mkdir(parents=True,exist_ok=True)
    by_mode={}; summary={}
    # First run baseline and gated. Random control uses per-sequence gated
    # update rate as a budget, without consulting GT IDs. This is descriptive
    # and does NOT guarantee exactly equal realized update counts.
    for mode in ['baseline','gated','random']:
        all_rows=[]; per_seq={}; budgets={}
        for seq,ids in sorted(groups.items()):
            if mode=='random':
                rate=summary['gated']['per_sequence'][seq]['matched_update_rate']
                rate=1.0 if rate is None else rate
            else:
                rate=1.0
            rows=run_sequence(seq,ids,obs,emb,centers,args,mode,random_keep=rate)
            all_rows.extend(rows)
            per_seq[seq]=summarize(rows)
            if mode=='random':
                budgets[seq]=rate
            print(f'{mode:8s} {seq:65s} F1={per_seq[seq]["metrics"]["pairwise_f1"]:.4f} '
                  f'updates={per_seq[seq]["matched_updates"]}/{per_seq[seq]["matched_decisions"]}')
        d=args.output/mode;d.mkdir(parents=True,exist_ok=True)
        write_csv(d/'assignments.csv',all_rows,list(all_rows[0]))
        # Pairwise pooling across sequences is not meaningful unless IDs are
        # namespaced; instead sum TP/FP/FN and derive micro scores.
        tp=sum(v['metrics']['pairwise_tp'] for v in per_seq.values())
        fp=sum(v['metrics']['pairwise_fp'] for v in per_seq.values())
        fn=sum(v['metrics']['pairwise_fn'] for v in per_seq.values())
        micro={'tp':tp,'fp':fp,'fn':fn,
               'precision':tp/(tp+fp) if tp+fp else None,
               'recall':tp/(tp+fn) if tp+fn else None,
               'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
        summary[mode]={'per_sequence':per_seq,'micro_pairwise':micro,
                       'matched_updates':sum(v['matched_updates'] for v in per_seq.values()),
                       'matched_decisions':sum(v['matched_decisions'] for v in per_seq.values()),
                       'random_target_rates':budgets if mode=='random' else None}
        (d/'summary.json').write_text(json.dumps(summary[mode],indent=2),encoding='utf-8')
        by_mode[mode]=all_rows
    # Verify exact baseline replay using the original Stage 12C assignments.
    verified=None
    if args.reference.is_file():
        reference=read_csv(args.reference)
        a={(r['sequence'],int(r['observation_index'])):(r['predicted_id'],r['decision']) for r in reference}
        b={(r['sequence'],int(r['observation_index'])):(r['predicted_id'],r['decision']) for r in by_mode['baseline']}
        if a!=b:
            mismatches=[k for k in b if a.get(k)!=b[k]]
            raise AssertionError(f'Baseline differs from Stage 12C on {len(mismatches)} observations: {mismatches[:5]}')
        verified=True
    else:
        print('WARNING: Stage 12C reference not found; baseline not verified against reference')
    report={'stage':'12E','baseline_replay_verified':verified,
            'margin':args.margin,'random_seed':args.seed,
            'protocol':'Causal Hungarian, same Stage 12C scoring. Gating changes only appearance-memory updates; spatial state advances for all matches. New IDs always seeded.',
            'results':summary,
            'limitations':['Margin .05 is exploratory, not calibrated on held-out sequences.',
                           'Random controls use gated per-sequence update rate as an approximate budget; realized counts can differ.',
                           'Candidate gap is pre-Hungarian and can be misleading for a forced second-choice assignment.',
                           'Same-category labels and segmentation-derived geometry are GT oracle inputs.',
                           'Sparse annotated-frame gaps are not physical elapsed time.',
                           'Pairwise F1 is not IDF1 or HOTA; only five evaluation sequences.']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nBaseline verified:',verified)
    for mode in ['baseline','gated','random']:
        print(f'{mode:8s} micro pairwise F1={summary[mode]["micro_pairwise"]["f1"]:.5f}')
    print('Saved:',(args.output/'summary.json').resolve())

if __name__=='__main__':
    main()
