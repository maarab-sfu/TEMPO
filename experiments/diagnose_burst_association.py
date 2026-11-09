"""Stage 12D: diagnostic replay of Stage 12C, without changing associations.

Save as experiments/diagnose_burst_association.py and run from project root:
  python -m experiments.diagnose_burst_association --sequences all

Requires Stage 12C's module, BURST annotations, numpy and scipy.
GT IDs are used ONLY to label decisions for analysis, never to score/assign.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, write_csv, pairwise_metrics


def auroc(scores, labels):
    """Probability that a randomly chosen correct match ranks above an incorrect one."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    good = np.isfinite(x)
    x, y = x[good], y[good]
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(x, kind='mergesort')
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def risk_at_coverage(scores, correct, coverage):
    x = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=bool)
    good = np.isfinite(x)
    x, y = x[good], y[good]
    if not len(x):
        return None
    k = max(1, math.floor(len(x) * coverage))
    chosen = np.argsort(-x, kind='mergesort')[:k]
    return float(np.mean(~y[chosen]))


def evaluate(seq, indices, obs, emb, centers, args):
    by_frame = defaultdict(list)
    for i in indices:
        by_frame[int(obs[i]['frame_position'])].append(i)
    memories = {}
    gt_seen = set()
    next_id = 0
    rows = []
    for frame in sorted(by_frame):
        frame_ids = sorted(by_frame[frame]); pred_ids = sorted(memories)
        n, m = len(frame_ids), len(pred_ids)
        utility = np.full((n, m + n), -1e6, dtype=float)
        for r in range(n):
            utility[r, m+r] = 0.0
        edges = {}
        for r, i in enumerate(frame_ids):
            category = str(obs[i]['category_id'])
            for c, pid in enumerate(pred_ids):
                mem = memories[pid]
                if mem['category'] != category:
                    continue
                appearance = float(np.max(np.asarray(mem['vectors']) @ emb[i]))
                gap = frame - mem['last_frame']
                assert gap > 0
                distance = float(np.linalg.norm(centers[i] - mem['last_center']))
                spatial = math.exp(-0.5 * (distance / args.spatial_sigma)**2)
                recency = math.exp(-(gap-1) / args.temporal_tau)
                st = spatial * recency
                advantage = (args.appearance_weight * (appearance - args.threshold)
                             + (1-args.appearance_weight) * (st - args.spatial_threshold))
                edges[r,c] = (appearance, st, advantage, gap, distance)
                if advantage >= 0:
                    utility[r,c] = advantage + 1e-10
        rr, cc = linear_sum_assignment(utility, maximize=True)
        assignments = dict(zip(rr.tolist(), cc.tolist()))
        pending = []
        for r,i in enumerate(frame_ids):
            gt = str(obs[i]['track_id'])
            c = assignments[r]
            matched = c < m and utility[r,c] >= 0
            # Candidate scores are computed from the pre-frame memory snapshot.
            candidates = sorted(((edges[r,j][2], j) for j in range(m) if (r,j) in edges), reverse=True)
            eligible = [(s,j) for s,j in candidates if s >= 0]
            top1 = candidates[0][0] if candidates else None
            top2 = candidates[1][0] if len(candidates)>1 else None
            top_gap = top1-top2 if top2 is not None else None
            if matched:
                pid = pred_ids[c]
                app,st,adv,gap,dist = edges[r,c]
                history = memories[pid]['gt_history']
                majority_gt, majority_count = Counter(history).most_common(1)[0]
                # GT-based diagnostic: match agrees with majority historical identity.
                majority_correct = gt == majority_gt
                # Also check most recent member: helps expose contaminated memories.
                latest_correct = gt == history[-1]
                purity = majority_count / len(history)
                competitors = [s for s,j in candidates if j != c]
                selected_gap = adv - max(competitors) if competitors else None
                correct = majority_correct
                decision = 'match'
            else:
                pid = f'p{next_id:04d}'; next_id += 1
                app=st=adv=gap=dist=selected_gap=purity=None
                majority_correct=latest_correct=None
                correct = None
                decision = 'new'
            pending.append((i,pid,decision,app,st,adv,gap,dist,top1,top2,top_gap,
                            selected_gap,len(eligible),majority_correct,latest_correct,purity,
                            gt in gt_seen,gt))
        # Freeze matching decisions for the whole frame, then commit.
        for (i,pid,decision,app,st,adv,gap,dist,top1,top2,top_gap,
             selected_gap,eligible_count,majority_correct,latest_correct,purity,seen_before,gt) in pending:
            if pid not in memories:
                memories[pid] = {'category':str(obs[i]['category_id']), 'vectors':[],
                                 'last_frame':frame, 'last_center':centers[i].copy(), 'gt_history':[]}
            mem = memories[pid]
            mem['vectors'].append(emb[i])
            if args.memory_size and len(mem['vectors']) > args.memory_size:
                mem['vectors'] = mem['vectors'][-args.memory_size:]
            mem['gt_history'].append(gt) # evaluation only, never consulted in matching
            mem['last_frame'] = frame
            mem['last_center'] = centers[i].copy()
            gt_seen.add(gt)
            rows.append({'sequence':seq, 'observation_index':i,'frame_position':frame,
                         'category_id':str(obs[i]['category_id']),'gt_track_id':gt,
                         'predicted_id':pid,'decision':decision,
                         'new_id_for_previously_seen_gt':int(seen_before) if decision=='new' else '',
                         'match_correct_majority_gt':int(majority_correct) if decision=='match' else '',
                         'match_correct_latest_gt':int(latest_correct) if decision=='match' else '',
                         'memory_purity_before_match':purity,
                         'selected_appearance':app,'selected_spatiotemporal':st,
                         'selected_advantage':adv,'selected_frame_gap':gap,
                         'selected_center_distance':dist,'best_candidate_advantage':top1,
                         'second_candidate_advantage':top2,'top_two_candidate_gap':top_gap,
                         'selected_vs_best_other_gap':selected_gap,
                         'eligible_candidate_count':eligible_count})
    return rows


def summarize(rows):
    matches = [r for r in rows if r['decision']=='match']
    news = [r for r in rows if r['decision']=='new']
    correct = np.asarray([bool(r['match_correct_majority_gt']) for r in matches], dtype=bool)
    result = {'observations':len(rows),'matches':len(matches),'new_id_decisions':len(news),
              'new_id_for_previously_seen_gt':sum(bool(r['new_id_for_previously_seen_gt']) for r in news),
              'majority_gt_match_accuracy':float(np.mean(correct)) if len(correct) else None,
              'latest_gt_match_accuracy':float(np.mean([bool(r['match_correct_latest_gt']) for r in matches])) if matches else None,
              'note':'Majority GT is a diagnostic label of the predicted memory history, not an association input.'}
    for field in ['selected_appearance','selected_spatiotemporal','selected_advantage',
                  'top_two_candidate_gap','selected_vs_best_other_gap','memory_purity_before_match']:
        vals = [float(r[field]) if r[field] is not None else np.nan for r in matches]
        result[field] = {'available':int(np.isfinite(vals).sum()),
                         'correct_match_auroc':auroc(vals,correct),
                         'error_risk_at_25pct_coverage':risk_at_coverage(vals,correct,.25),
                         'error_risk_at_50pct_coverage':risk_at_coverage(vals,correct,.50),
                         'error_risk_at_75pct_coverage':risk_at_coverage(vals,correct,.75)}
    # This is an oracle diagnostic, not an available deployment confidence feature.
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--output',type=Path,default=Path('results/burst/association_diagnostics'))
    p.add_argument('--reference',type=Path,default=Path('results/burst/temporal_association/combined/assignments.csv'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--spatial-threshold',type=float,default=.50)
    p.add_argument('--appearance-weight',type=float,default=.75)
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.0)
    p.add_argument('--memory-size',type=int,default=0)
    args=p.parse_args()
    if not (-1<=args.threshold<=1 and 0<=args.spatial_threshold<=1 and 0<=args.appearance_weight<=1 and args.spatial_sigma>0 and args.temporal_tau>0 and args.memory_size>=0):
        p.error('Invalid parameter values')
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
        raise ValueError('No matching sequences')
    centers=load_geometry(obs,args.annotations,args.input/'normalized_mask_centers.npz')
    all_rows=[]; per_sequence={}; per_sequence_pairwise={}
    for seq,indices in sorted(groups.items()):
        rows=evaluate(seq,indices,obs,emb,centers,args)
        all_rows.extend(rows)
        per_sequence[seq]=summarize(rows)
        per_sequence_pairwise[seq]=pairwise_metrics(rows)
        print(f"{seq}: majority-GT match accuracy={per_sequence[seq]['majority_gt_match_accuracy']:.3f}, "
              f"pairwise F1={per_sequence_pairwise[seq]['pairwise_f1']:.3f}")
    # Verify that diagnostic replay did not change any associations.
    replay_verified=None
    if args.reference.is_file():
        reference=read_csv(args.reference)
        a={(r['sequence'],int(r['observation_index'])):(r['predicted_id'],r['decision']) for r in reference}
        b={(r['sequence'],int(r['observation_index'])):(r['predicted_id'],r['decision']) for r in all_rows}
        if a!=b:
            bad=[k for k in b if k not in a or b[k]!=a[k]]
            raise AssertionError(f'Diagnostic replay differs from Stage 12C on {len(bad)} observations; first: {bad[:5]}')
        replay_verified=True
    else:
        print(f'WARNING: reference assignments not found at {args.reference}; replay unverified')
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'decision_diagnostics.csv',all_rows,list(all_rows[0]))
    summary={'protocol':'Stage 12D diagnostics of causal combined Stage 12C association, unchanged matching.',
             'parameters':{'threshold':args.threshold,'spatial_threshold':args.spatial_threshold,
                           'appearance_weight':args.appearance_weight,'spatial_sigma':args.spatial_sigma,
                           'temporal_tau':args.temporal_tau,'memory_size':args.memory_size},
             'replay_verified_against_stage12c':replay_verified,
             'pooled':summarize(all_rows),'per_sequence':per_sequence,
             'pairwise_per_sequence':per_sequence_pairwise,
             'limitations':['Correctness of a match is evaluated against majority GT identity in its predicted memory before the match; this is a diagnostic proxy, not global identity clustering accuracy.',
                            'Memory purity is GT-oracle information and must not be used as a deployable confidence feature.',
                            'New-ID fragmentation flag only means GT identity appeared earlier; a legitimate return after occlusion may be uncertain.',
                            'AUROC and risk coverage are descriptive on these five videos, not held-out calibrated estimates.',
                            'Margins computed before one-to-one assignment can be negative for a selected match because another observation wins a better memory.',
                            'GT segmentation geometry and category labels remain oracle; GT IDs are evaluation-only.',
                            'Sparse annotated-frame gaps are not physical elapsed time.']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Saved:',(args.output/'summary.json').resolve())
    print('Replay verified:',replay_verified)
    print('Pooled diagnostic:',json.dumps(summary['pooled'],indent=2))

if __name__=='__main__':
    main()
