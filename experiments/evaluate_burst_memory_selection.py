"""TEMPO Stage 12F: causal memory-selection ablations and held-out calibration.

Install as experiments/evaluate_burst_memory_selection.py, then run:
    python -m experiments.evaluate_burst_memory_selection --sequences all

Requires the same Stage 12C/12E data and modules. Uses GT masks/categories as
oracle observations, but GT track IDs ONLY for evaluation and calibration labels.
Leave-one-SEQUENCE-out calibration prevents tuning on the evaluated sequence.
"""
from __future__ import annotations
import argparse
import json
import math
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, write_csv, pairwise_metrics


def uniform(seq, idx, seed):
    raw = f'{seed}|{seq}|{idx}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64


def run_sequence(seq, indices, obs, emb, centers, args, policy='baseline', margin=0.05,
                 fraction=0.6, seed=42, random_rate=1.0):
    """All decisions use earlier frames only; identity GT is not read in matching."""
    frames = defaultdict(list)
    for i in indices:
        frames[int(obs[i]['frame_position'])].append(i)
    memories, rows, past_margins = {}, [], []
    next_id = 0
    for frame in sorted(frames):
        frame_ids = sorted(frames[frame])
        pred_ids = sorted(memories)
        n, m = len(frame_ids), len(pred_ids)
        utility = np.full((n, m+n), -1e6, dtype=np.float64)
        for r in range(n):
            utility[r, m+r] = 0.0
        edges = {}
        for r, i in enumerate(frame_ids):
            category = str(obs[i]['category_id'])
            for c, pid in enumerate(pred_ids):
                mem = memories[pid]
                if mem['category'] != category:
                    continue
                app = float(np.max(np.asarray(mem['vectors']) @ emb[i]))
                gap = frame - mem['last_frame']
                if gap <= 0:
                    raise AssertionError('Noncausal match')
                dist = float(np.linalg.norm(centers[i] - mem['last_center']))
                st = math.exp(-0.5*(dist/args.spatial_sigma)**2) * math.exp(-(gap-1)/args.temporal_tau)
                advantage = args.appearance_weight*(app-args.threshold) + (1-args.appearance_weight)*(st-args.spatial_threshold)
                edges[r,c] = (app,st,advantage)
                if advantage >= 0:
                    utility[r,c] = advantage + 1e-10
        rr, cc = linear_sum_assignment(utility, maximize=True)
        assignments = dict(zip(rr.tolist(),cc.tolist()))
        pending = []
        # Freeze the confidence-history quantile for all observations in a frame.
        cutoff = None
        if policy == 'adaptive' and len(past_margins) >= args.warmup:
            cutoff = float(np.quantile(past_margins, 1-fraction))
        for r, i in enumerate(frame_ids):
            c = assignments[r]
            matched = c < m and utility[r,c] >= 0
            candidates = sorted((v[2] for (row,col),v in edges.items() if row==r),reverse=True)
            gap_score = candidates[0]-candidates[1] if len(candidates)>=2 else None
            if matched:
                pid = pred_ids[c]
                if policy == 'baseline':
                    keep = True
                elif policy == 'fixed':
                    keep = gap_score is None or gap_score >= margin
                elif policy == 'adaptive':
                    keep = gap_score is None or cutoff is None or gap_score >= cutoff
                elif policy == 'random':
                    keep = uniform(seq,i,seed) < random_rate
                else:
                    raise ValueError(policy)
                decision = 'match'
            else:
                pid = f'p{next_id:04d}'
                next_id += 1
                keep = True
                decision = 'new'
            pending.append((i,pid,decision,keep,gap_score))
        # All frame decisions committed together, so same-frame observations
        # cannot influence one another's appearance or spatial scores.
        for i,pid,decision,keep,gap_score in pending:
            if pid not in memories:
                memories[pid] = {'category':str(obs[i]['category_id']), 'vectors':[],
                                 'last_frame':frame,'last_center':centers[i].copy()}
            mem = memories[pid]
            if keep:
                mem['vectors'].append(emb[i])
                if args.memory_size and len(mem['vectors']) > args.memory_size:
                    mem['vectors'] = mem['vectors'][-args.memory_size:]
            mem['last_frame'] = frame
            mem['last_center'] = centers[i].copy()
            rows.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                         'category_id':str(obs[i]['category_id']),
                         'gt_track_id':str(obs[i]['track_id']),
                         'predicted_id':pid,'decision':decision,
                         'appearance_memory_updated':int(keep),
                         'top_two_candidate_gap':gap_score if gap_score is not None else ''})
        # Only past frames affect adaptive thresholds. Only matched observations
        # with >=2 candidates contribute to the quantile history.
        for _,_,decision,_,gap_score in pending:
            if decision == 'match' and gap_score is not None:
                past_margins.append(gap_score)
    return rows


def summarize(rows):
    matches = [r for r in rows if r['decision']=='match']
    updates = sum(int(r['appearance_memory_updated']) for r in matches)
    return {'metrics':pairwise_metrics(rows),'matched_updates':updates,
            'matched_decisions':len(matches),
            'matched_update_rate':updates/len(matches) if matches else None,
            'new_identity_seeds':sum(r['decision']=='new' for r in rows)}


def micro(per_sequence):
    tp = sum(v['metrics']['pairwise_tp'] for v in per_sequence.values())
    fp = sum(v['metrics']['pairwise_fp'] for v in per_sequence.values())
    fn = sum(v['metrics']['pairwise_fn'] for v in per_sequence.values())
    return {'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def parse_numbers(value, kind=float):
    return [kind(x.strip()) for x in value.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--output',type=Path,default=Path('results/burst/memory_selection'))
    p.add_argument('--reference',type=Path,default=Path('results/burst/temporal_association/combined/assignments.csv'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--spatial-threshold',type=float,default=.50)
    p.add_argument('--appearance-weight',type=float,default=.75)
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.)
    p.add_argument('--memory-size',type=int,default=0)
    p.add_argument('--margins',default='0,0.01,0.02,0.05,0.1,0.2')
    p.add_argument('--fractions',default='0.3,0.5,0.7,0.9')
    p.add_argument('--warmup',type=int,default=30,
                   help='Minimum earlier matched, competitive margins before adaptive filtering')
    p.add_argument('--random-seeds',default='11,22,33,44,55')
    args = p.parse_args()
    margins = parse_numbers(args.margins)
    fractions = parse_numbers(args.fractions)
    seeds = parse_numbers(args.random_seeds,int)
    if not margins or not fractions or not seeds or any(x<0 for x in margins) or any(not 0<x<=1 for x in fractions):
        p.error('Provide nonnegative margins, fractions in (0,1], and random seeds')
    if args.spatial_sigma<=0 or args.temporal_tau<=0 or args.memory_size<0 or args.warmup<1 or not 0<=args.appearance_weight<=1:
        p.error('Invalid tracking parameters')
    obs = read_csv(args.input/'observations.csv')
    with np.load(args.input/'embeddings.npz') as z:
        emb = np.asarray(z['embeddings'],dtype=np.float32)
    if len(obs)!=len(emb):
        raise ValueError('Embedding/observation count mismatch')
    emb /= np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    groups = defaultdict(list)
    for i,o in enumerate(obs):
        groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences!=['all']:
        groups = {k:v for k,v in groups.items() if k in args.sequences or k.split('/',1)[1] in args.sequences}
    if len(groups)<2:
        raise ValueError('Leave-one-sequence-out evaluation requires at least 2 sequences')
    centers = load_geometry(obs,args.annotations,args.input/'normalized_mask_centers.npz')
    args.output.mkdir(parents=True,exist_ok=True)
    sequence_names = sorted(groups)
    candidates = [('fixed',x) for x in margins] + [('adaptive',x) for x in fractions]
    cache = {}
    def get(policy, parameter, seq):
        key = (policy,parameter,seq)
        if key not in cache:
            kwargs = {'margin':parameter} if policy=='fixed' else {'fraction':parameter} if policy=='adaptive' else {}
            rows = run_sequence(seq,groups[seq],obs,emb,centers,args,policy=policy,**kwargs)
            cache[key] = (rows,summarize(rows))
        return cache[key]
    baseline_rows = []
    baseline = {}
    for seq in sequence_names:
        rows,summary = get('baseline',None,seq)
        baseline_rows.extend(rows)
        baseline[seq] = summary
    verified = None
    if args.reference.is_file():
        reference = read_csv(args.reference)
        a = {(r['sequence'],int(r['observation_index'])):(r['predicted_id'],r['decision']) for r in reference if r['sequence'] in groups}
        b = {(r['sequence'],int(r['observation_index'])):(r['predicted_id'],r['decision']) for r in baseline_rows}
        if a!=b:
            raise AssertionError('Baseline does not exactly reproduce Stage 12C assignments')
        verified = True
    else:
        print('WARNING: Reference assignments missing; baseline not independently verified')
    write_csv(args.output/'baseline_assignments.csv',baseline_rows,list(baseline_rows[0]))
    print('Baseline pooled F1:',round(micro(baseline)['f1'],5),'reference verified:',verified)
    for policy,param in candidates:
        for seq in sequence_names:
            get(policy,param,seq)
    # Each held-out sequence has a threshold selected on all OTHER sequences.
    # We calibrate fixed and adaptive families separately, then compare them.
    folds = []
    heldout = {'fixed':{},'adaptive':{}}
    for test in sequence_names:
        train = [s for s in sequence_names if s!=test]
        for family in ['fixed','adaptive']:
            options = [(p,v) for p,v in candidates if p==family]
            # Tie-break toward fewer restrictions (smaller fixed threshold,
            # larger retained fraction) to avoid unnecessary suppression.
            scored = []
            for policy,param in options:
                train_micro = micro({s:get(policy,param,s)[1] for s in train})
                tie = -param if policy=='fixed' else param
                scored.append((train_micro['f1'],tie,param))
            best_train_f1,_,chosen = max(scored)
            rows,summary = get(family,chosen,test)
            heldout[family][test] = summary
            folds.append({'held_out_sequence':test,'family':family,
                          'selected_parameter':chosen,'train_micro_f1':best_train_f1,
                          'test_pairwise_f1':summary['metrics']['pairwise_f1'],
                          'test_matched_update_rate':summary['matched_update_rate']})
            out = args.output/'heldout'/family
            out.mkdir(parents=True,exist_ok=True)
            write_csv(out/(test.replace('/','__')+'.csv'),rows,list(rows[0]))
            print(f'LOSO {family:8s} {test:62s} parameter={chosen:.3f} F1={summary["metrics"]["pairwise_f1"]:.4f}')
    # Random controls use the observed per-sequence update rate of each
    # HELD-OUT policy as a target probability; this is a descriptive budget,
    # not an exactly matched number of updates. Several seeds quantify noise.
    random_controls = {}
    for family in ['fixed','adaptive']:
        per_seed = {}
        for seed in seeds:
            per_seq = {}
            for seq in sequence_names:
                target = heldout[family][seq]['matched_update_rate']
                rows = run_sequence(seq,groups[seq],obs,emb,centers,args,
                                    policy='random',seed=seed,
                                    random_rate=1.0 if target is None else target)
                per_seq[seq] = summarize(rows)
            per_seed[str(seed)] = {'per_sequence':per_seq,'micro_pairwise':micro(per_seq)}
        random_controls[family] = per_seed
    report = {'stage':'12F','baseline_replay_verified':verified,
              'protocol':'Leave-one-sequence-out selection of memory-update policy parameter; causal online Hungarian tracker; oracle BURST observations.',
              'settings':{'margins':margins,'fractions':fractions,'warmup':args.warmup,'random_seeds':seeds},
              'baseline':{'per_sequence':baseline,'micro_pairwise':micro(baseline)},
              'calibration_folds':folds,
              'heldout':{family:{'per_sequence':result,'micro_pairwise':micro(result)} for family,result in heldout.items()},
              'random_controls':random_controls,
              'limitations':['Only five videos; leave-one-video-out results may be unstable.',
                             'Adaptive fractions are approximate online targets, not exact update budgets.',
                             'Random controls target per-sequence heldout update rates, not exactly matched counts.',
                             'Single-run baseline uses oracle GT masks/categories and GT-derived centers.',
                             'GT identity is used only for evaluation and calibration selection, never matching.',
                             'Pre-Hungarian candidate gap may not reflect assigned-vs-alternative ambiguity.',
                             'Pairwise F1 is not IDF1/HOTA; annotated frame gaps are not elapsed time.']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nPooled held-out micro F1:')
    print('baseline:',f'{report["baseline"]["micro_pairwise"]["f1"]:.5f}')
    for family in ['fixed','adaptive']:
        print(family+':',f'{report["heldout"][family]["micro_pairwise"]["f1"]:.5f}')
        values = [v['micro_pairwise']['f1'] for v in random_controls[family].values()]
        print(f'  random controls ({len(values)} seeds): mean={np.mean(values):.5f}, std={np.std(values,ddof=1) if len(values)>1 else 0:.5f}')
    print('Saved:',(args.output/'summary.json').resolve())

if __name__=='__main__':
    main()
