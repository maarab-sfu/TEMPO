"""TEMPO Stage 12K: LOSO score calibration and assignment-margin diagnostics.

Install as experiments/evaluate_burst_association_calibration.py
Run: python -m experiments.evaluate_burst_association_calibration --sequences all

All ground truth is used only for evaluation and leave-one-sequence-out selection.
Candidate geometry, masks, categories are GT-derived (oracle input). The held-out
sequence never selects its own parameters. This is exploratory on five previously
studied sequences, not independent test-set validation.
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, pairwise_metrics
from experiments.evaluate_burst_memory_scoring import appearance, micro


def solve(utility):
    r,c=linear_sum_assignment(utility,maximize=True)
    return dict(zip(r.tolist(),c.tolist())),float(utility[r,c].sum())


def simulate(seq, indices, obs, emb, centers, args, cfg, collect=False):
    by_frame=defaultdict(list)
    for i in indices:by_frame[int(obs[i]['frame_position'])].append(i)
    memories={}; rows=[]; diagnostic=[]; next_id=0
    for frame in sorted(by_frame):
        ids=sorted(by_frame[frame]); pids=sorted(memories); n=len(ids);m=len(pids)
        util=np.full((n,m+n),-1e6,dtype=np.float64)
        for r in range(n):util[r,m+r]=0.
        edges={};meta={}
        for r,i in enumerate(ids):
            for c,pid in enumerate(pids):
                mem=memories[pid]
                if mem['category']!=str(obs[i]['category_id']):continue
                sim=appearance(mem['vectors']@emb[i],cfg['policy'])
                gap=frame-mem['last_frame']
                if gap<=0:raise AssertionError('Noncausal frame ordering')
                distance=float(np.linalg.norm(centers[i]-mem['last_center']))
                st=math.exp(-.5*(distance/args.spatial_sigma)**2)*math.exp(-(gap-1)/args.temporal_tau)
                score=cfg['weight']*(sim-cfg['appearance_threshold'])+(1-cfg['weight'])*(st-cfg['spatial_threshold'])
                edges[r,c]=score
                meta[r,c]=(sim,st,mem['gt_history'][-1] if mem['gt_history'] else None,
                           Counter(mem['gt_history']).most_common(1)[0][0])
                if score>=0:util[r,c]=score+1e-10
        assigned,total=solve(util)
        pending=[]
        for r,i in enumerate(ids):
            c=assigned[r]; matched=c<m and util[r,c]>=0
            ranked=sorted((v for (rr,_),v in edges.items() if rr==r),reverse=True)
            pre_gap=ranked[0]-ranked[1] if len(ranked)>1 else None
            gt=str(obs[i]['track_id'])
            if matched:
                pid=pids[c];decision='match';keep=pre_gap is None or pre_gap>=args.gate_margin
            else:
                pid=f'p{next_id:04d}';next_id+=1;decision='new';keep=True
            pending.append((i,pid,gt,decision,keep))
            if collect:
                assigned_score=edges.get((r,c)) if matched else None
                alternatives=[v for (rr,cc),v in edges.items() if rr==r and cc!=c]
                best_other=max(alternatives) if alternatives else None
                assigned_gap=(assigned_score-best_other) if assigned_score is not None and best_other is not None else None
                # Counterfactual global assignment margin: forbid this row's assigned
                # edge, then re-solve the *same* frame utility. Can be zero on ties.
                alternate=util.copy();alternate[r,c]=-1e6
                _,alternative_total=solve(alternate)
                global_gap=max(0.,total-alternative_total)
                majority=meta[r,c][3] if matched else None
                latest=meta[r,c][2] if matched else None
                diagnostic.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                    'gt_track_id':gt,'predicted_id':pid,'decision':decision,
                    'correct_majority_proxy':int(gt==majority) if matched else None,
                    'correct_latest_proxy':int(gt==latest) if matched else None,
                    'pre_hungarian_top2_gap':pre_gap,'assigned_vs_best_other_gap':assigned_gap,
                    'global_assignment_gap':global_gap,'assigned_score':assigned_score,
                    'assigned_appearance':meta[r,c][0] if matched else None,
                    'assigned_spatiotemporal':meta[r,c][1] if matched else None,
                    'appearance_updated':int(keep)})
        for i,pid,gt,decision,keep in pending:
            if pid not in memories:
                memories[pid]={'category':str(obs[i]['category_id']),
                    'vectors':np.empty((0,emb.shape[1]),dtype=np.float32),
                    'gt_history':[],'last_frame':frame,'last_center':centers[i].copy()}
            mem=memories[pid]
            if keep:
                mem['vectors']=np.concatenate((mem['vectors'],emb[i:i+1]),axis=0)
                mem['gt_history'].append(gt)
            mem['last_frame']=frame;mem['last_center']=centers[i].copy()
            rows.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                'category_id':str(obs[i]['category_id']),'gt_track_id':gt,'predicted_id':pid,
                'decision':decision,'appearance_memory_updated':int(keep)})
    return rows,diagnostic


def auc(y,x):
    # AUROC for predicting *error* from low confidence, using average ranks.
    if len(set(y))<2:return None
    order=np.argsort(x,kind='mergesort');ranks=np.empty(len(x),dtype=float)
    sorted_x=np.asarray(x)[order];start=0
    while start<len(x):
        end=start+1
        while end<len(x) and sorted_x[end]==sorted_x[start]:end+=1
        ranks[order[start:end]]=(start+1+end)/2
        start=end
    y=np.asarray(y,dtype=bool);np_=int(y.sum());nn=len(y)-np_
    return float((ranks[y].sum()-np_*(np_+1)/2)/(np_*nn))


def diagnostic_summary(rows):
    matched=[r for r in rows if r['decision']=='match']
    out={'matched_count':len(matched),'new_count':len(rows)-len(matched)}
    for field in ['pre_hungarian_top2_gap','assigned_vs_best_other_gap','global_assignment_gap']:
        subset=[r for r in matched if r[field] is not None and np.isfinite(r[field])]
        values=np.asarray([r[field] for r in subset],dtype=float)
        errors=np.asarray([1-r['correct_majority_proxy'] for r in subset],dtype=int)
        out[field]={'count':len(subset),'error_rate':float(errors.mean()) if len(errors) else None,
                    'error_detection_auroc':auc(errors,-values) if len(values) else None,
                    'mean_correct':float(values[errors==0].mean()) if (errors==0).any() else None,
                    'mean_error':float(values[errors==1].mean()) if (errors==1).any() else None}
    out['majority_proxy_match_accuracy']=sum(r['correct_majority_proxy'] for r in matched)/len(matched) if matched else None
    out['latest_proxy_match_accuracy']=sum(r['correct_latest_proxy'] for r in matched)/len(matched) if matched else None
    return out


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--stage12f-summary',type=Path,default=Path('results/burst/memory_selection/summary.json'))
    p.add_argument('--output',type=Path,default=Path('results/burst/association_calibration'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.)
    p.add_argument('--gate-margin',type=float,default=.01)
    p.add_argument('--max-configs',type=int,default=0,help='Optional cap for debugging; 0=all')
    args=p.parse_args()
    f=json.loads(args.stage12f_summary.read_text(encoding='utf-8'))
    if f.get('stage')!='12F':raise ValueError('Expected Stage12F summary')
    margins={x['held_out_sequence']:float(x['selected_parameter']) for x in f['calibration_folds'] if x['family']=='fixed'}
    if any(abs(v-args.gate_margin)>1e-12 for v in margins.values()):
        raise ValueError('Stage12F held-out fixed gate differs from requested margin')
    obs=read_csv(args.input/'observations.csv')
    with np.load(args.input/'embeddings.npz') as z:emb=np.asarray(z['embeddings'],dtype=np.float32)
    if len(obs)!=len(emb):raise ValueError('Embedding/observation mismatch')
    emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    centers=load_geometry(obs,args.annotations,args.input/'normalized_mask_centers.npz')
    groups=defaultdict(list)
    for i,o in enumerate(obs):groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences!=['all']:
        groups={s:v for s,v in groups.items() if s in args.sequences or s.split('/',1)[1] in args.sequences}
    if len(groups)<2:raise ValueError('LOSO needs at least two sequences')
    if any(s not in margins for s in groups):raise ValueError('Stage12F calibration missing a sequence')
    args.output.mkdir(parents=True,exist_ok=True)
    # Small, explicitly predeclared grid: no tuning on held-out sequence.
    configs=[{'policy':pol,'weight':w,'appearance_threshold':a,'spatial_threshold':st}
        for pol in ['max','softmax_0.01','softmax_0.02']
        for w in [.65,.75,.85]
        for a in [.88,.90,.92]
        for st in [.40,.50,.60]]
    if args.max_configs>0:configs=configs[:args.max_configs]
    baseline={'policy':'max','weight':.75,'appearance_threshold':.90,'spatial_threshold':.50}
    if baseline not in configs:configs.insert(0,baseline)
    per={};diagnostics=[];base_rows=[]
    for k,cfg in enumerate(configs):
        key=f'c{k:03d}';per[key]={}
        for seq in sorted(groups):
            collect=(cfg==baseline)
            rows,diag=simulate(seq,groups[seq],obs,emb,centers,args,cfg,collect=collect)
            per[key][seq]=pairwise_metrics(rows)
            if collect:
                ref=f['heldout']['fixed']['per_sequence'][seq]['metrics']['pairwise_f1']
                if abs(per[key][seq]['pairwise_f1']-ref)>1e-10:
                    raise AssertionError(f'Baseline Stage12F mismatch for {seq}: {per[key][seq]["pairwise_f1"]} vs {ref}')
                if not diagnostics or seq not in {d['sequence'] for d in diagnostics}:
                    diagnostics.extend(diag);base_rows.extend(rows)
        print(f'Configuration {k+1}/{len(configs)}: {key} {cfg}',flush=True)
    # Train-only calibration. Tie-break in favor of the simpler max baseline.
    priority=lambda cfg:(0 if cfg['policy']=='max' else 1,abs(cfg['weight']-.75),
        abs(cfg['appearance_threshold']-.90),abs(cfg['spatial_threshold']-.50),cfg['policy'])
    selected={};heldout={}
    for seq in sorted(groups):
        train=[s for s in groups if s!=seq]
        ranked=sorted(per,key=lambda k:(-micro({s:per[k][s] for s in train})['f1'],priority(configs[int(k[1:])]),k))
        best=ranked[0]
        selected[seq]={'config_id':best,'config':configs[int(best[1:])],
                       'train_micro_f1':micro({s:per[best][s] for s in train})['f1'],
                       'heldout_f1':per[best][seq]['pairwise_f1'],'train_sequences':train}
        heldout[seq]=per[best][seq]
    by_seq={s:diagnostic_summary([r for r in diagnostics if r['sequence']==s]) for s in sorted(groups)}
    report={'stage':'12K','protocol':'Predeclared LOSO calibration grid; full causal tracking; baseline assignment confidence diagnostics.',
        'baseline':baseline,'configurations':{f'c{k:03d}':c for k,c in enumerate(configs)},
        'baseline_micro':micro({s:per[f'c{configs.index(baseline):03d}'][s] for s in groups}),
        'loso_micro':micro(heldout),'loso_selection':selected,
        'per_configuration':{k:{'micro':micro(v),'per_sequence':v} for k,v in per.items()},
        'baseline_assignment_diagnostics':{'pooled':diagnostic_summary(diagnostics),'per_sequence':by_seq},
        'limitations':['Five previously investigated sequences; LOSO does not constitute untouched external validation.',
        'GT-derived masks, categories and centers; GT identity only for evaluation/calibration diagnostics.',
        'GT-majority correctness is a contaminated-memory proxy, not identity-switch ground truth.',
        'The global assignment gap is a counterfactual utility difference on the same frame, not a probability.',
        'Gating remains the original pre-Hungarian top-two gap; all configurations use gate 0.01.',
        'Pairwise F1 is not IDF1/HOTA; per-sequence parameter selection may overfit with only four training sequences.']}
    write_csv(args.output/'baseline_assignment_diagnostics.csv',diagnostics)
    write_csv(args.output/'baseline_observations.csv',base_rows)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Baseline F1',report['baseline_micro']['f1'],'LOSO F1',report['loso_micro']['f1'])
    print('Saved',args.output/'summary.json')

if __name__=='__main__':main()
