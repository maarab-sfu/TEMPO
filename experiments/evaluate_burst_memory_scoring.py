"""TEMPO Stage 12J: frozen-snapshot memory scoring and LOSO causal association.

Install: experiments/evaluate_burst_memory_scoring.py
Run: python -m experiments.evaluate_burst_memory_scoring --sequences all

Ground truth is used for post-hoc frozen diagnostics and calibration fold F1,
never for association or memory updates. Five sequences only; oracle GT masks,
categories and centers. All candidate policies use unlimited memory and the
Stage 12F fixed confidence gate (0.01, independently LOSO selected).
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, pairwise_metrics


def appearance(sims, policy):
    s=np.sort(np.asarray(sims,dtype=np.float64))[::-1]
    if policy=='max':return float(s[0])
    if policy.startswith('top'):
        k=int(policy[3:]);return float(s[:k].mean())
    if policy.startswith('softmax'):
        tau=float(policy.split('_')[1]);z=np.exp((s-s[0])/tau);return float(np.dot(s,z)/z.sum())
    if policy=='second_best':return float(s[1] if len(s)>1 else s[0])
    raise ValueError(policy)


def micro(per):
    tp=sum(x['pairwise_tp'] for x in per.values());fp=sum(x['pairwise_fp'] for x in per.values());fn=sum(x['pairwise_fn'] for x in per.values())
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def run(seq,indices,obs,emb,centers,args,policy,margin,collect_frozen=False):
    by_frame=defaultdict(list)
    for i in indices:by_frame[int(obs[i]['frame_position'])].append(i)
    memories={};rows=[];next_id=0;frozen=[]
    for frame in sorted(by_frame):
        ids=sorted(by_frame[frame]);pids=sorted(memories);n=len(ids);m=len(pids)
        utility=np.full((n,m+n),-1e6,dtype=float)
        for r in range(n):utility[r,m+r]=0.
        edges={};snapshots={}
        for r,i in enumerate(ids):
            for c,pid in enumerate(pids):
                mem=memories[pid]
                if mem['category']!=str(obs[i]['category_id']):continue
                sims=mem['vectors']@emb[i]
                app=appearance(sims,policy)
                dt=frame-mem['last_frame']
                if dt<=0:raise AssertionError('Noncausal frame order')
                dist=float(np.linalg.norm(centers[i]-mem['last_center']))
                st=math.exp(-.5*(dist/args.spatial_sigma)**2)*math.exp(-(dt-1)/args.temporal_tau)
                score=args.appearance_weight*(app-args.threshold)+(1-args.appearance_weight)*(st-args.spatial_threshold)
                edges[r,c]=score
                if score>=0:utility[r,c]=score+1e-10
                if collect_frozen:
                    snapshots[(r,c)]=(sims.astype(np.float32).tolist(),st,Counter(mem['gt_history']).most_common(1)[0][0],len(mem['gt_history']))
        rr,cc=linear_sum_assignment(utility,maximize=True)
        assigned=dict(zip(rr.tolist(),cc.tolist()));pending=[]
        for r,i in enumerate(ids):
            c=assigned[r];matched=c<m and utility[r,c]>=0
            scores=sorted((s for (row,col),s in edges.items() if row==r),reverse=True)
            gap=scores[0]-scores[1] if len(scores)>1 else None
            gt=str(obs[i]['track_id'])
            if matched:
                pid=pids[c];decision='match';keep=gap is None or gap>=margin
            else:
                pid=f'p{next_id:04d}';next_id+=1;decision='new';keep=True
            pending.append((i,pid,gt,decision,keep))
            if collect_frozen and len(snapshots):
                for cc2 in range(m):
                    if (r,cc2) not in snapshots:continue
                    sim,st,majority,size=snapshots[(r,cc2)]
                    frozen.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                                   'gt_track_id':gt,'candidate_predicted_id':pids[cc2],
                                   'candidate_majority_gt':majority,'candidate_memory_size':size,
                                   'candidate_correct_vs_majority':int(gt==majority),
                                   'spatiotemporal_score':st,'cosine_similarities':sim})
        for i,pid,gt,decision,keep in pending:
            if pid not in memories:
                memories[pid]={'category':str(obs[i]['category_id']),'vectors':np.empty((0,emb.shape[1]),dtype=np.float32),
                               'gt_history':[],'last_frame':frame,'last_center':centers[i].copy()}
            mem=memories[pid]
            if keep:
                mem['vectors']=np.concatenate((mem['vectors'],emb[i:i+1]),axis=0)
                mem['gt_history'].append(gt)
            mem['last_frame']=frame;mem['last_center']=centers[i].copy()
            rows.append({'sequence':seq,'observation_index':i,'frame_position':frame,'category_id':str(obs[i]['category_id']),
                         'gt_track_id':gt,'predicted_id':pid,'decision':decision,'appearance_memory_updated':int(keep)})
    return rows,frozen


def frozen_analysis(frozen,policies,args):
    # Same candidate memories and spatial features for every policy. This is a
    # post-hoc majority-label proxy, not identity F1 and not an oracle tracker.
    groups=defaultdict(list)
    for item in frozen:groups[(item['sequence'],item['observation_index'])].append(item)
    out={}
    for policy in policies:
        n=0;correct=0;eligible=0;eligible_correct=0;gaps=[]
        for entries in groups.values():
            if len(entries)<2:continue
            scored=[]
            for e in entries:
                a=appearance(e['cosine_similarities'],policy)
                val=args.appearance_weight*(a-args.threshold)+(1-args.appearance_weight)*(e['spatiotemporal_score']-args.spatial_threshold)
                scored.append((val,e['candidate_correct_vs_majority']))
            scored.sort(key=lambda x:x[0],reverse=True)
            n+=1;correct+=int(scored[0][1]);gaps.append(scored[0][0]-scored[1][0])
            if scored[0][0]>=0:eligible+=1;eligible_correct+=int(scored[0][1])
        out[policy]={'queries_with_2plus_candidates':n,'top1_majority_proxy_accuracy':correct/n if n else None,
                     'top1_eligible_queries':eligible,'eligible_top1_majority_proxy_accuracy':eligible_correct/eligible if eligible else None,
                     'mean_top2_utility_gap':float(np.mean(gaps)) if gaps else None}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--stage12f-summary',type=Path,default=Path('results/burst/memory_selection/summary.json'))
    p.add_argument('--output',type=Path,default=Path('results/burst/memory_scoring'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--spatial-threshold',type=float,default=.50)
    p.add_argument('--appearance-weight',type=float,default=.75)
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.)
    args=p.parse_args()
    policies=['max','top2','top3','top5','second_best','softmax_0.01','softmax_0.02','softmax_0.05','softmax_0.10']
    f=json.loads(args.stage12f_summary.read_text(encoding='utf-8'))
    if f.get('stage')!='12F':raise ValueError('Expected Stage 12F summary')
    margins={x['held_out_sequence']:float(x['selected_parameter']) for x in f['calibration_folds'] if x['family']=='fixed'}
    obs=read_csv(args.input/'observations.csv')
    with np.load(args.input/'embeddings.npz') as z:emb=np.asarray(z['embeddings'],dtype=np.float32)
    if len(obs)!=len(emb):raise ValueError('Observation/embedding mismatch')
    emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    centers=load_geometry(obs,args.annotations,args.input/'normalized_mask_centers.npz')
    groups=defaultdict(list)
    for i,o in enumerate(obs):groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences!=['all']:
        groups={s:v for s,v in groups.items() if s in args.sequences or s.split('/',1)[1] in args.sequences}
    if not groups:raise ValueError('No sequences selected')
    if any(s not in margins for s in groups):raise ValueError('Missing fixed LOSO margin')
    args.output.mkdir(parents=True,exist_ok=True)
    results={policy:{} for policy in policies};frozen=[]
    for seq in sorted(groups):
        for policy in policies:
            rows,shots=run(seq,groups[seq],obs,emb,centers,args,policy,margins[seq],collect_frozen=(policy=='max'))
            results[policy][seq]=pairwise_metrics(rows)
            if policy=='max':
                frozen.extend(shots)
                ref=f['heldout']['fixed']['per_sequence'][seq]['metrics']['pairwise_f1']
                if abs(results[policy][seq]['pairwise_f1']-ref)>1e-10:
                    raise AssertionError(f'Stage 12F gated replay mismatch: {seq}')
        print('Completed',seq,flush=True)
    # LOSO selects policy using only the other sequences' causal pairwise F1.
    selected={};heldout={}
    for seq in sorted(groups):
        train=[s for s in groups if s!=seq]
        if not train:raise ValueError('LOSO needs at least two sequences')
        ranked=sorted(policies,key=lambda pol:(-micro({s:results[pol][s] for s in train})['f1'],pol))
        selected[seq]={'selected_policy':ranked[0], 'train_micro_f1':micro({s:results[ranked[0]][s] for s in train})['f1'],
                       'train_sequences':train,'heldout_f1':results[ranked[0]][seq]['pairwise_f1']}
        heldout[seq]=results[ranked[0]][seq]
    report={'stage':'12J','protocol':'Frozen baseline gated memory snapshots plus full causal Hungarian replay; LOSO aggregation-policy selection.',
            'policies':policies,'stage12f_fixed_margins':{s:margins[s] for s in groups},
            'frozen_diagnostics':frozen_analysis(frozen,policies,args),
            'causal_per_policy':{pol:{'micro_pairwise':micro(results[pol]),'per_sequence':results[pol]} for pol in policies},
            'loso_selection':selected,'loso_micro_pairwise':micro(heldout),
            'limitations':['GT-derived masks/categories/centers; only five sequences.',
              'Frozen candidate majority-GT label is a post-hoc proxy and can be contaminated.',
              'Frozen snapshots are generated by the fixed-gated max-similarity reference tracker.',
              'Candidate counts differ by frame; single-candidate frozen queries excluded from top1 comparison.',
              'Policy selection is LOSO over these five sequences; all policy families and hyperparameters were proposed after earlier experiments on the same benchmark.',
              'Changing aggregation alters score scale and eligibility; original appearance and spatial thresholds are frozen, not independently recalibrated.',
              'Top-two gate uses pre-Hungarian same-category candidate utilities, including ineligible edges.',
              'Pairwise F1 is not IDF1/HOTA.']}
    out=args.output/'summary.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',out)
    for pol in policies:print(pol,round(report['causal_per_policy'][pol]['micro_pairwise']['f1'],6))
    print('LOSO',round(report['loso_micro_pairwise']['f1'],6))

if __name__=='__main__':main()
