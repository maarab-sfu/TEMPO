"""TEMPO Stage 12H: oracle interventions on appearance-memory updates.

Install as experiments/evaluate_burst_oracle_memory.py
Run: python -m experiments.evaluate_burst_oracle_memory --sequences all

The oracle uses GT track IDs ONLY to decide whether a matched observation may
enter appearance memory, never to choose a Hungarian match or update geometry.
The anchor is the GT identity of the first observation assigned to a predicted
identity. This is a diagnostic oracle, NOT a deployable system or strict upper bound.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, pairwise_metrics


def uniform(seq, idx, seed):
    h = hashlib.sha256(f'{seed}|{seq}|{idx}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2**64


def replay(seq, indices, obs, emb, centers, args, policy, margin, target_rate, seed):
    by_frame=defaultdict(list)
    for i in indices: by_frame[int(obs[i]['frame_position'])].append(i)
    memories={}; rows=[]; next_id=0
    for frame in sorted(by_frame):
        ids=sorted(by_frame[frame]); pids=sorted(memories); n=len(ids); m=len(pids)
        utility=np.full((n,m+n),-1e6,dtype=float)
        for r in range(n): utility[r,m+r]=0.
        edges={}
        for r,i in enumerate(ids):
            for c,pid in enumerate(pids):
                mem=memories[pid]
                if mem['category']!=str(obs[i]['category_id']): continue
                app=float(np.max(np.asarray(mem['vectors']) @ emb[i]))
                dt=frame-mem['last_frame']
                if dt<=0: raise AssertionError('Noncausal frame order')
                dist=float(np.linalg.norm(centers[i]-mem['last_center']))
                st=math.exp(-.5*(dist/args.spatial_sigma)**2)*math.exp(-(dt-1)/args.temporal_tau)
                score=args.appearance_weight*(app-args.threshold)+(1-args.appearance_weight)*(st-args.spatial_threshold)
                edges[r,c]=score
                if score>=0: utility[r,c]=score+1e-10
        rr,cc=linear_sum_assignment(utility,maximize=True)
        assigned=dict(zip(rr.tolist(),cc.tolist()))
        pending=[]
        for r,i in enumerate(ids):
            c=assigned[r]; matched=c<m and utility[r,c]>=0
            scores=sorted((s for (row,col),s in edges.items() if row==r),reverse=True)
            gap=scores[0]-scores[1] if len(scores)>1 else None
            gt=str(obs[i]['track_id'])
            if matched:
                pid=pids[c]; mem=memories[pid]
                oracle_correct=gt==mem['anchor_gt']
                if policy=='baseline': keep=True
                elif policy=='fixed': keep=gap is None or gap>=margin
                elif policy=='oracle_anchor': keep=oracle_correct
                elif policy=='oracle_budget': keep=oracle_correct and uniform(seq,i,seed)<target_rate
                elif policy=='random_budget': keep=uniform(seq,i,seed)<target_rate
                elif policy=='seed_only': keep=False
                else: raise ValueError(policy)
                decision='match'
                before_counts=Counter(mem['gt_history'])
                purity=max(before_counts.values())/sum(before_counts.values())
                correct_majority=int(gt==before_counts.most_common(1)[0][0])
            else:
                pid=f'p{next_id:04d}';next_id+=1
                keep=True; decision='new';oracle_correct=None;purity=None;correct_majority=None
            pending.append((i,pid,gt,decision,keep,gap,oracle_correct,purity,correct_majority))
        for i,pid,gt,decision,keep,gap,oracle_correct,purity,correct_majority in pending:
            if pid not in memories:
                memories[pid]={'category':str(obs[i]['category_id']),'vectors':[],
                    'last_frame':frame,'last_center':centers[i].copy(),'anchor_gt':gt,'gt_history':[]}
            mem=memories[pid]
            if keep:
                mem['vectors'].append(emb[i]);mem['gt_history'].append(gt)
                if args.memory_size and len(mem['vectors'])>args.memory_size:
                    mem['vectors']=mem['vectors'][-args.memory_size:]
                    mem['gt_history']=mem['gt_history'][-args.memory_size:]
            mem['last_frame']=frame;mem['last_center']=centers[i].copy()
            rows.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                'category_id':str(obs[i]['category_id']),'gt_track_id':gt,
                'predicted_id':pid,'decision':decision,'appearance_memory_updated':int(keep),
                'oracle_correct_vs_anchor':oracle_correct if oracle_correct is not None else '',
                'memory_purity_before':purity if purity is not None else '',
                'correct_vs_memory_majority_before':correct_majority if correct_majority is not None else '',
                'top_two_candidate_gap':gap if gap is not None else ''})
    return rows


def summarize(rows):
    matches=[r for r in rows if r['decision']=='match']
    kept=sum(int(r['appearance_memory_updated']) for r in matches)
    correct=sum(r['oracle_correct_vs_anchor'] is True for r in matches)
    wrong_kept=sum(r['oracle_correct_vs_anchor'] is False and r['appearance_memory_updated'] for r in matches)
    pure=[float(r['memory_purity_before']) for r in matches]
    return {'metrics':pairwise_metrics(rows),'matched_decisions':len(matches),
            'matched_updates':kept,'matched_update_rate':kept/len(matches) if matches else None,
            'incorrect_updates_vs_anchor':wrong_kept,
            'anchor_correct_match_rate':correct/len(matches) if matches else None,
            'mean_memory_purity_before_match':float(np.mean(pure)) if pure else None}


def micro(per_seq):
    tp=sum(v['metrics']['pairwise_tp'] for v in per_seq.values())
    fp=sum(v['metrics']['pairwise_fp'] for v in per_seq.values())
    fn=sum(v['metrics']['pairwise_fn'] for v in per_seq.values())
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--stage12f-summary',type=Path,default=Path('results/burst/memory_selection/summary.json'))
    p.add_argument('--output',type=Path,default=Path('results/burst/oracle_memory'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--spatial-threshold',type=float,default=.50)
    p.add_argument('--appearance-weight',type=float,default=.75)
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.)
    p.add_argument('--memory-size',type=int,default=0)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    args=p.parse_args()
    f=json.loads(args.stage12f_summary.read_text(encoding='utf-8'))
    if f.get('stage')!='12F': raise ValueError('Expected Stage 12F summary')
    margins={x['held_out_sequence']:float(x['selected_parameter']) for x in f['calibration_folds'] if x['family']=='fixed'}
    targets={s:float(x['matched_update_rate']) for s,x in ((s,v) for s,v in f['heldout']['fixed']['per_sequence'].items())}
    obs=read_csv(args.input/'observations.csv')
    with np.load(args.input/'embeddings.npz') as z: emb=np.asarray(z['embeddings'],dtype=np.float32)
    if len(obs)!=len(emb): raise ValueError('Observation/embedding mismatch')
    emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    centers=load_geometry(obs,args.annotations,args.input/'normalized_mask_centers.npz')
    groups=defaultdict(list)
    for i,o in enumerate(obs):groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences!=['all']:
        groups={s:v for s,v in groups.items() if s in args.sequences or s.split('/',1)[1] in args.sequences}
    if not groups: raise ValueError('No sequences selected')
    args.output.mkdir(parents=True,exist_ok=True)
    policies=['baseline','fixed','oracle_anchor','seed_only']
    for seed in args.seeds: policies.extend([f'oracle_budget_{seed}',f'random_budget_{seed}'])
    summary={'stage':'12H','protocol':'Causal Hungarian matching; oracle GT only filters matched appearance-memory updates. Spatial state always advances.',
        'oracle_definition':'First assigned observation anchors each predicted identity. An oracle update is permitted iff observation GT equals anchor GT.',
        'budget_note':'Budget policies use the held-out Stage 12F fixed-gating per-sequence update rate as a Bernoulli probability. They are NOT exact count matched; oracle_budget additionally excludes incorrect updates and therefore retains fewer updates.',
        'limitations':['Oracle GT track IDs are not available at inference.',
        'Oracle-anchor policy cannot correct an incorrectly seeded predicted identity.',
        'Oracle filtering changes later matching trajectories; oracle results are not necessarily a strict upper bound.',
        'The budget rate is taken from held-out Stage 12F outcomes, making this a diagnostic intervention rather than independent held-out tuning.',
        'Oracle BURST masks/categories; only five sequences; pairwise F1 is not IDF1/HOTA.'],
        'sequences':{},'policies':{}}
    for policy in policies:summary['policies'][policy]={'per_sequence':{}}
    for seq in sorted(groups):
        if seq not in margins:raise ValueError(f'Missing 12F held-out parameter for {seq}')
        summary['sequences'][seq]={'fixed_margin':margins[seq],'target_update_rate':targets[seq]}
        for policy in policies:
            if policy.startswith('oracle_budget_'):
                kind='oracle_budget';seed=int(policy.rsplit('_',1)[1])
            elif policy.startswith('random_budget_'):
                kind='random_budget';seed=int(policy.rsplit('_',1)[1])
            else:kind=policy;seed=0
            rows=replay(seq,groups[seq],obs,emb,centers,args,kind,margins[seq],targets[seq],seed)
            result=summarize(rows)
            summary['policies'][policy]['per_sequence'][seq]=result
            if policy=='baseline':
                expected=f['baseline']['per_sequence'][seq]['metrics']['pairwise_f1']
                if abs(result['metrics']['pairwise_f1']-expected)>1e-10:raise AssertionError(f'Baseline replay mismatch {seq}')
            if policy=='fixed':
                expected=f['heldout']['fixed']['per_sequence'][seq]['metrics']['pairwise_f1']
                if abs(result['metrics']['pairwise_f1']-expected)>1e-10:raise AssertionError(f'12F fixed replay mismatch {seq}')
            if kind=='oracle_anchor' and result['incorrect_updates_vs_anchor']!=0:
                raise AssertionError('Oracle admitted an incorrect update')
            if policy in ('baseline','fixed','oracle_anchor','seed_only'):
                path=args.output/policy/(seq.replace('/','__')+'.csv')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('w',newline='',encoding='utf-8') as fh:
                    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print(seq,'baseline',round(summary['policies']['baseline']['per_sequence'][seq]['metrics']['pairwise_f1'],4),
              'fixed',round(summary['policies']['fixed']['per_sequence'][seq]['metrics']['pairwise_f1'],4),
              'oracle',round(summary['policies']['oracle_anchor']['per_sequence'][seq]['metrics']['pairwise_f1'],4))
    for policy in policies:
        summary['policies'][policy]['micro_pairwise']=micro(summary['policies'][policy]['per_sequence'])
    summary['random_control_summary']={}
    for prefix in ['oracle_budget','random_budget']:
        vals=[summary['policies'][f'{prefix}_{seed}']['micro_pairwise']['f1'] for seed in args.seeds]
        summary['random_control_summary'][prefix]={'seeds':args.seeds,'f1_values':vals,
            'mean_f1':float(np.mean(vals)),'std_f1':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Saved',args.output/'summary.json')

if __name__=='__main__':main()
