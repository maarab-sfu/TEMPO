"""TEMPO Stage 12G: diagnostic replay of baseline vs calibrated fixed-gating.

Run: python -m experiments.evaluate_burst_memory_mechanism --sequences all
Requires Stage 12C/12F inputs. This is an observational mechanism analysis,
NOT a causal attribution of individual update decisions.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, pairwise_metrics


def replay(seq, indices, obs, emb, centers, args, margin):
    frames=defaultdict(list)
    for i in indices: frames[int(obs[i]['frame_position'])].append(i)
    memories={}; rows=[]; next_id=0
    for frame in sorted(frames):
        ids=sorted(frames[frame]); pids=sorted(memories); n=len(ids); m=len(pids)
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
                edges[r,c]=(app,st,score)
                if score>=0: utility[r,c]=score+1e-10
        rr,cc=linear_sum_assignment(utility,maximize=True)
        assigned=dict(zip(rr.tolist(),cc.tolist()))
        pending=[]
        for r,i in enumerate(ids):
            c=assigned[r]; matched=c<m and utility[r,c]>=0
            options=sorted(((e[2],col) for (row,col),e in edges.items() if row==r),reverse=True)
            top_gap=options[0][0]-options[1][0] if len(options)>1 else None
            gt=str(obs[i]['track_id'])
            if matched:
                pid=pids[c]; mem=memories[pid]
                counts=Counter(mem['gt_history'])
                majority=counts.most_common(1)[0][0]
                correct=int(majority==gt)
                pure=int(len(counts)==1)
                purity=max(counts.values())/sum(counts.values())
                selected=edges[r,c][2]
                others=[v[2] for (row,col),v in edges.items() if row==r and col!=c]
                assigned_gap=selected-max(others) if others else None
                keep=margin is None or top_gap is None or top_gap>=margin
                decision='match'
            else:
                pid=f'p{next_id:04d}'; next_id+=1; keep=True; decision='new'
                correct=None; pure=None; purity=None; selected=None; assigned_gap=None
            pending.append((i,pid,decision,keep,top_gap,assigned_gap,correct,pure,purity,selected))
        for i,pid,decision,keep,top_gap,assigned_gap,correct,pure,purity,selected in pending:
            gt=str(obs[i]['track_id'])
            if pid not in memories:
                memories[pid]={'category':str(obs[i]['category_id']),'vectors':[],
                               'last_frame':frame,'last_center':centers[i].copy(), 'gt_history':[]}
            mem=memories[pid]
            if keep:
                mem['vectors'].append(emb[i])
                if args.memory_size and len(mem['vectors'])>args.memory_size:
                    mem['vectors']=mem['vectors'][-args.memory_size:]
                mem['gt_history'].append(gt)
            mem['last_frame']=frame; mem['last_center']=centers[i].copy()
            rows.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                         'category_id':str(obs[i]['category_id']), 'gt_track_id':gt,
                         'predicted_id':pid,'decision':decision,'appearance_memory_updated':int(keep),
                         'top_two_candidate_gap':top_gap,'assigned_vs_other_gap':assigned_gap,
                         'correct_vs_memory_majority_before':correct,
                         'memory_pure_before':pure,'memory_purity_before':purity,
                         'selected_utility':selected})
    return rows


def stats(rows):
    matches=[r for r in rows if r['decision']=='match']
    n=len(matches)
    correct=sum(r['correct_vs_memory_majority_before'] for r in matches)
    pure=[r for r in matches if r['memory_pure_before']==1]
    contaminated=[r for r in matches if r['memory_pure_before']==0]
    def rate(xs): return sum(r['correct_vs_memory_majority_before'] for r in xs)/len(xs) if xs else None
    return {'pairwise':pairwise_metrics(rows),'matched_decisions':n,
            'appearance_updates':sum(r['appearance_memory_updated'] for r in matches),
            'association_accuracy_vs_memory_majority':correct/n if n else None,
            'pure_memory_matches':len(pure),'pure_memory_accuracy':rate(pure),
            'contaminated_memory_matches':len(contaminated),
            'contaminated_memory_accuracy':rate(contaminated),
            'mean_purity_before_match':float(np.mean([r['memory_purity_before'] for r in matches])) if matches else None}


def compare(a,b):
    A={int(r['observation_index']):r for r in a}; B={int(r['observation_index']):r for r in b}
    assert A.keys()==B.keys()
    changed=[i for i in A if A[i]['predicted_id']!=B[i]['predicted_id']]
    # Predicted ID strings are locally allocated and can shift after a new-ID event;
    # raw ID differences alone do not establish identity switches.
    shared_matches=[i for i in A if A[i]['decision']=='match' and B[i]['decision']=='match']
    accuracy_a=sum(A[i]['correct_vs_memory_majority_before'] for i in shared_matches)
    accuracy_b=sum(B[i]['correct_vs_memory_majority_before'] for i in shared_matches)
    return {'observations_with_different_numeric_predicted_ids':len(changed),
            'caution':'Numeric predicted IDs are not aligned between runs; do not interpret count as switches.',
            'shared_matched_observations':len(shared_matches),
            'baseline_correct_vs_own_memory_majority':accuracy_a,
            'gated_correct_vs_own_memory_majority':accuracy_b,
            'baseline_correct_gated_incorrect':sum(A[i]['correct_vs_memory_majority_before']==1 and B[i]['correct_vs_memory_majority_before']==0 for i in shared_matches),
            'baseline_incorrect_gated_correct':sum(A[i]['correct_vs_memory_majority_before']==0 and B[i]['correct_vs_memory_majority_before']==1 for i in shared_matches)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--stage12f-summary',type=Path,default=Path('results/burst/memory_selection/summary.json'))
    p.add_argument('--stage12f-assignments',type=Path,default=Path('results/burst/memory_selection/baseline_assignments.csv'))
    p.add_argument('--output',type=Path,default=Path('results/burst/memory_mechanism'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--spatial-threshold',type=float,default=.50)
    p.add_argument('--appearance-weight',type=float,default=.75)
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.)
    p.add_argument('--memory-size',type=int,default=0)
    args=p.parse_args()
    report=json.loads(args.stage12f_summary.read_text(encoding='utf-8'))
    if report.get('stage')!='12F': raise ValueError('Expected Stage 12F summary')
    params={x['held_out_sequence']:float(x['selected_parameter']) for x in report['calibration_folds'] if x['family']=='fixed'}
    obs=read_csv(args.input/'observations.csv')
    with np.load(args.input/'embeddings.npz') as z: emb=np.asarray(z['embeddings'],dtype=np.float32)
    if len(obs)!=len(emb): raise ValueError('Observation/embedding mismatch')
    emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    centers=load_geometry(obs,args.annotations,args.input/'normalized_mask_centers.npz')
    groups=defaultdict(list)
    for i,o in enumerate(obs): groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences!=['all']:
        groups={s:v for s,v in groups.items() if s in args.sequences or s.split('/',1)[1] in args.sequences}
    if not groups: raise ValueError('No matching sequences')
    args.output.mkdir(parents=True,exist_ok=True)
    original={}
    if args.stage12f_assignments.exists():
        original={(r['sequence'],int(r['observation_index'])):(r['predicted_id'],r['decision']) for r in read_csv(args.stage12f_assignments)}
    summary={'stage':'12G','protocol':'Replay baseline and held-out fixed gate from 12F; GT labels used only for post-hoc diagnostics.',
             'per_sequence':{},'limitations':['Memory purity measured using retained GT labels is oracle-only.',
             'Majority-GT association correctness is a proxy, not IDF1 or a global identity alignment.',
             'Contamination differences are observational and do not establish mediation or causal mechanism.',
             'Different predicted ID numbers between runs are not directly comparable.',
             'Only five sequences; oracle masks and categories.']}
    for seq in sorted(groups):
        if seq not in params: raise ValueError(f'Missing held-out fixed parameter for {seq}')
        base=replay(seq,groups[seq],obs,emb,centers,args,None)
        gated=replay(seq,groups[seq],obs,emb,centers,args,params[seq])
        if original:
            for r in base:
                key=(seq,int(r['observation_index']))
                if key not in original or original[key]!=(r['predicted_id'],r['decision']):
                    raise AssertionError(f'Baseline mismatch at {key}')
        for policy,rows in [('baseline',base),('fixed_gated',gated)]:
            path=args.output/policy/(seq.replace('/','__')+'.csv')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        s={'selected_margin':params[seq],'baseline':stats(base),'fixed_gated':stats(gated),
           'comparison':compare(base,gated)}
        expected=report['heldout']['fixed']['per_sequence'][seq]['metrics']['pairwise_f1']
        if abs(s['fixed_gated']['pairwise']['pairwise_f1']-expected)>1e-10:
            raise AssertionError(f'12F heldout replay mismatch {seq}')
        summary['per_sequence'][seq]=s
        print(seq,'F1 baseline',round(s['baseline']['pairwise']['pairwise_f1'],4),
              'gated',round(s['fixed_gated']['pairwise']['pairwise_f1'],4),
              'pure before baseline',s['baseline']['pure_memory_matches'],
              'gated',s['fixed_gated']['pure_memory_matches'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Saved',args.output/'summary.json')

if __name__=='__main__': main()
