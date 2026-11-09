"""TEMPO Stage 12I: causal, capacity-controlled appearance-memory replacement ablation.

Install: experiments/evaluate_burst_memory_coverage.py
Run: python -m experiments.evaluate_burst_memory_coverage --sequences all

Same Stage 12C Hungarian utility. Policies use the same capacity K but NOT identical
update counts or memory contents; replacement changes later assignment trajectories.
GT labels are used only for evaluation and purity diagnostics.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.evaluate_burst_temporal_association import load_geometry
from experiments.evaluate_burst_hungarian_association import read_csv, pairwise_metrics


def rand01(*parts):
    digest=hashlib.sha256('|'.join(map(str,parts)).encode()).digest()
    return int.from_bytes(digest[:8],'big')/2**64


def choose_memory(entries, capacity, policy, seq, idx, seed):
    if capacity==0 or len(entries)<=capacity:return entries
    if policy in ('fifo','gated_fifo'):
        return entries[-capacity:]
    if policy=='confidence':
        # Seed retained for anchor coverage; other entries ranked by pre-Hungarian
        # top-two candidate gap. No competitor => gap unavailable, assigned 0.
        anchor=entries[0]
        ranked=sorted(entries[1:],key=lambda e:(-e['confidence'],e['index']))
        return sorted([anchor]+ranked[:capacity-1],key=lambda e:e['index'])
    if policy=='diversity':
        # Greedy farthest-point subset in cosine distance, anchored to the seed.
        # Recompute on each insertion to avoid a permanently stale representation.
        selected=[0]
        while len(selected)<capacity:
            chosen=max((j for j in range(1,len(entries)) if j not in selected),
                       key=lambda j:(min(1-float(entries[j]['vector']@entries[k]['vector']) for k in selected),-entries[j]['index']))
            selected.append(chosen)
        return sorted((entries[j] for j in selected),key=lambda e:e['index'])
    if policy=='random':
        # Fixed-size deterministic priority sample, seed retained. This is an
        # order-independent reservoir sample of all candidate updates.
        anchor=entries[0]
        rest=sorted(entries[1:],key=lambda e:(rand01('priority',seed,seq,e['index']),e['index']))
        return sorted([anchor]+rest[:capacity-1],key=lambda e:e['index'])
    raise ValueError(policy)


def replay(seq, indices, obs, emb, centers, args, policy, capacity, margin, seed):
    by_frame=defaultdict(list)
    for i in indices:by_frame[int(obs[i]['frame_position'])].append(i)
    memories={}; rows=[]; next_id=0
    for frame in sorted(by_frame):
        ids=sorted(by_frame[frame]); pids=sorted(memories);n=len(ids);m=len(pids)
        utility=np.full((n,m+n),-1e6,dtype=float)
        for r in range(n):utility[r,m+r]=0.
        edges={}
        for r,i in enumerate(ids):
            for c,pid in enumerate(pids):
                mem=memories[pid]
                if mem['category']!=str(obs[i]['category_id']):continue
                app=max(float(e['vector']@emb[i]) for e in mem['entries'])
                dt=frame-mem['last_frame']
                if dt<=0:raise AssertionError('Noncausal frame order')
                dist=float(np.linalg.norm(centers[i]-mem['last_center']))
                st=math.exp(-.5*(dist/args.spatial_sigma)**2)*math.exp(-(dt-1)/args.temporal_tau)
                score=args.appearance_weight*(app-args.threshold)+(1-args.appearance_weight)*(st-args.spatial_threshold)
                edges[r,c]=score
                if score>=0:utility[r,c]=score+1e-10
        rr,cc=linear_sum_assignment(utility,maximize=True)
        assigned=dict(zip(rr.tolist(),cc.tolist()));pending=[]
        for r,i in enumerate(ids):
            c=assigned[r];matched=c<m and utility[r,c]>=0
            scores=sorted((s for (row,col),s in edges.items() if row==r),reverse=True)
            gap=scores[0]-scores[1] if len(scores)>1 else None
            gt=str(obs[i]['track_id'])
            if matched:
                pid=pids[c]; decision='match'
                keep=policy!='gated_fifo' or gap is None or gap>=margin
            else:
                pid=f'p{next_id:04d}';next_id+=1;decision='new';keep=True
            pending.append((i,pid,gt,decision,keep,gap))
        # Freeze memory during frame matching; commit only afterward.
        for i,pid,gt,decision,keep,gap in pending:
            if pid not in memories:
                memories[pid]={'category':str(obs[i]['category_id']),'entries':[],
                               'last_frame':frame,'last_center':centers[i].copy()}
            mem=memories[pid]
            if keep:
                mem['entries'].append({'vector':emb[i], 'index':i,'gt':gt,'frame':frame,
                                       'confidence':float(gap) if gap is not None else 0.})
                mem['entries']=choose_memory(mem['entries'],capacity,policy,seq,i,seed)
            mem['last_frame']=frame;mem['last_center']=centers[i].copy()
            entries=mem['entries'];cnt=Counter(e['gt'] for e in entries)
            vectors=np.stack([e['vector'] for e in entries]);sim=vectors@vectors.T
            tri=sim[np.triu_indices(len(entries),k=1)]
            rows.append({'sequence':seq,'observation_index':i,'frame_position':frame,
                         'category_id':str(obs[i]['category_id']),'gt_track_id':gt,
                         'predicted_id':pid,'decision':decision,'appearance_memory_updated':int(keep),
                         'memory_entries':len(entries),'memory_purity':max(cnt.values())/len(entries),
                         'memory_mean_cosine_distance':float(np.mean(1-tri)) if len(tri) else 0.,
                         'memory_frame_span':max(e['frame'] for e in entries)-min(e['frame'] for e in entries),
                         'top_two_candidate_gap':gap if gap is not None else ''})
    return rows


def summary(rows):
    matches=[r for r in rows if r['decision']=='match']
    return {'metrics':pairwise_metrics(rows),'matched_updates':sum(r['appearance_memory_updated'] for r in matches),
            'matched_decisions':len(matches),
            'mean_memory_size':float(np.mean([r['memory_entries'] for r in rows])),
            'mean_memory_purity':float(np.mean([r['memory_purity'] for r in rows])),
            'mean_memory_cosine_distance':float(np.mean([r['memory_mean_cosine_distance'] for r in rows])),
            'mean_memory_frame_span':float(np.mean([r['memory_frame_span'] for r in rows]))}


def micro(seq_results):
    tp=sum(v['metrics']['pairwise_tp'] for v in seq_results.values())
    fp=sum(v['metrics']['pairwise_fp'] for v in seq_results.values())
    fn=sum(v['metrics']['pairwise_fn'] for v in seq_results.values())
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--stage12f-summary',type=Path,default=Path('results/burst/memory_selection/summary.json'))
    p.add_argument('--output',type=Path,default=Path('results/burst/memory_coverage'))
    p.add_argument('--sequences',nargs='+',default=['all'])
    p.add_argument('--capacities',type=int,nargs='+',default=[5,10,20])
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--spatial-threshold',type=float,default=.50)
    p.add_argument('--appearance-weight',type=float,default=.75)
    p.add_argument('--spatial-sigma',type=float,default=.15)
    p.add_argument('--temporal-tau',type=float,default=10.)
    args=p.parse_args()
    if any(k<2 for k in args.capacities):raise ValueError('Capacity must be >=2')
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
    args.output.mkdir(parents=True,exist_ok=True)
    specs=[('baseline_unlimited','fifo',0,0),('gated_unlimited','gated_fifo',0,0)]
    for k in args.capacities:
        for policy in ('fifo','confidence','diversity','gated_fifo'):
            specs.append((f'{policy}_k{k}',policy,k,0))
        for seed in args.seeds:specs.append((f'random_k{k}_seed{seed}','random',k,seed))
    report={'stage':'12I','protocol':'Online causal Hungarian matching with fixed-capacity appearance memory. Oracle observations and geometry; GT labels only for post-hoc diagnostics.',
            'settings':{'capacities':args.capacities,'random_seeds':args.seeds,'fixed_margins':margins},
            'limitations':['Memory capacity is equal within each K, but update counts and realized contents are not exactly matched.',
                           'Confidence is pre-Hungarian top-two candidate gap; single-candidate gaps are assigned zero ranking score.',
                           'Diversity uses greedy farthest-point cosine sampling with seed anchoring; selection is a heuristic.',
                           'Seed anchoring also applies to confidence and random replacement but not FIFO.',
                           'Memory statistics are averaged per observation after update, not weighted by identity or frame.',
                           'Only five oracle-observation sequences; pairwise F1 is not IDF1/HOTA.'],
            'policies':{name:{'per_sequence':{}} for name,_,_,_ in specs}}
    for seq in sorted(groups):
        if seq not in margins:raise ValueError(f'Missing Stage 12F margin for {seq}')
        for name,policy,k,seed in specs:
            rows=replay(seq,groups[seq],obs,emb,centers,args,policy,k,margins[seq],seed)
            s=summary(rows);report['policies'][name]['per_sequence'][seq]=s
            if name in ('baseline_unlimited','gated_unlimited'):
                reference=(f['baseline']['per_sequence'][seq] if name=='baseline_unlimited'
                           else f['heldout']['fixed']['per_sequence'][seq])
                if abs(s['metrics']['pairwise_f1']-reference['metrics']['pairwise_f1'])>1e-10:
                    raise AssertionError(f'Replay mismatch for {name} {seq}')
                path=args.output/'assignments'/name/(seq.replace('/','__')+'.csv')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('w',newline='',encoding='utf-8') as fh:
                    writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('Completed',seq,flush=True)
    for name in report['policies']:
        report['policies'][name]['micro_pairwise']=micro(report['policies'][name]['per_sequence'])
    report['random_aggregates']={}
    for k in args.capacities:
        values=[report['policies'][f'random_k{k}_seed{seed}']['micro_pairwise']['f1'] for seed in args.seeds]
        report['random_aggregates'][f'k{k}']={'mean_f1':float(np.mean(values)),
          'std_f1':float(np.std(values,ddof=1)) if len(values)>1 else 0.,'f1_by_seed':dict(zip(map(str,args.seeds),values))}
    path=args.output/'summary.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',path)
    for name in report['policies']:
        if '_seed' not in name:print(name,round(report['policies'][name]['micro_pairwise']['f1'],5))

if __name__=='__main__':main()
