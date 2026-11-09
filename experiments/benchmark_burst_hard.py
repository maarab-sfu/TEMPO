"""BURST hard-negative CLIP evaluation, using GT segmentation masks.
Run: python -m experiments.benchmark_burst_hard
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from experiments.burst_ground_truth import ROOT, DEFAULT_JSON, decode_mask
from experiments.benchmark_burst_identity import crop_masked
from src.embeddings.clip_encoder import CLIPEncoder

SELECTED = {
    'LaSOT': ['volleyball-13', 'person-17', 'sheep-19'],
    'HACS': ['Grooming_horse_v_ANnFXWVJ_x0_scene_0_0-5143',
             'Playing_badminton_v_PnA8TaQeCSk_scene_0_0-1002'],
}
GAPS = [(1,3),(4,10),(11,30),(31,None)]  # annotated frame positions, NOT seconds


def metrics(rows):
    y = np.asarray([r['same_identity'] for r in rows], dtype=int)
    s = np.asarray([r['similarity'] for r in rows], dtype=float)
    out = dict(pairs=len(rows), positives=int(y.sum()), negatives=int(len(y)-y.sum()),
               mean_positive=float(s[y==1].mean()) if (y==1).any() else None,
               mean_negative=float(s[y==0].mean()) if (y==0).any() else None,
               roc_auc=None, average_precision=None)
    if len(np.unique(y))==2:
        out['roc_auc']=float(roc_auc_score(y,s))
        out['average_precision']=float(average_precision_score(y,s))
    return out


def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--output',type=Path,default=ROOT/'results/burst/identity_hard')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    crops=args.output/'crops'; crops.mkdir(exist_ok=True)
    data=json.loads(DEFAULT_JSON.read_text(encoding='utf-8'))
    cats={int(x['id']):x['name'] for x in data['categories']}
    seqs=[s for s in data['sequences'] if s['dataset'] in SELECTED and s['seq_name'] in SELECTED[s['dataset']]]
    if len(seqs)!=5: raise RuntimeError(f'Expected five sequences, found {len(seqs)}')
    obs=[]; paths=[]
    for seq in seqs:
        dataset, sid=seq['dataset'],seq['seq_name']
        h,w=int(seq['height']),int(seq['width'])
        frames=seq['annotated_image_paths']
        masks=seq['segmentations']
        if len(frames)!=len(masks): raise ValueError(f'{sid}: annotation lengths differ')
        if args.max_frames: frames=frames[:args.max_frames]; masks=masks[:args.max_frames]
        for t,(raw,objects) in enumerate(zip(frames,masks)):
            filename=Path(raw.replace('\\','/')).name
            frame=ROOT/'data/raw/burst/frames/val'/dataset/sid/filename
            if not frame.is_file(): raise FileNotFoundError(frame)
            with Image.open(frame) as im: image=im.convert('RGB')
            if image.size!=(w,h): raise ValueError(f'{frame}: dimensions {image.size} != {(w,h)}')
            for track,ann in objects.items():
                mask=decode_mask(ann,h,w)
                if not mask.any(): continue
                cat=int(seq['track_category_ids'][str(track)])
                path=crops/f'{dataset}_{sid}_{t:04d}_{track}.png'
                if not path.is_file(): crop_masked(image,mask).save(path)
                obs.append(dict(dataset=dataset,sequence=sid,frame_position=t,
                                frame_name=filename,track_id=int(track),category_id=cat,
                                category=cats.get(cat,str(cat)),crop_path=str(path)))
                paths.append(str(path))
        print(f'{dataset}/{sid}: {sum(o["sequence"]==sid for o in obs)} observations',flush=True)
    if not obs: raise RuntimeError('No object observations')
    encoder=CLIPEncoder()
    emb=np.asarray(encoder.encode_images(paths,batch_size=args.batch_size),dtype=np.float32)
    emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    np.savez_compressed(args.output/'embeddings.npz',embeddings=emb)
    write_csv(args.output/'observations.csv',obs,list(obs[0]))
    byseq=defaultdict(list)
    for i,o in enumerate(obs): byseq[(o['dataset'],o['sequence'])].append(i)
    pairs=[]; retrieval=[]
    for (ds,sid),indices in byseq.items():
        sim=emb[indices]@emb[indices].T
        for a in range(len(indices)):
            x=obs[indices[a]]
            for b in range(a+1,len(indices)):
                y=obs[indices[b]]
                gap=abs(x['frame_position']-y['frame_position'])
                if gap==0: continue # cross-frame evaluation only
                pairs.append(dict(dataset=ds,sequence=sid,i=indices[a],j=indices[b],
                    same_identity=int(x['track_id']==y['track_id']),
                    same_category=int(x['category_id']==y['category_id']),
                    similarity=float(sim[a,b]),annotated_frame_gap=gap))
        # Retrieval: query frame must have >=1 matching identity in a different frame.
        # Candidate gallery is restricted to SAME category to prevent trivial class matching.
        for a,idx in enumerate(indices):
            x=obs[idx]
            candidate=[b for b,j in enumerate(indices) if b!=a and
                       obs[j]['frame_position']!=x['frame_position'] and
                       obs[j]['category_id']==x['category_id']]
            positives=[b for b in candidate if obs[indices[b]]['track_id']==x['track_id']]
            negatives=[b for b in candidate if obs[indices[b]]['track_id']!=x['track_id']]
            if not positives or not negatives: continue # informative queries only
            ordered=sorted(candidate,key=lambda b:float(sim[a,b]),reverse=True)
            rank=next(r for r,b in enumerate(ordered,1) if obs[indices[b]]['track_id']==x['track_id'])
            retrieval.append(dict(dataset=ds,sequence=sid,query_index=idx,track_id=x['track_id'],
                                  gallery_size=len(candidate),negative_candidates=len(negatives),
                                  first_correct_rank=rank,hit_at_1=int(rank<=1),
                                  hit_at_5=int(rank<=5),reciprocal_rank=1/rank))
    if not pairs: raise RuntimeError('No cross-frame pairs')
    write_csv(args.output/'pairs.csv',pairs,list(pairs[0]))
    write_csv(args.output/'retrieval.csv',retrieval,list(retrieval[0]) if retrieval else
              ['dataset','sequence','query_index','track_id','gallery_size','negative_candidates',
               'first_correct_rank','hit_at_1','hit_at_5','reciprocal_rank'])
    hard=[r for r in pairs if r['same_category']]
    def retrieve_stats(rows):
        return dict(queries=len(rows),recall_at_1=float(np.mean([r['hit_at_1'] for r in rows])) if rows else None,
                    recall_at_5=float(np.mean([r['hit_at_5'] for r in rows])) if rows else None,
                    mrr=float(np.mean([r['reciprocal_rank'] for r in rows])) if rows else None)
    per_seq={f'{ds}/{sid}':dict(all_pairs=metrics([r for r in pairs if r['dataset']==ds and r['sequence']==sid]),
                                same_category=metrics([r for r in hard if r['dataset']==ds and r['sequence']==sid]),
                                retrieval=retrieve_stats([r for r in retrieval if r['dataset']==ds and r['sequence']==sid]))
             for ds,sid in byseq}
    gap_stats={f'{lo}-{hi if hi is not None else "plus"}':metrics([r for r in hard if
                 r['annotated_frame_gap']>=lo and (hi is None or r['annotated_frame_gap']<=hi)])
               for lo,hi in GAPS}
    summary=dict(observations=len(obs),sequences=len(byseq),all_pairs=metrics(pairs),
                 same_category_pairs=metrics(hard),retrieval=retrieve_stats(retrieval),
                 temporal_gap_same_category=gap_stats,per_sequence=per_seq,
                 caveats=['Temporal gap bins count annotated-frame positions, not seconds.',
                          'BURST annotations are sparse and not exhaustive.',
                          'Pair scores are correlated; no confidence intervals are implied.',
                          'Retrieval gallery is same-category and cross-frame, and excludes queries without both positive and negative candidates.',
                          'GT masks exclude detector and tracker errors.',
                          'Same-category negatives need not be visually similar; no threshold tuned.',
                          'Do not compare pair-weighted AP across datasets without accounting for class prevalence.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print('Saved to:',args.output)

if __name__=='__main__': main()
