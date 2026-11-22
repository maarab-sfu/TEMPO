"""M4.6 experimental object-memory sidecar and hybrid frame selection.

No learned re-identification, no claims of object identity correctness.
The existing production frame retriever is deliberately left unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def load_tracks(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('schema_version') != 1 or not isinstance(obj.get('tracks'),list):
        raise ValueError('Expected M4.5 tracks JSON schema v1')
    return obj['tracks']


def build_candidates(tracks, min_observations=3, min_score=0.35, labels=None):
    """Each track contributes representative observations, not all redundant frames."""
    out=[]
    for tr in tracks:
        if labels is not None and tr['label'] not in labels: continue
        obs=sorted((o for o in tr['observations'] if float(o.get('score',0))>=min_score),key=lambda o:o['timestamp'])
        if len(obs)<min_observations: continue
        # First, median, last, and strongest confidence; deduplicated.
        picks={0,len(obs)//2,len(obs)-1,max(range(len(obs)),key=lambda i:obs[i]['score'])}
        for i in sorted(picks):
            o=obs[i]
            out.append({'track_id':tr['track_id'],'label':tr['label'],
                        'timestamp':float(o['timestamp']),'frame_index':int(o['frame_index']),
                        'box':[float(x) for x in o['box']], 'detector_score':float(o['score'])})
    return out


def save_index(candidates, embeddings, out_path, video_name):
    emb=np.asarray(embeddings,dtype=np.float32)
    if emb.ndim!=2 or emb.shape[0]!=len(candidates): raise ValueError('Embedding count mismatch')
    norms=np.linalg.norm(emb,axis=1,keepdims=True)
    emb=emb/np.maximum(norms,1e-12)
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path, embeddings=emb, metadata=json.dumps({'schema_version':1,'video':video_name,'candidates':candidates}))


def search_index(index_path, text_embedding, top_k=8, label=None):
    with np.load(index_path,allow_pickle=False) as z:
        embeddings=z['embeddings'];meta=json.loads(str(z['metadata']))
    if not len(embeddings): return []
    q=np.asarray(text_embedding,dtype=np.float32).reshape(-1)
    if q.shape[0]!=embeddings.shape[1]:raise ValueError('Embedding dimension mismatch')
    q=q/max(float(np.linalg.norm(q)),1e-12)
    scores=embeddings@q
    ranked=sorted((dict(c,object_similarity=float(scores[i])) for i,c in enumerate(meta['candidates']) if label is None or c['label']==label),key=lambda c:c['object_similarity'],reverse=True)
    return ranked[:top_k]


def relevant_object_labels(question, available_labels):
    """Conservative keyword routing. No forced class for generic questions."""
    import re
    words=set(re.findall(r"[a-z]+", question.lower()))
    labels=set(available_labels)
    targets=set()
    # Only use a label if the question explicitly mentions the object category.
    for label in labels:
        tokens=set(re.findall(r"[a-z]+", label.lower()))
        if tokens and tokens.issubset(words):
            targets.add(label)
    if 'toy' in words:
        targets.update(label for label in labels if 'toy' in label.lower())
    if 'child' in words or 'kid' in words or 'boy' in words or 'girl' in words:
        targets.update(label for label in labels if label.lower() in ('person','child'))
    return targets


def rank_object_hits(hits, question):
    """Prefer explicitly relevant labels, then CLIP similarity; diversify tracks."""
    if not hits:
        return []
    relevant=relevant_object_labels(question, [h['label'] for h in hits])
    if not relevant:
        return []  # Conservative fallback; no arbitrary CLIP-only replacements.
    eligible=[h for h in hits if h['label'] in relevant]
    eligible.sort(key=lambda h: h['object_similarity'],reverse=True)
    # One representative per track first; allow other observations later.
    first, remaining, seen=[],[],set()
    for h in eligible:
        if h['track_id'] not in seen:
            first.append(h)
            seen.add(h['track_id'])
        else:
            remaining.append(h)
    return first+remaining


def hybrid_select(baseline_frames, object_hits, frame_catalog, max_frames=8,
                  object_slots=2, max_time_delta=1.0, question=None):
    """Experimental hybrid selection, preserving all semantic anchors.

    Adds new frame evidence only when a candidate maps to a different indexed
    frame. When full, replaces the latest temporal context, never an anchor.
    No claims of semantic grounding are implied by a detection or CLIP score.
    """
    if max_frames < 1:
        raise ValueError('max_frames must be positive')
    baseline=[dict(f) for f in baseline_frames[:max_frames]]
    if not frame_catalog or not object_hits or object_slots <= 0:
        return sorted(baseline,key=lambda x:x['timestamp'])
    ranked=rank_object_hits(object_hits,question) if question is not None else list(object_hits)
    if not ranked:
        return sorted(baseline,key=lambda x:x['timestamp'])
    selected={str(f['image_path']):f for f in baseline}
    original=set(selected)
    additions=0
    for hit in ranked:
        if additions >= object_slots:
            break
        nearest=min(frame_catalog,key=lambda f:abs(float(f['timestamp'])-float(hit['timestamp'])))
        delta=abs(float(nearest['timestamp'])-float(hit['timestamp']))
        key=str(nearest['image_path'])
        if delta > max_time_delta or key in selected or key in original:
            continue
        if len(selected) >= max_frames:
            contexts=[k for k,f in selected.items()
                      if f.get('evidence_role') not in ('anchor','semantic_anchor','object_memory')]
            if not contexts:
                break
            # Keep object additions and all semantic anchors.
            drop=max(contexts,key=lambda k:float(selected[k]['timestamp']))
            del selected[drop]
        selected[key]=dict(nearest,evidence_role='object_memory',anchor_rank=None,
                           similarity=None,object_track_id=hit['track_id'],
                           object_label=hit['label'],object_similarity=hit['object_similarity'],
                           object_observation_timestamp=float(hit['timestamp']))
        additions+=1
    return sorted(selected.values(),key=lambda f:f['timestamp'])
