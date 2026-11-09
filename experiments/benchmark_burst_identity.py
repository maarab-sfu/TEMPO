"""BURST Charades ground-truth masked-crop CLIP identity baseline.

Run: python -m experiments.benchmark_burst_identity
Optional: --sequence DHVLP --max-frames 20
Requires project CLIPEncoder and experiments.burst_ground_truth.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.burst_ground_truth import (ROOT, DEFAULT_JSON, DEFAULT_FRAMES,
    load_sequences, decode_mask, find_frame)
from src.embeddings.clip_encoder import CLIPEncoder


def crop_masked(image: Image.Image, mask: np.ndarray, padding: float = .10) -> Image.Image:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('Empty mask')
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x, pad_y = round((x1-x0)*padding), round((y1-y0)*padding)
    x0, y0 = max(0, x0-pad_x), max(0, y0-pad_y)
    x1, y1 = min(image.width, x1+pad_x), min(image.height, y1+pad_y)
    arr = np.asarray(image.convert('RGB')).copy()
    # Neutralize background while retaining a small spatial context margin.
    bg = np.full_like(arr, 127)
    bg[mask] = arr[mask]
    return Image.fromarray(bg[y0:y1, x0:x1])


def summarize_pairs(rows):
    pos = np.array([r['similarity'] for r in rows if r['same_identity'] == 1])
    neg = np.array([r['similarity'] for r in rows if r['same_identity'] == 0])
    result = {'positive_pairs': int(len(pos)), 'negative_pairs': int(len(neg)),
              'mean_positive_similarity': float(pos.mean()) if len(pos) else None,
              'mean_negative_similarity': float(neg.mean()) if len(neg) else None}
    if len(pos) and len(neg):
        from sklearn.metrics import roc_auc_score, average_precision_score
        y = np.array([r['same_identity'] for r in rows])
        s = np.array([r['similarity'] for r in rows])
        result.update(roc_auc=float(roc_auc_score(y, s)),
                      average_precision=float(average_precision_score(y, s)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sequence', default=None)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--min-gap', type=int, default=1,
                   help='Minimum difference in annotated frame positions for positive pairs')
    p.add_argument('--output', type=Path, default=ROOT/'results/burst/identity_baseline')
    args = p.parse_args()
    if args.min_gap < 1:
        p.error('--min-gap must be >= 1')
    cats, sequences = load_sequences(DEFAULT_JSON)
    sequences = [s for s in sequences if (DEFAULT_FRAMES/s['seq_name']).is_dir()]
    if args.sequence:
        sequences = [s for s in sequences if s['seq_name'] == args.sequence]
    if not sequences:
        p.error('No downloaded sequences matched')
    args.output.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output/'crops'
    crops_dir.mkdir(exist_ok=True)
    observations = []
    paths = []
    for seq in sequences:
        sid = seq['seq_name']
        h, w = int(seq['height']), int(seq['width'])
        limit = args.max_frames or len(seq['annotated_image_paths'])
        for t, (img_rel, masks) in enumerate(zip(seq['annotated_image_paths'][:limit], seq['segmentations'][:limit])):
            path = find_frame(DEFAULT_FRAMES, sid, img_rel)
            with Image.open(path) as im:
                image = im.convert('RGB')
            if image.size != (w, h):
                raise ValueError(f'Image shape mismatch: {path}')
            for track, ann in masks.items():
                mask = decode_mask(ann, h, w)
                if not mask.any():
                    continue
                track = int(track)
                cat = int(seq['track_category_ids'][str(track)])
                out = crops_dir/f'{sid}_{t:04d}_{track}.png'
                crop_masked(image, mask).save(out)
                paths.append(str(out))
                observations.append(dict(sequence=sid, frame_position=t,
                    frame_name=path.name, track_id=track, category_id=cat,
                    category=cats.get(cat, str(cat)), crop_path=str(out)))
        print(f'{sid}: {sum(o["sequence"] == sid for o in observations)} observations', flush=True)
    if not observations:
        raise RuntimeError('No valid masks found')
    encoder = CLIPEncoder()
    embeddings = np.asarray(encoder.encode_images(paths, batch_size=args.batch_size), dtype=np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    np.savez_compressed(args.output/'embeddings.npz', embeddings=embeddings)
    with (args.output/'observations.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(observations[0])); writer.writeheader(); writer.writerows(observations)
    # Only compare within sequences. Different videos have unrelated identity namespaces.
    # Same-frame distinct IDs are useful hard negatives, but reported separately.
    by_seq = defaultdict(list)
    for i, o in enumerate(observations): by_seq[o['sequence']].append(i)
    pairs = []
    for sid, indices in by_seq.items():
        mat = embeddings[indices] @ embeddings[indices].T
        for a in range(len(indices)):
            for b in range(a+1, len(indices)):
                x, y = observations[indices[a]], observations[indices[b]]
                gap = abs(x['frame_position']-y['frame_position'])
                same = int(x['track_id'] == y['track_id'])
                if same and gap < args.min_gap: continue
                pairs.append(dict(sequence=sid, i=indices[a], j=indices[b],
                    same_identity=same, similarity=float(mat[a,b]),
                    annotated_frame_gap=gap, same_frame=int(gap==0),
                    same_category=int(x['category_id']==y['category_id'])))
    with (args.output/'pairs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    per_video = {sid: summarize_pairs([r for r in pairs if r['sequence']==sid]) for sid in by_seq}
    overall = summarize_pairs(pairs)
    hard = summarize_pairs([r for r in pairs if r['same_identity'] or (r['same_category'] and not r['same_identity'])])
    summary = dict(observations=len(observations), sequences=len(by_seq), overall_pair_weighted=overall,
        same_category_negatives=hard, per_sequence=per_video,
        caveats=['Pair-weighted scores are descriptive, not independent-sample confidence intervals.',
                 'Different annotated IDs within a sequence are negatives; unannotated objects are not labeled negatives.',
                 'Only GT-mask crops are evaluated; detector and tracker errors are excluded.',
                 'No threshold has been tuned; this is a similarity-ranking baseline.'])
    with (args.output/'summary.json').open('w', encoding='utf-8') as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print('Saved to:', args.output)

if __name__ == '__main__': main()
