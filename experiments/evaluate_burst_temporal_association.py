"""Stage 12C: causal spatial/appearance Hungarian association on BURST GT crops.

Run from project root:
  python -m experiments.evaluate_burst_temporal_association --sequences all
  python -m experiments.evaluate_burst_temporal_association --sequences all --mode appearance
  python -m experiments.evaluate_burst_temporal_association --sequences all --mode spatial
  python -m experiments.evaluate_burst_temporal_association --sequences all --mode combined --appearance-weight 0.75

Requires numpy, scipy, pycocotools; existing identity_hard embeddings and BURST annotations.
Ground-truth track IDs are used ONLY to locate segmentation masks and evaluate outputs;
never for association, initialization, or motion prediction. Geometry is GT-derived.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.evaluate_burst_hungarian_association import pairwise_metrics, read_csv, write_csv
from experiments.burst_ground_truth import decode_mask


def load_geometry(obs, annotation_path, cache_path):
    """Compute normalized mask bounding-box centers, keyed by observation index.

    The cache is tied to observation ordering and annotation file size/mtime.
    Masks are decoded only for observations in the input CSV.
    """
    signature = {'observation_count': len(obs), 'observations_sha256': hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest(), 'annotation_size': annotation_path.stat().st_size,
                 'annotation_mtime_ns': annotation_path.stat().st_mtime_ns,
                 'first': [obs[0][k] for k in ('dataset', 'sequence', 'frame_position', 'track_id')],
                 'last': [obs[-1][k] for k in ('dataset', 'sequence', 'frame_position', 'track_id')]}
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached['signature'].item()) == json.dumps(signature, sort_keys=True):
                centers = cached['centers'].astype(np.float32)
                if centers.shape == (len(obs), 2):
                    return centers
    with annotation_path.open(encoding='utf-8') as f:
        data = json.load(f)
    needed = defaultdict(list)
    for i, o in enumerate(obs):
        needed[(o['dataset'], o['sequence'])].append(i)
    centers = np.full((len(obs), 2), np.nan, dtype=np.float32)
    found = set()
    for seq in data['sequences']:
        key = (seq['dataset'], seq['seq_name'])
        if key not in needed:
            continue
        found.add(key)
        h, w = int(seq['height']), int(seq['width'])
        for i in needed[key]:
            o = obs[i]
            frame = int(o['frame_position'])
            objects = seq['segmentations'][frame]
            track = str(o['track_id'])
            if track not in objects:
                raise ValueError(f'Missing GT mask for {key} frame={frame} track={track}')
            # Ground-truth track ID is used only for selecting the corresponding
            # mask to extract observation geometry, not for matching observations.
            mask = decode_mask(objects[track], h, w)
            yy, xx = np.nonzero(mask)
            if not len(xx):
                raise ValueError(f'Empty mask for {key} frame={frame} track={track}')
            centers[i] = ((float(xx.min()) + float(xx.max()) + 1) / (2 * w),
                          (float(yy.min()) + float(yy.max()) + 1) / (2 * h))
    missing = set(needed) - found
    if missing or not np.isfinite(centers).all():
        raise ValueError(f'Geometry incomplete; missing sequences: {missing}')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, centers=centers, signature=json.dumps(signature, sort_keys=True))
    return centers


def association(seq, indices, obs, emb, centers, args):
    by_frame = defaultdict(list)
    for i in indices:
        by_frame[int(obs[i]['frame_position'])].append(i)
    memories = {}
    next_id = 0
    rows = []
    for frame in sorted(by_frame):
        frame_ids = sorted(by_frame[frame]); pred_ids = sorted(memories)
        n, m = len(frame_ids), len(pred_ids)
        utility = np.full((n, m + n), -1e6, dtype=np.float64)
        for r in range(n):
            utility[r, m + r] = 0.0
        edges = {}
        for r, i in enumerate(frame_ids):
            category = str(obs[i]['category_id'])
            for c, pid in enumerate(pred_ids):
                mem = memories[pid]
                if category != mem['category']:
                    continue
                appearance = float(np.max(np.asarray(mem['vectors']) @ emb[i]))
                gap = frame - mem['last_frame']  # annotated-frame positions, not seconds
                if gap <= 0:
                    raise AssertionError('Noncausal memory access')
                distance = float(np.linalg.norm(centers[i] - mem['last_center']))
                spatial = math.exp(-0.5 * (distance / args.spatial_sigma) ** 2)
                recency = math.exp(-(gap - 1) / args.temporal_tau)
                spatial_temporal = spatial * recency
                if args.mode == 'appearance':
                    advantage = appearance - args.threshold
                elif args.mode == 'spatial':
                    advantage = spatial_temporal - args.spatial_threshold
                else:
                    # Center each evidence term at its own acceptance threshold.
                    # lambda=1 exactly reproduces Stage 12B, including matching.
                    advantage = (args.appearance_weight * (appearance - args.threshold)
                                 + (1 - args.appearance_weight) *
                                 (spatial_temporal - args.spatial_threshold))
                edges[(r, c)] = (appearance, spatial, recency, spatial_temporal, advantage, gap, distance)
                if advantage >= 0:
                    utility[r, c] = advantage + 1e-10
        rr, cc = linear_sum_assignment(utility, maximize=True)
        assignment = dict(zip(rr.tolist(), cc.tolist()))
        pending = []
        for r, i in enumerate(frame_ids):
            c = assignment[r]
            if c < m and utility[r, c] >= 0:
                pid = pred_ids[c]
                decision = 'match'
                a, s, t, st, advantage, gap, distance = edges[(r, c)]
            else:
                pid = f'p{next_id:04d}'; next_id += 1
                decision = 'new'
                a = s = t = st = advantage = gap = distance = None
            pending.append((i, pid, decision, a, s, t, st, advantage, gap, distance))
        # Commit all updates only after frame-level assignment is complete.
        for i, pid, decision, a, s, t, st, advantage, gap, distance in pending:
            if pid not in memories:
                memories[pid] = {'category': str(obs[i]['category_id']), 'vectors': [],
                                 'last_frame': frame, 'last_center': centers[i].copy()}
            mem = memories[pid]
            mem['vectors'].append(emb[i])
            if args.memory_size and len(mem['vectors']) > args.memory_size:
                mem['vectors'] = mem['vectors'][-args.memory_size:]
            mem['last_frame'] = frame
            mem['last_center'] = centers[i].copy()
            rows.append({'sequence': seq, 'observation_index': i,
                         'frame_position': frame, 'category_id': str(obs[i]['category_id']),
                         'gt_track_id': str(obs[i]['track_id']), 'predicted_id': pid,
                         'decision': decision, 'center_x': float(centers[i, 0]),
                         'center_y': float(centers[i, 1]),
                         'selected_appearance': a, 'selected_spatial': s,
                         'selected_recency': t, 'selected_spatiotemporal': st,
                         'selected_advantage': advantage, 'matched_frame_gap': gap,
                         'matched_center_distance': distance})
    return rows


def summarize(rows, per_sequence):
    tp = sum(m['pairwise_tp'] for m in per_sequence.values())
    fp = sum(m['pairwise_fp'] for m in per_sequence.values())
    fn = sum(m['pairwise_fn'] for m in per_sequence.values())
    fragments = defaultdict(set)
    for r in rows:
        fragments[(r['sequence'], r['gt_track_id'])].add(r['predicted_id'])
    return {'observations': len(rows), 'sequences': len(per_sequence),
            'pairwise_tp': tp, 'pairwise_fp': fp, 'pairwise_fn': fn,
            'pairwise_precision': tp/(tp+fp) if tp+fp else None,
            'pairwise_recall': tp/(tp+fn) if tp+fn else None,
            'pairwise_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'mean_predicted_fragments_per_gt': float(np.mean([len(x) for x in fragments.values()])),
            'same_frame_identity_collisions': sum(m['same_frame_identity_collisions'] for m in per_sequence.values()),
            'predicted_identities': sum(m['predicted_identities'] for m in per_sequence.values()),
            'gt_identities': sum(m['gt_identities'] for m in per_sequence.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=Path('results/burst/identity_hard'))
    p.add_argument('--annotations', type=Path, default=Path('data/annotations/burst/val/all_classes.json'))
    p.add_argument('--output', type=Path, default=Path('results/burst/temporal_association'))
    p.add_argument('--sequences', nargs='+', default=['person-17', 'sheep-19'])
    p.add_argument('--mode', choices=['appearance', 'spatial', 'combined'], default='combined')
    p.add_argument('--threshold', type=float, default=0.90, help='Appearance acceptance threshold')
    p.add_argument('--spatial-threshold', type=float, default=0.50, help='Spatial-temporal acceptance threshold')
    p.add_argument('--appearance-weight', type=float, default=0.75, help='Combined appearance evidence weight')
    p.add_argument('--spatial-sigma', type=float, default=0.15, help='Gaussian normalized center-distance scale')
    p.add_argument('--temporal-tau', type=float, default=10.0, help='Exponential recency scale in annotated frames')
    p.add_argument('--memory-size', type=int, default=0)
    args = p.parse_args()
    if not -1 <= args.threshold <= 1 or not 0 <= args.spatial_threshold <= 1:
        p.error('Invalid threshold')
    if not 0 <= args.appearance_weight <= 1 or args.spatial_sigma <= 0 or args.temporal_tau <= 0 or args.memory_size < 0:
        p.error('Invalid weight, spatial sigma, temporal tau, or memory size')
    obs = read_csv(args.input / 'observations.csv')
    required = {'dataset', 'sequence', 'frame_position', 'category_id', 'track_id'}
    if not obs or not required.issubset(obs[0]):
        raise ValueError(f'Missing observation columns: {required}')
    with np.load(args.input / 'embeddings.npz') as z:
        emb = np.asarray(z['embeddings'], dtype=np.float32)
    if len(obs) != len(emb):
        raise ValueError('Observation/embedding count mismatch')
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    groups = defaultdict(list)
    for i, o in enumerate(obs):
        groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences != ['all']:
        groups = {k: v for k, v in groups.items() if k in args.sequences or k.split('/', 1)[1] in args.sequences}
    if not groups:
        raise ValueError('No matching sequences')
    # Geometry cache is shared between modes and output directories.
    cache_path = args.input / 'normalized_mask_centers.npz'
    centers = load_geometry(obs, args.annotations, cache_path)
    all_rows = []; per_sequence = {}
    for seq, ids in sorted(groups.items()):
        rows = association(seq, ids, obs, emb, centers, args)
        all_rows.extend(rows)
        per_sequence[seq] = pairwise_metrics(rows)
        m = per_sequence[seq]
        print(f"{seq}: F1={m['pairwise_f1']:.4f}, P={m['pairwise_precision']:.4f}, "
              f"R={m['pairwise_recall']:.4f}, IDs={m['predicted_identities']}, "
              f"collisions={m['same_frame_identity_collisions']}")
    assert all(x['same_frame_identity_collisions'] == 0 for x in per_sequence.values())
    pooled = summarize(all_rows, per_sequence)
    out = args.output / args.mode
    out.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0])
    write_csv(out / 'assignments.csv', all_rows, fields)
    summary = {'protocol': 'Causal frame-level Hungarian matching; GT mask centers; GT IDs for evaluation only.',
               'mode': args.mode, 'appearance_threshold': args.threshold,
               'spatial_threshold': args.spatial_threshold, 'appearance_weight': args.appearance_weight,
               'spatial_sigma': args.spatial_sigma, 'temporal_tau_annotated_frames': args.temporal_tau,
               'memory_size': args.memory_size,
               'scoring': 'appearance advantage=cosine-appearance_threshold; spatial advantage=exp(-distance^2/(2 sigma^2))*exp(-(gap-1)/tau)-spatial_threshold; combined is weighted sum of advantages; Hungarian with unmatched dummies',
               'pooled': pooled, 'per_sequence': per_sequence,
               'limitations': ['Oracle segmentation geometry and crops; not detector-based tracking.',
                               'GT IDs only select each observation mask and evaluate predicted assignments.',
                               'Spatial centers use image-normalized coordinates, not camera-motion compensation.',
                               'Temporal gaps count annotated frames, not seconds.',
                               'Hyperparameters are exploratory; no independent calibration.',
                               'Pairwise clustering F1 is not MOT IDF1 or HOTA.',
                               'Observations within videos are correlated; no confidence intervals.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nPooled:', json.dumps(pooled, indent=2))
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
