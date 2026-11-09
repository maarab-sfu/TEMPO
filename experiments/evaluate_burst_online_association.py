"""Stage 12A: causal, appearance-only online association on BURST GT observations.

Uses GT segmentation-derived CLIP crops as detections, but NEVER uses GT identity
in assignment. GT identity is read only for evaluation. Not a full MOT benchmark.

Run from project root:
  python -m experiments.evaluate_burst_online_association --sequences all
  python -m experiments.evaluate_burst_online_association --sequences person-17 sheep-19
  python -m experiments.evaluate_burst_online_association --sequences all --threshold 0.90
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from math import comb
import numpy as np


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pairwise_metrics(rows):
    # Exact contingency-table pair counts; avoids enumerating O(N^2) pairs.
    true_counts = Counter(r['gt_track_id'] for r in rows)
    pred_counts = Counter(r['predicted_id'] for r in rows)
    joint = Counter((r['gt_track_id'], r['predicted_id']) for r in rows)
    tp = sum(comb(n, 2) for n in joint.values() if n >= 2)
    predicted_pairs = sum(comb(n, 2) for n in pred_counts.values() if n >= 2)
    true_pairs = sum(comb(n, 2) for n in true_counts.values() if n >= 2)
    fp, fn = predicted_pairs - tp, true_pairs - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    fragments = defaultdict(set)
    for r in rows:
        fragments[r['gt_track_id']].add(r['predicted_id'])
    # Count different GT objects sharing a predicted identity in the same frame.
    same_frame_collisions = 0
    frame_groups = defaultdict(list)
    for r in rows:
        frame_groups[r['frame_position']].append(r)
    for frame_rows in frame_groups.values():
        by_pred = defaultdict(set)
        for r in frame_rows:
            by_pred[r['predicted_id']].add(r['gt_track_id'])
        same_frame_collisions += sum(comb(len(ids), 2) for ids in by_pred.values() if len(ids) >= 2)
    return {
        'observations': len(rows), 'gt_identities': len(true_counts),
        'predicted_identities': len(pred_counts),
        'pairwise_tp': tp, 'pairwise_fp': fp, 'pairwise_fn': fn,
        'pairwise_precision': precision, 'pairwise_recall': recall,
        'pairwise_f1': f1,
        'mean_predicted_fragments_per_gt': float(np.mean([len(s) for s in fragments.values()])),
        'gt_identities_split_across_multiple_predicted_ids': sum(len(s)>1 for s in fragments.values()),
        'same_frame_identity_collisions': same_frame_collisions,
        'new_id_decisions': sum(r['decision'] == 'new' for r in rows),
        'matched_decisions': sum(r['decision'] == 'match' for r in rows),
    }


def associate_sequence(seq, ids, obs, emb, threshold, memory_size):
    # Group by annotated frame, and process all objects in each frame against a
    # frozen memory snapshot. No observation from the current frame is available
    # to another observation in the same frame.
    by_frame = defaultdict(list)
    for idx in ids:
        by_frame[int(obs[idx]['frame_position'])].append(idx)
    memories = {}  # predicted ID -> {'category': ..., 'vectors': [...]}
    next_id = 0
    output = []
    for frame in sorted(by_frame):
        pending = []
        for idx in sorted(by_frame[frame]):
            o = obs[idx]
            category = str(o['category_id'])
            query = emb[idx]
            scores = []
            for pred_id, mem in memories.items():
                if mem['category'] != category:
                    continue
                # Stage 12A appearance baseline: max cosine over recent history.
                score = float(np.max(np.asarray(mem['vectors']) @ query))
                scores.append((score, pred_id))
            scores.sort(key=lambda x: (-x[0], x[1]))
            best = scores[0][0] if scores else None
            runner_up = scores[1][0] if len(scores) > 1 else None
            if best is not None and best >= threshold:
                predicted_id = scores[0][1]
                decision = 'match'
            else:
                predicted_id = f'p{next_id:04d}'
                next_id += 1
                decision = 'new'
            pending.append((idx, predicted_id, category, decision, best, runner_up))
        # Commit after entire frame has been scored; same-frame collisions are
        # intentionally possible in this ungated baseline and are evaluated.
        for idx, pred_id, category, decision, best, second in pending:
            if pred_id not in memories:
                memories[pred_id] = {'category': category, 'vectors': []}
            memories[pred_id]['vectors'].append(emb[idx])
            if memory_size > 0 and len(memories[pred_id]['vectors']) > memory_size:
                memories[pred_id]['vectors'] = memories[pred_id]['vectors'][-memory_size:]
            output.append({
                'sequence': seq, 'observation_index': idx,
                'frame_position': int(obs[idx]['frame_position']),
                'category_id': category,
                'gt_track_id': str(obs[idx]['track_id']),
                'predicted_id': pred_id, 'decision': decision,
                'best_similarity': best, 'second_similarity': second,
                'identity_gap': (best - second) if second is not None else None,
            })
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=Path('results/burst/identity_hard'))
    p.add_argument('--output', type=Path, default=Path('results/burst/online_association'))
    p.add_argument('--sequences', nargs='+', default=['person-17', 'sheep-19'])
    p.add_argument('--threshold', type=float, default=0.90,
                   help='Exploratory fixed cosine threshold; NOT calibrated on held-out videos')
    p.add_argument('--memory-size', type=int, default=0,
                   help='Maximum observations per predicted identity (0=unlimited)')
    args = p.parse_args()
    if not -1 <= args.threshold <= 1:
        p.error('--threshold must be between -1 and 1')
    if args.memory_size < 0:
        p.error('--memory-size must be nonnegative')
    obs = read_csv(args.input / 'observations.csv')
    required = {'dataset', 'sequence', 'frame_position', 'category_id', 'track_id'}
    if not obs or not required.issubset(obs[0]):
        raise ValueError(f'Expected observation columns {sorted(required)}; found {list(obs[0]) if obs else []}')
    with np.load(args.input / 'embeddings.npz') as z:
        emb = np.asarray(z['embeddings'], dtype=np.float32)
    if len(obs) != len(emb):
        raise ValueError(f'Observation/embedding mismatch: {len(obs)} vs {len(emb)}')
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    groups = defaultdict(list)
    for i, o in enumerate(obs):
        groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if args.sequences != ['all']:
        groups = {seq: ids for seq, ids in groups.items()
                  if seq in args.sequences or seq.split('/', 1)[1] in args.sequences}
    if not groups:
        raise ValueError('No matching sequences; use --sequences all to inspect available sequences')
    all_rows, per_sequence = [], {}
    for seq, ids in sorted(groups.items()):
        rows = associate_sequence(seq, ids, obs, emb, args.threshold, args.memory_size)
        all_rows.extend(rows)
        per_sequence[seq] = pairwise_metrics(rows)
        m = per_sequence[seq]
        print(f"{seq}: observations={m['observations']}, GT={m['gt_identities']}, "
              f"predicted={m['predicted_identities']}, pair-F1={m['pairwise_f1']:.3f}, "
              f"fragmentation={m['mean_predicted_fragments_per_gt']:.2f}, "
              f"same-frame collisions={m['same_frame_identity_collisions']}")
    # Pooled pair counts must be summed per video: identities cannot match across videos.
    tp = sum(m['pairwise_tp'] for m in per_sequence.values())
    fp = sum(m['pairwise_fp'] for m in per_sequence.values())
    fn = sum(m['pairwise_fn'] for m in per_sequence.values())
    pooled = {
        'observations': len(all_rows), 'sequences': len(per_sequence),
        'pairwise_tp': tp, 'pairwise_fp': fp, 'pairwise_fn': fn,
        'pairwise_precision': tp/(tp+fp) if tp+fp else None,
        'pairwise_recall': tp/(tp+fn) if tp+fn else None,
        'pairwise_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'mean_predicted_fragments_per_gt': float(np.mean([
            len({r['predicted_id'] for r in all_rows
                 if r['sequence']==seq and r['gt_track_id']==gt})
            for seq in per_sequence
            for gt in {r['gt_track_id'] for r in all_rows if r['sequence']==seq}])),
        'same_frame_identity_collisions': sum(m['same_frame_identity_collisions'] for m in per_sequence.values()),
        'predicted_identities': sum(m['predicted_identities'] for m in per_sequence.values()),
        'gt_identities': sum(m['gt_identities'] for m in per_sequence.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ['sequence','observation_index','frame_position','category_id','gt_track_id',
              'predicted_id','decision','best_similarity','second_similarity','identity_gap']
    write_csv(args.output / 'assignments.csv', all_rows, fields)
    summary = {
        'protocol': 'Online appearance-only association of GT segmentation crops; GT IDs used for evaluation only.',
        'threshold': args.threshold, 'memory_size': args.memory_size,
        'score': 'maximum CLIP cosine similarity to prior observations of a predicted identity',
        'pooled': pooled, 'per_sequence': per_sequence,
        'limitations': [
            'Uses oracle segmentation and category labels, not detector/tracker outputs.',
            'GT track IDs are excluded from association and used only for evaluation.',
            'No spatial/motion constraints, no same-frame one-to-one assignment, no uncertainty gating.',
            'A new identity is created when no similarity exceeds the fixed threshold.',
            'Pairwise clustering scores are not MOT IDF1, ID switches, or HOTA.',
            'Threshold is exploratory and not selected on independent calibration videos.',
            'Observations within videos are correlated; no confidence intervals.',
        ],
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nPooled:', json.dumps(pooled, indent=2))
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
