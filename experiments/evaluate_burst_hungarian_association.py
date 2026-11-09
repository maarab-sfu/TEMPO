"""Stage 12B: causal, frame-level one-to-one online association on BURST GT observations.

Uses GT segmentation-derived CLIP crops as detections, but NEVER uses GT identity
in assignment. GT identity is read only for evaluation. Not a full MOT benchmark.

Run from project root:
  python -m experiments.evaluate_burst_hungarian_association --sequences all
  python -m experiments.evaluate_burst_hungarian_association --sequences person-17 sheep-19
  python -m experiments.evaluate_burst_hungarian_association --sequences all --threshold 0.90
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from math import comb
import numpy as np
from scipy.optimize import linear_sum_assignment


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
    """One-to-one per-frame assignment with explicit unmatched dummy columns.

    Existing memories are frozen while the frame is scored. Real edge utility is
    cosine_similarity - threshold; unmatched utility is zero. This optimizes
    which *valid* matches to accept jointly, rather than post-filtering an
    unconstrained Hungarian assignment. GT IDs are never consulted here.
    """
    by_frame = defaultdict(list)
    for idx in ids:
        by_frame[int(obs[idx]['frame_position'])].append(idx)
    memories = {}
    next_id = 0
    output = []
    for frame in sorted(by_frame):
        frame_ids = sorted(by_frame[frame])
        pred_ids = sorted(memories)
        n, m = len(frame_ids), len(pred_ids)
        # Every row has a private dummy column, so it can always choose 'new'.
        # Invalid real edges are strictly worse than an unmatched assignment.
        utility = np.full((n, m + n), -1e6, dtype=np.float64)
        for r in range(n):
            utility[r, m + r] = 0.0
        ranked = []
        for r, idx in enumerate(frame_ids):
            category = str(obs[idx]['category_id'])
            query = emb[idx]
            scores = []
            for c, pred_id in enumerate(pred_ids):
                mem = memories[pred_id]
                if mem['category'] != category:
                    continue
                score = float(np.max(np.asarray(mem['vectors']) @ query))
                scores.append((score, pred_id))
                if score >= threshold:
                    # Tiny bonus makes an exactly-threshold match preferable to
                    # creating a new ID; it cannot outweigh meaningful scores.
                    utility[r, c] = score - threshold + 1e-10
            scores.sort(key=lambda x: (-x[0], x[1]))
            ranked.append(scores)
        # Maximize valid association utility with one-to-one constraints.
        row_ind, col_ind = linear_sum_assignment(utility, maximize=True)
        assignment = dict(zip(row_ind.tolist(), col_ind.tolist()))
        pending = []
        for r, idx in enumerate(frame_ids):
            scores = ranked[r]
            best = scores[0][0] if scores else None
            runner_up = scores[1][0] if len(scores) > 1 else None
            col = assignment[r]
            if col < m and utility[r, col] >= 0:
                predicted_id = pred_ids[col]
                decision = 'match'
                selected_similarity = next(s for s, pid in scores if pid == predicted_id)
            else:
                predicted_id = f'p{next_id:04d}'
                next_id += 1
                decision = 'new'
                selected_similarity = None
            pending.append((idx, predicted_id, str(obs[idx]['category_id']),
                            decision, best, runner_up, selected_similarity))
        # Update only after ALL observations in the frame have been assigned.
        for idx, pred_id, category, decision, best, second, selected in pending:
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
                'selected_similarity': selected,
            })
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=Path('results/burst/identity_hard'))
    p.add_argument('--output', type=Path, default=Path('results/burst/hungarian_association'))
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
    assert all(m['same_frame_identity_collisions'] == 0 for m in per_sequence.values()), 'One-to-one invariant violated'
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
              'predicted_id','decision','best_similarity','second_similarity','identity_gap','selected_similarity']
    write_csv(args.output / 'assignments.csv', all_rows, fields)
    summary = {
        'protocol': 'Online one-to-one Hungarian association of GT segmentation crops; GT IDs used for evaluation only.',
        'threshold': args.threshold, 'memory_size': args.memory_size,
        'score': 'maximum CLIP cosine similarity to prior observations of a predicted identity',
        'assignment': 'Frame-level Hungarian, real-edge utility = cosine - threshold, private unmatched dummy columns',
        'pooled': pooled, 'per_sequence': per_sequence,
        'limitations': [
            'Uses oracle segmentation and category labels, not detector/tracker outputs.',
            'GT track IDs are excluded from association and used only for evaluation.',
            'No spatial/motion constraints or uncertainty gating; same-frame one-to-one assignment is enforced.',
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
