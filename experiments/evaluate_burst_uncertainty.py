"""Stage 11.7: inference-available confidence signals for BURST object retrieval.

Run from project root:
    python -m experiments.evaluate_burst_uncertainty

Uses the same cross-frame, same-category gallery as benchmark_burst_hard.py.
No model inference or image downloads are required.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def risk_coverage(errors: np.ndarray, uncertainty: np.ndarray):
    """Sort from most confident to least; include all prefixes of size 1..N.

    Stable sort for ties. AURC is the mean prefix risk (discrete integral).
    Coverage thresholds are selected from the evaluation set: descriptive only.
    """
    n = len(errors)
    order = np.argsort(uncertainty, kind='stable')
    accepted_errors = errors[order]
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.cumsum(accepted_errors) / np.arange(1, n + 1)
    return coverage, risk


def metric_summary(rows, signal, coverages):
    if not rows:
        return None
    error = np.asarray([int(r['wrong_top1']) for r in rows], dtype=int)
    values = np.asarray([float(r[signal]) for r in rows], dtype=float)
    # Larger uncertainty means more likely wrong.
    uncertainty = -values
    cov, risk = risk_coverage(error, uncertainty)
    out = {
        'queries': len(rows),
        'errors': int(error.sum()),
        'baseline_error_rate': float(error.mean()),
        'error_detection_auroc': float(roc_auc_score(error, uncertainty)) if len(np.unique(error)) == 2 else None,
        'error_detection_average_precision': float(average_precision_score(error, uncertainty)) if error.sum() > 0 else None,
        'aurc': float(risk.mean()),
        'selective_risk': {},
    }
    for c in coverages:
        # Fixed number of accepted queries; threshold learned on same test set is NOT validated.
        k = max(1, min(len(rows), int(np.ceil(c * len(rows)))))
        out['selective_risk'][f'{c:.2f}'] = {
            'accepted': k,
            'actual_coverage': float(cov[k - 1]),
            'error_rate': float(risk[k - 1]),
        }
    return out


def build_rows(observations, embeddings, requested):
    grouped = defaultdict(list)
    for i, row in enumerate(observations):
        key = f"{row['dataset']}/{row['sequence']}"
        grouped[key].append(i)
    selected = grouped if requested == ['all'] else {
        key: ids for key, ids in grouped.items()
        if key in requested or key.split('/', 1)[1] in requested
    }
    if not selected:
        raise ValueError(f'No matching sequences; available: {list(grouped)}')
    rows = []
    for sequence, indices in selected.items():
        sim = embeddings[indices] @ embeddings[indices].T
        for a, query_idx in enumerate(indices):
            query = observations[query_idx]
            candidates = [b for b, idx in enumerate(indices)
                          if b != a
                          and observations[idx]['frame_position'] != query['frame_position']
                          and observations[idx]['category_id'] == query['category_id']]
            # Keep the identical eligibility criterion used in Stage 11.5C.
            positive = [b for b in candidates if observations[indices[b]]['track_id'] == query['track_id']]
            negative = [b for b in candidates if observations[indices[b]]['track_id'] != query['track_id']]
            if not positive or not negative or len(candidates) < 2:
                continue
            # Stable tie breaking: first occurrence in gallery order.
            ranked = sorted(candidates, key=lambda b: -float(sim[a, b]))
            best, second = ranked[:2]
            top1 = float(sim[a, best])
            top2 = float(sim[a, second])
            # Ground truth is used ONLY to compute the evaluation label, never the scores.
            wrong = int(observations[indices[best]]['track_id'] != query['track_id'])
            rows.append({
                'sequence': sequence,
                'query_index': query_idx,
                'query_track_id': query['track_id'],
                'predicted_track_id_for_evaluation_only': observations[indices[best]]['track_id'],
                'top1_index': indices[best],
                'top2_index': indices[second],
                'gallery_size': len(candidates),
                'wrong_top1': wrong,
                'top1_similarity': top1,
                'top2_similarity': top2,
                'top1_top2_gap': top1 - top2,
            })
        seqrows = [r for r in rows if r['sequence'] == sequence]
        print(f'{sequence}: {len(seqrows)} queries, {sum(r["wrong_top1"] for r in seqrows)} errors')
    return rows


def plot_curves(groups, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; skipping plots')
        return
    for group_name, rows in groups.items():
        if not rows:
            continue
        errors = np.array([int(r['wrong_top1']) for r in rows])
        plt.figure(figsize=(7, 5))
        for signal, label in [('top1_similarity', 'Top-1 similarity'),
                              ('top1_top2_gap', 'Top-1 / top-2 gap')]:
            cov, risk = risk_coverage(errors, -np.array([float(r[signal]) for r in rows]))
            plt.plot(cov, risk, label=f'{label} (AURC={risk.mean():.3f})')
        plt.axhline(float(errors.mean()), linestyle='--', color='gray', label='No abstention')
        plt.xlabel('Coverage (fraction of queries accepted)')
        plt.ylabel('Selective error rate (risk)')
        plt.title(f'Risk–coverage: {group_name}')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        safe = group_name.replace('/', '_')
        plt.savefig(output / f'risk_coverage_{safe}.png', dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('results/burst/identity_hard'))
    parser.add_argument('--output', type=Path, default=Path('results/burst/uncertainty_baselines'))
    parser.add_argument('--sequences', nargs='+', default=['sheep-19', 'person-17'],
                        help='Sequence IDs, dataset/name, or all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    obs = read_csv(args.input / 'observations.csv')
    with np.load(args.input / 'embeddings.npz') as npz:
        emb = np.asarray(npz['embeddings'], dtype=np.float32)
    if len(obs) != len(emb):
        raise ValueError(f'Observations ({len(obs)}) != embeddings ({len(emb)})')
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    rows = build_rows(obs, emb, args.sequences)
    if not rows:
        raise RuntimeError('No eligible cross-frame, same-category queries')
    write_csv(args.output / 'query_confidence.csv', rows, list(rows[0]))
    groups = {'pooled': rows}
    for sequence in sorted(set(r['sequence'] for r in rows)):
        groups[sequence] = [r for r in rows if r['sequence'] == sequence]
    summary = {
        'protocol': 'Cross-frame, same-category gallery; ground truth only labels top-1 errors.',
        'signals': {'top1_similarity': 'Higher means more confident',
                    'top1_top2_gap': 'Higher means more confident; may compare two observations of the same identity'},
        'coverage_note': 'Coverage is an evaluation-set quantile, not a deployable calibrated threshold.',
        'groups': {},
    }
    for group_name, subset in groups.items():
        summary['groups'][group_name] = {
            s: metric_summary(subset, s, [0.25, 0.50, 0.75, 1.0])
            for s in ['top1_similarity', 'top1_top2_gap']
        }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_curves(groups, args.output)
    print('\nResults (AUROC / AURC / error @ 50% coverage):')
    for group_name, results in summary['groups'].items():
        print(f'\n{group_name}')
        for signal, stats in results.items():
            print(f'  {signal}: AUROC={stats["error_detection_auroc"]}  '
                  f'AURC={stats["aurc"]:.4f}  '
                  f'Risk@50%={stats["selective_risk"]["0.50"]["error_rate"]:.4f}')
    print('\nSaved:', args.output.resolve())


if __name__ == '__main__':
    main()
