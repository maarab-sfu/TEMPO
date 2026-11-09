"""Stage 11.8A: GT-ID oracle multi-observation memory for BURST.

Run from project root:
    python -m experiments.evaluate_burst_identity_memory
    python -m experiments.evaluate_burst_identity_memory --sequences all
    python -m experiments.evaluate_burst_identity_memory --gallery past

No images/GPU needed. IMPORTANT: track IDs are ground truth, used to construct
oracle candidate memories; these are NOT available to a deployable tracker.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def read_csv(p):
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    if not rows:
        return
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_memory(query, memory, method, k):
    sims = memory @ query
    if method == 'nearest':
        return float(sims.max())
    if method == 'mean':
        return float(sims.mean())
    if method == 'topk':
        return float(np.sort(sims)[-k:].mean())
    if method == 'prototype':
        proto = memory.mean(axis=0)
        proto /= max(float(np.linalg.norm(proto)), 1e-12)
        return float(proto @ query)
    raise ValueError(method)


def metrics(rows, method):
    if not rows:
        return None
    wrong = np.array([int(r[f'{method}_wrong']) for r in rows], dtype=np.int32)
    conf = np.array([float(r[f'{method}_score']) for r in rows])
    gap = np.array([float(r[f'{method}_gap']) for r in rows])
    out = {'queries': len(rows), 'errors': int(wrong.sum()),
           'recall_at_1': float(1-wrong.mean()),
           'error_rate': float(wrong.mean()), 'confidence': {}}
    for name, values in [('score', conf), ('identity_gap', gap)]:
        order = np.argsort(-values, kind='stable')
        risks = np.cumsum(wrong[order]) / np.arange(1, len(rows)+1)
        stat = {'error_detection_auroc': float(roc_auc_score(wrong, -values))
                if len(np.unique(wrong)) == 2 else None,
                'aurc': float(risks.mean()), 'risk_at_coverage': {}}
        for c in [0.25, 0.5, 0.75, 1.0]:
            n = max(1, int(np.ceil(c*len(rows))))
            stat['risk_at_coverage'][str(c)] = float(risks[n-1])
        out['confidence'][name] = stat
    return out


def run(obs, emb, requested, gallery, min_history, k):
    groups = defaultdict(list)
    for i, o in enumerate(obs):
        groups[f"{o['dataset']}/{o['sequence']}"].append(i)
    if requested != ['all']:
        groups = {name: ids for name, ids in groups.items()
                  if name in requested or name.split('/',1)[1] in requested}
    if not groups:
        raise ValueError('No matching sequences')
    methods = ['nearest', 'mean', 'topk', 'prototype']
    results = []
    for seq, ids in groups.items():
        frame = np.array([int(obs[i]['frame_position']) for i in ids])
        track = np.array([str(obs[i]['track_id']) for i in ids])
        cat = np.array([str(obs[i]['category_id']) for i in ids])
        X = emb[ids]
        for a, idx in enumerate(ids):
            # Strictly exclude query frame (including other objects in that frame).
            allowed = (frame != frame[a]) & (cat == cat[a])
            if gallery == 'past':
                allowed &= frame < frame[a]
            # Each GT track is an oracle identity memory.
            memories = {}
            for t in np.unique(track[allowed]):
                members = np.flatnonzero(allowed & (track == t))
                if len(members) >= min_history:
                    memories[t] = members
            if len(memories) < 2 or track[a] not in memories:
                continue
            row = {'sequence': seq, 'query_index': idx, 'query_track_id': track[a],
                   'query_frame_position': int(frame[a]), 'gallery_mode': gallery,
                   'candidate_identities': len(memories),
                   'positive_history_count': len(memories[track[a]])}
            for method in methods:
                scores = [(t, score_memory(X[a], X[members], method, k))
                          for t, members in memories.items()]
                scores.sort(key=lambda pair: (-pair[1], pair[0]))
                pred, best = scores[0]
                row[f'{method}_predicted_track_id'] = pred
                row[f'{method}_score'] = best
                row[f'{method}_gap'] = best - scores[1][1]
                row[f'{method}_wrong'] = int(pred != track[a])
            results.append(row)
        n = sum(r['sequence'] == seq for r in results)
        print(f'{seq}: {n} eligible queries')
    return results, methods


def plot(rows, methods, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; skipping plots')
        return
    groups = {'pooled': rows}
    for seq in sorted(set(r['sequence'] for r in rows)):
        groups[seq] = [r for r in rows if r['sequence'] == seq]
    for group, subset in groups.items():
        plt.figure(figsize=(8,5))
        for method in methods:
            wrong = np.array([r[f'{method}_wrong'] for r in subset], dtype=int)
            confidence = np.array([r[f'{method}_score'] for r in subset], dtype=float)
            order = np.argsort(-confidence, kind='stable')
            risk = np.cumsum(wrong[order]) / np.arange(1,len(wrong)+1)
            coverage = np.arange(1,len(wrong)+1) / len(wrong)
            plt.plot(coverage, risk, label=f'{method} (AURC={risk.mean():.3f})')
        plt.xlabel('Coverage'); plt.ylabel('Selective top-1 error')
        plt.title(f'GT oracle memory: {group}')
        plt.xlim(0,1); plt.ylim(0,1); plt.grid(alpha=.25); plt.legend()
        plt.tight_layout()
        plt.savefig(output / f'risk_coverage_{group.replace("/","_")}.png', dpi=160)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('results/burst/identity_hard'))
    p.add_argument('--output', type=Path, default=Path('results/burst/identity_memory'))
    p.add_argument('--sequences', nargs='+', default=['person-17','sheep-19'])
    p.add_argument('--gallery', choices=['crossframe','past'], default='crossframe',
                   help='crossframe matches Stage 11.7 gallery; past is causal but changes eligible queries')
    p.add_argument('--min-history', type=int, default=1)
    p.add_argument('--topk', type=int, default=3)
    args = p.parse_args()
    if args.min_history < 1 or args.topk < 1:
        p.error('--min-history and --topk must be >=1')
    obs = read_csv(args.input/'observations.csv')
    with np.load(args.input/'embeddings.npz') as z:
        emb = np.asarray(z['embeddings'], dtype=np.float32)
    if len(obs) != len(emb):
        raise ValueError(f'Observation/embedding mismatch: {len(obs)} vs {len(emb)}')
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    rows, methods = run(obs, emb, args.sequences, args.gallery, args.min_history, args.topk)
    if not rows:
        raise RuntimeError('No eligible queries; try --min-history 1 or --gallery crossframe')
    output = args.output / args.gallery
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'query_results.csv', rows)
    groups = {'pooled': rows}
    for seq in sorted(set(r['sequence'] for r in rows)):
        groups[seq] = [r for r in rows if r['sequence'] == seq]
    summary = {
        'protocol': 'GT-ID ORACLE candidate memories. Same-category, cross-frame retrieval.',
        'gallery': args.gallery,
        'topk': args.topk, 'min_history': args.min_history,
        'query_eligibility': 'At least two GT identity memories including the query identity, each with >=min_history observations.',
        'caveats': [
            'GT track IDs are used to construct candidate memories: NOT a deployable method.',
            'The nearest baseline is max similarity over an oracle-identity memory; it should reproduce the original top-1 retrieval on the same eligible queries when min_history=1 and gallery=crossframe.',
            'All methods are evaluated on identical eligible queries within this run.',
            'Crossframe mode uses past and future observations; past mode uses only earlier annotated frames.',
            'No independent videos used for calibration; risk-coverage is descriptive.',
            'Observations are correlated within sequences; no confidence intervals.',
            'Mean/prototype use all eligible history; topk uses up to k highest-similarity history observations.',
        ],
        'groups': {name: {method: metrics(subset, method) for method in methods}
                   for name, subset in groups.items()},
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot(rows, methods, output)
    print('\nRecall@1 / score AUROC / score AURC:')
    for name, group in summary['groups'].items():
        print(name)
        for method, stat in group.items():
            print(f'  {method:10} R@1={stat["recall_at_1"]:.3f} '
                  f'AUROC={stat["confidence"]["score"]["error_detection_auroc"]} '
                  f'AURC={stat["confidence"]["score"]["aurc"]:.3f}')
    print('\nSaved:', output.resolve())


if __name__ == '__main__':
    main()
