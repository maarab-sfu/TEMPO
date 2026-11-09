"""Rank BURST validation sequences with distinct GT tracks of the same category.

Run: python -m experiments.mine_burst_hard_negatives --top 20
No images or GPU required. Uses the existing BURST validation annotation JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / 'data/annotations/burst/val/all_classes.json'
DEFAULT_OUTPUT = ROOT / 'results/burst/hard_negative_mining'


def analyze(sequence: dict, categories: dict[int, str], min_observations: int) -> dict | None:
    dataset = str(sequence.get('dataset', 'unknown'))
    seq_name = str(sequence.get('seq_name', 'unknown'))
    paths = sequence.get('annotated_image_paths', [])
    segmentations = sequence.get('segmentations', [])
    if len(paths) != len(segmentations):
        raise ValueError(f'{dataset}/{seq_name}: paths/segmentations length mismatch')

    cat_by_id = {int(k): int(v) for k, v in sequence.get('track_category_ids', {}).items()}
    seen: dict[int, set[int]] = defaultdict(set)
    for frame_idx, frame in enumerate(segmentations):
        for raw_id, mask_data in frame.items():
            # Only use frames with an actual annotated mask; no inference about unannotated objects.
            if mask_data is not None:
                seen[int(raw_id)].add(frame_idx)

    by_cat: dict[int, list[int]] = defaultdict(list)
    for tid, frame_set in seen.items():
        if len(frame_set) >= min_observations and tid in cat_by_id:
            by_cat[cat_by_id[tid]].append(tid)

    pairs = []
    for cat_id, tids in by_cat.items():
        for a, b in combinations(sorted(tids), 2):
            fa, fb = seen[a], seen[b]
            overlap = len(fa & fb)
            pairs.append({
                'track_a': a, 'track_b': b,
                'category_id': cat_id,
                'category': categories.get(cat_id, f'category_{cat_id}'),
                'observations_a': len(fa), 'observations_b': len(fb),
                'co_visible_frames': overlap,
                'first_a': min(fa), 'last_a': max(fa),
                'first_b': min(fb), 'last_b': max(fb),
            })
    if not pairs:
        return None

    co_visible_pairs = sum(p['co_visible_frames'] > 0 for p in pairs)
    total_co_visible = sum(p['co_visible_frames'] for p in pairs)
    max_co_visible = max(p['co_visible_frames'] for p in pairs)
    return {
        'dataset': dataset, 'seq_name': seq_name,
        'annotated_frames': len(paths),
        'eligible_tracks': sum(len(tids) for tids in by_cat.values()),
        'same_category_pairs': len(pairs),
        'co_visible_pairs': co_visible_pairs,
        'total_co_visible_frames_across_pairs': total_co_visible,
        'max_co_visible_frames_for_one_pair': max_co_visible,
        'categories': ', '.join(sorted({p['category'] for p in pairs})),
        'pairs': pairs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--min-observations', type=int, default=5,
                    help='Minimum annotated frames for each track (default: 5)')
    ap.add_argument('--min-co-visible', type=int, default=3,
                    help='Minimum frames with both same-category tracks annotated (default: 3)')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--per-dataset', type=int, default=10,
                    help='Maximum sequences per source dataset in selected shortlist')
    args = ap.parse_args()
    if args.min_observations < 1 or args.min_co_visible < 0 or args.top < 1 or args.per_dataset < 1:
        ap.error('min-observations >= 1, min-co-visible >= 0, top >= 1, per-dataset >= 1 required')

    with args.annotations.open(encoding='utf-8') as f:
        data = json.load(f)
    categories = {int(c['id']): c['name'] for c in data['categories']}
    results = []
    for seq in data['sequences']:
        r = analyze(seq, categories, args.min_observations)
        if r is None:
            continue
        if not any(p['co_visible_frames'] >= args.min_co_visible for p in r['pairs']):
            continue
        results.append(r)

    results.sort(key=lambda r: (
        r['co_visible_pairs'], r['total_co_visible_frames_across_pairs'],
        r['same_category_pairs'], r['annotated_frames']), reverse=True)
    selected, dataset_counts = [], defaultdict(int)
    for r in results:
        if dataset_counts[r['dataset']] >= args.per_dataset:
            continue
        selected.append(r)
        dataset_counts[r['dataset']] += 1
        if len(selected) >= args.top:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    columns = ['dataset', 'seq_name', 'annotated_frames', 'eligible_tracks',
               'same_category_pairs', 'co_visible_pairs',
               'total_co_visible_frames_across_pairs',
               'max_co_visible_frames_for_one_pair', 'categories']
    with (args.output / 'ranked_sequences.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in columns})
    with (args.output / 'selected_sequences.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in selected:
            w.writerow({k: r[k] for k in columns})
    with (args.output / 'selected_pairs.csv').open('w', newline='', encoding='utf-8') as f:
        pair_columns = ['dataset', 'seq_name', 'track_a', 'track_b', 'category_id', 'category',
                        'observations_a', 'observations_b', 'co_visible_frames',
                        'first_a', 'last_a', 'first_b', 'last_b']
        w = csv.DictWriter(f, fieldnames=pair_columns)
        w.writeheader()
        for r in selected:
            for p in r['pairs']:
                w.writerow({'dataset': r['dataset'], 'seq_name': r['seq_name'], **p})
    with (args.output / 'selected_details.json').open('w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2)

    print(f'Total validation sequences: {len(data["sequences"])}')
    print(f'Eligible hard-negative sequences: {len(results)}')
    print(f'Selected: {len(selected)} (max {args.per_dataset} per dataset)')
    print(f'Criteria: >= {args.min_observations} observations/track; '
          f'>= {args.min_co_visible} co-visible frames for at least one same-category pair')
    print('\nSelected sequences:')
    for r in selected:
        print(f'  {r["dataset"]}/{r["seq_name"]}: '
              f'{r["co_visible_pairs"]} co-visible pairs, '
              f'{r["total_co_visible_frames_across_pairs"]} pair-frame overlaps, '
              f'{r["annotated_frames"]} frames; {r["categories"]}')
    print(f'\nOutputs: {args.output}')
    print('No image downloads performed.')


if __name__ == '__main__':
    main()
