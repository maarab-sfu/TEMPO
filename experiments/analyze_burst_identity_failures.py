"""Stage 11.6: inspect CLIP identity association failures without rerunning CLIP.
Run: python -m experiments.analyze_burst_identity_failures
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def panel(obs, index, title, width=240, image_height=210):
    row = obs[index]
    image_path = Path(row['crop_path'])
    if not image_path.is_file():
        raise FileNotFoundError(f'Missing saved crop: {image_path}')
    with Image.open(image_path) as im:
        im = im.convert('RGB')
        im.thumbnail((width - 16, image_height - 16))
        out = Image.new('RGB', (width, image_height + 75), 'white')
        out.paste(im, ((width - im.width) // 2, (image_height - im.height) // 2))
    draw = ImageDraw.Draw(out)
    lines = [title, f"ID {row['track_id']} | {row['category']}",
             f"Frame pos {row['frame_position']} | {row['sequence'][:23]}"]
    for k, line in enumerate(lines):
        draw.text((8, image_height + 5 + 22 * k), line, fill='black')
    return out


def montage(obs, entries, output, category, count):
    entries = entries[:count]
    if not entries:
        return
    canvas = Image.new('RGB', (720, 300 * len(entries)), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for k, entry in enumerate(entries):
        qi, pi, ni = (int(entry[key]) for key in ('query_index', 'positive_index', 'negative_index'))
        for j, (index, label) in enumerate(((qi, 'Query'), (pi, 'Best genuine match'), (ni, 'Confusing impostor'))):
            canvas.paste(panel(obs, index, label), (j * 240, k * 300))
        draw.text((8, k * 300 + 286),
                  f"{category} #{k+1}: pos={float(entry['positive_score']):.3f}  "
                  f"neg={float(entry['negative_score']):.3f}  "
                  f"margin={float(entry['margin']):+.3f}", fill='black')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('results/burst/identity_hard'))
    p.add_argument('--output', type=Path, default=Path('results/burst/failure_analysis'))
    p.add_argument('--sequences', nargs='*', default=['sheep-19', 'person-17'],
                   help='Sequence names or dataset/name; use --sequences all for all five')
    p.add_argument('--top', type=int, default=20, help='Rows per visualization')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    obs = load_csv(args.input / 'observations.csv')
    npz = np.load(args.input / 'embeddings.npz')
    emb = np.asarray(npz['embeddings'], dtype=np.float32)
    if len(obs) != len(emb):
        raise ValueError('Observation count differs from embeddings; regenerate the benchmark outputs.')
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    grouped = defaultdict(list)
    for i, row in enumerate(obs):
        key = f"{row['dataset']}/{row['sequence']}"
        grouped[key].append(i)

    selected = grouped if args.sequences == ['all'] else {
        key: ids for key, ids in grouped.items()
        if key in args.sequences or key.split('/', 1)[1] in args.sequences
    }
    if not selected:
        raise ValueError(f'No matching sequences. Available: {list(grouped)}')

    comparisons = []
    per_sequence = {}
    for key, indices in selected.items():
        sub = emb[indices]
        scores = sub @ sub.T
        rows = []
        for a, idx in enumerate(indices):
            q = obs[idx]
            # Cross-frame, same-category candidate gallery, matching benchmark_burst_hard.py.
            candidates = [b for b, j in enumerate(indices)
                          if b != a and obs[j]['frame_position'] != q['frame_position']
                          and obs[j]['category_id'] == q['category_id']]
            pos = [b for b in candidates if obs[indices[b]]['track_id'] == q['track_id']]
            neg = [b for b in candidates if obs[indices[b]]['track_id'] != q['track_id']]
            if not pos or not neg:
                continue
            best_pos = max(pos, key=lambda b: float(scores[a, b]))
            best_neg = max(neg, key=lambda b: float(scores[a, b]))
            ps, ns = float(scores[a, best_pos]), float(scores[a, best_neg])
            margin = ps - ns
            record = dict(sequence=key, query_index=idx,
                          positive_index=indices[best_pos], negative_index=indices[best_neg],
                          query_track=q['track_id'], impostor_track=obs[indices[best_neg]]['track_id'],
                          query_frame=q['frame_position'],
                          positive_frame=obs[indices[best_pos]]['frame_position'],
                          negative_frame=obs[indices[best_neg]]['frame_position'],
                          positive_score=ps, negative_score=ns, margin=margin,
                          wrong_top1=int(margin < 0),
                          abs_margin=abs(margin))
            rows.append(record)
        comparisons.extend(rows)
        per_sequence[key] = {
            'queries': len(rows),
            'wrong_top1': sum(r['wrong_top1'] for r in rows),
            'wrong_top1_fraction': (sum(r['wrong_top1'] for r in rows) / len(rows)) if rows else None,
            'median_margin': float(np.median([r['margin'] for r in rows])) if rows else None,
            'fraction_margin_below_0_02': (sum(r['margin'] < .02 for r in rows) / len(rows)) if rows else None,
        }
        print(f"{key}: {len(rows)} queries, {per_sequence[key]['wrong_top1']} top-1 errors")

    if not comparisons:
        raise RuntimeError('No valid queries with both genuine and impostor candidates.')
    fields = list(comparisons[0])
    save_csv(args.output / 'query_margins.csv', comparisons, fields)
    false_matches = sorted((r for r in comparisons if r['wrong_top1']),
                           key=lambda r: r['margin'])
    ambiguous = sorted(comparisons, key=lambda r: r['abs_margin'])
    weak_positive = sorted(comparisons, key=lambda r: r['positive_score'])
    for name, data in [('false_matches', false_matches),
                       ('ambiguous_queries', ambiguous),
                       ('weak_genuine_matches', weak_positive)]:
        save_csv(args.output / f'{name}.csv', data, fields)
        montage(obs, data, args.output / f'{name}.jpg', name, args.top)

    # A margin is descriptive, not a calibrated confidence probability.
    summary = dict(sequences=per_sequence, queries=len(comparisons),
                   wrong_top1=sum(r['wrong_top1'] for r in comparisons),
                   note='Margin = highest genuine similarity minus highest impostor similarity. '
                        'Genuine identity is used to calculate this diagnostic; it is NOT '
                        'available to a deployed association model. '
                        'No calibration or independence assumptions are made.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
