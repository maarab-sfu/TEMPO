"""TEMPO Stage 13: no-oracle anchor-to-successor retrieval ablation.

Run from repository root:
 python -m experiments.evaluate_stage13_successor --batch-dir results/stage13/batch

Uses the same frozen CLIP backbone, 2-second sampling and 12-frame evidence
budget as the Stage 13 pilot. Baseline JSON files are read, never modified.
This is a VISUAL CHANGE proposal baseline, not learned event segmentation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from experiments.evaluate_video_rag_stage13_event import sample_video, clip_embeddings, interval_metrics, load_vlm, answer


def successor_window(scores, image_embeddings, times, k=12, lookahead=24, gap=1):
    """Top CLIP anchor -> strongest nearby visual transition -> following k frames.

    The boundary score is cosine distance between adjacent normalized CLIP
    image embeddings, penalized mildly by distance from the anchor. Neither
    the target event interval nor the reference answer is used for retrieval.
    """
    n = len(scores)
    if n < k + 2:
        raise ValueError('Video too short for successor retrieval')
    anchor = int(np.argmax(scores))
    lo = anchor + gap + 1
    hi = min(n - k, anchor + lookahead)
    if lo > hi:
        start = min(max(anchor + gap, 0), n - k)
        return list(range(start, start + k)), anchor, None
    candidates = np.arange(lo, hi + 1)
    visual_change = 1.0 - np.sum(image_embeddings[candidates] * image_embeddings[candidates - 1], axis=1)
    # Normalize only within the lookahead. Distance is in sampled frames.
    scale = max(float(np.max(visual_change)), 1e-8)
    utility = visual_change / scale - 0.15 * (candidates - anchor) / max(lookahead, 1)
    boundary = int(candidates[int(np.argmax(utility))])
    return list(range(boundary, boundary + k)), anchor, boundary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', type=Path, default=Path('results/stage13/batch'))
    p.add_argument('--videos', type=Path, default=Path('data/raw/youcook2'))
    p.add_argument('--output', type=Path, default=Path('results/stage13/successor'))
    p.add_argument('--lookahead-frames', type=int, default=24)
    p.add_argument('--gap-frames', type=int, default=1)
    p.add_argument('--with-vlm', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=160)
    args = p.parse_args()
    if args.lookahead_frames < 1 or args.gap_frames < 0:
        p.error('lookahead-frames must be positive and gap-frames nonnegative')
    manifest = json.loads((args.batch_dir / 'manifest.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    model = processor = None
    for entry in manifest:
        if entry.get('status') != 'completed':
            continue
        vid = entry['video_id']
        baseline_path = args.batch_dir / 'predictions' / (vid + '_stage13_event.json')
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        k = int(baseline['evidence_frames'])
        if k != int(baseline['event_frames']):
            raise ValueError(f'{vid}: baseline budgets differ')
        video = args.videos / (vid + '.mp4')
        questions = {}
        for row in baseline['results']:
            if row['mode'] == 'frame' and row['question_id'].startswith('next_'):
                questions[row['question_id']] = row
        if not questions:
            continue
        print(f'Processing {vid}: {len(questions)} next-action questions', flush=True)
        frames, times, duration = sample_video(video, baseline['sampling_step_seconds'], 2000, 768)
        if len(times) != baseline['sampled_frames']:
            raise RuntimeError(f'{vid}: frame count changed; cannot compare to frozen baseline')
        qids = list(questions)
        img, txt = clip_embeddings(frames, [questions[q]['question'] for q in qids], baseline['clip_model'], 16, baseline['device'])
        if args.with_vlm and model is None:
            model, processor = load_vlm('Qwen/Qwen2.5-VL-3B-Instruct', baseline['device'])
        video_rows = []
        for j, qid in enumerate(qids):
            ref = questions[qid]
            scores = txt[j] @ img.T
            indices, anchor, boundary = successor_window(scores, img, times, k, args.lookahead_frames, args.gap_frames)
            q = {'id': qid, 'evidence_start_seconds': ref['reference_evidence_start_seconds'],
                 'evidence_end_seconds': ref['reference_evidence_end_seconds']}
            prediction = answer(model, processor, ref['question'], frames, times, indices, args.max_new_tokens) if args.with_vlm else None
            row = {'video_id': vid, 'question_id': qid, 'question': ref['question'],
                   'reference_answer': ref['reference_answer'], 'mode': 'successor',
                   'anchor_seconds': float(times[anchor]),
                   'visual_boundary_seconds': float(times[boundary]) if boundary is not None else None,
                   'retrieved_seconds': [float(times[i]) for i in indices],
                   'interval_retrieval_metrics': interval_metrics(indices, times, q),
                   'prediction': prediction}
            video_rows.append(row)
            all_rows.append(row)
        (args.output / (vid + '_successor.json')).write_text(json.dumps(video_rows, indent=2), encoding='utf-8')
    # Compare next-action queries only; calculate macro-average over videos.
    records = []
    for entry in manifest:
        vid = entry['video_id']
        base_path = args.batch_dir / 'predictions' / (vid + '_stage13_event.json')
        if not base_path.exists():
            continue
        baseline = json.loads(base_path.read_text(encoding='utf-8'))
        rows = [r for r in baseline['results'] if r['question_id'].startswith('next_')]
        rows += [r for r in all_rows if r['video_id'] == vid]
        for row in rows:
            m = row.get('interval_retrieval_metrics')
            if m is None:
                continue
            records.append({'video_id': vid, 'question_id': row['question_id'],
                            'mode': row['mode'], 'precision': m['inside_count'] / m['selected_count'],
                            'hit': int(m['any_hit'])})
    modes = ('frame', 'temporal', 'event', 'successor')
    summary = {'evaluation': 'next annotated event interval retrieval; silver-standard',
               'budget_frames': 12, 'methods': {}}
    for mode in modes:
        subset = [r for r in records if r['mode'] == mode]
        by_video = {}
        for r in subset:
            by_video.setdefault(r['video_id'], []).append(r)
        if not subset:
            continue
        summary['methods'][mode] = {
            'questions': len(subset), 'videos': len(by_video),
            'micro_frame_precision': float(np.mean([r['precision'] for r in subset])),
            'micro_hit_rate': float(np.mean([r['hit'] for r in subset])),
            'macro_video_frame_precision': float(np.mean([np.mean([r['precision'] for r in v]) for v in by_video.values()])),
            'macro_video_hit_rate': float(np.mean([np.mean([r['hit'] for r in v]) for v in by_video.values()]))}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.output / 'per_question.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'question_id', 'mode', 'precision', 'hit'])
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    print(f'Saved: {args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
