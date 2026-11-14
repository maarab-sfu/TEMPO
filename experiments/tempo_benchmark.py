"""TEMPO Stage 14: reproducible YouCook2 retrieval benchmark (silver labels).

Run from TEMPO repository root:
  python -m experiments.tempo_benchmark --split dev
  python -m experiments.tempo_benchmark --split test

Uses the existing Stage 13 CLIP encoder and retrieval functions; no Qwen inference.
Split is deterministic by video ID, independent of local file enumeration. All
available annotated videos are included unless --max-videos is specified.

Important: This is a retrieval benchmark, not a temporal QA benchmark. Generated
next-event questions identify the next *annotated* event, not necessarily the
next visually observable action. Video-level bootstrap intervals reflect sampling
variation, not annotation uncertainty or dataset shift.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.evaluate_video_rag_stage13_event import (
    sample_video, clip_embeddings, retrieve, event_window,
)

METHODS = ('frame', 'temporal', 'event')


def split_for_video(video_id: str, seed: int, test_fraction: float) -> str:
    digest = hashlib.sha256(f'{seed}:{video_id}'.encode()).digest()
    u = int.from_bytes(digest[:8], 'big') / 2**64
    return 'test' if u < test_fraction else 'dev'


def events_from_annotations(annotations):
    events = []
    for a in annotations:
        seg = a.get('segment')
        sentence = a.get('sentence')
        if not isinstance(seg, (list, tuple)) or len(seg) != 2 or not isinstance(sentence, str):
            continue
        try:
            start, end = float(seg[0]), float(seg[1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start or not sentence.strip():
            continue
        events.append((start, end, sentence.strip().rstrip('.'), str(a.get('id', len(events)))))
    return sorted(events, key=lambda e: (e[0], e[1], e[3]))


def make_questions(annotations, types):
    events = events_from_annotations(annotations)
    out = []
    for i, (start, end, sentence, event_id) in enumerate(events):
        if 'localization' in types:
            out.append(dict(question_id=f'loc_{i}', question_type='localization',
                            question=f'When does this action happen: {sentence}? Give an approximate interval in seconds.',
                            reference_start=start, reference_end=end,
                            reference_text=sentence, anchor_start=None, anchor_end=None))
        if 'successor' in types and i + 1 < len(events):
            nxt_start, nxt_end, nxt_sentence, _ = events[i + 1]
            if nxt_start >= end:
                out.append(dict(question_id=f'next_{i}_{i+1}', question_type='successor',
                                question=f'What happens after the action: {sentence}?',
                                reference_start=nxt_start, reference_end=nxt_end,
                                reference_text=nxt_sentence, anchor_start=start, anchor_end=end))
    return out


def retrieval_stats(indices, times, start, end):
    selected = np.asarray(times)[indices]
    relevant = (times >= start) & (times <= end)
    hits = int(((selected >= start) & (selected <= end)).sum())
    denom = int(relevant.sum())
    return dict(hits=hits, precision=hits / len(indices), any_hit=int(hits > 0),
                sampled_reference_frames=denom,
                sampled_recall=(hits / denom if denom else None),
                evaluable=int(denom > 0))


def evaluate_video(video_path, annotations, args, device):
    questions = make_questions(annotations, set(args.question_types))
    if not questions:
        return [], dict(sampled_frames=0, questions=0)
    frames, times, duration = sample_video(video_path, args.step, args.max_frames, args.resize)
    image_emb, query_emb = clip_embeddings(frames, [q['question'] for q in questions],
                                           args.clip_model, args.clip_batch, device)
    scores = query_emb @ image_emb.T
    rows = []
    for j, q in enumerate(questions):
        for mode in METHODS:
            if mode == 'event':
                indices, anchor = event_window(scores[j], times, args.budget, q['question'])
            else:
                indices = retrieve(scores[j], times, args.budget, mode, 0.)
                anchor = None
            m = retrieval_stats(indices, times, q['reference_start'], q['reference_end'])
            rows.append(dict(video_id=video_path.stem, question_id=q['question_id'],
                             question_type=q['question_type'], question=q['question'],
                             reference_text=q['reference_text'],
                             reference_start=q['reference_start'], reference_end=q['reference_end'],
                             anchor_start=q['anchor_start'], anchor_end=q['anchor_end'],
                             method=mode, budget=len(indices), anchor_seconds=(float(times[anchor]) if anchor is not None else None),
                             retrieved_seconds=json.dumps([round(float(times[i]), 3) for i in indices]), **m))
    return rows, dict(sampled_frames=len(frames), questions=len(questions), duration=duration)


def summarize(rows, bootstrap, seed):
    summary = {}
    rng = np.random.default_rng(seed)
    for question_type in ('localization', 'successor'):
        subset = [r for r in rows if r['question_type'] == question_type and r['evaluable']]
        if not subset:
            continue
        summary[question_type] = {}
        for method in METHODS:
            selected = [r for r in subset if r['method'] == method]
            by_video = defaultdict(list)
            for r in selected:
                by_video[r['video_id']].append(r)
            vids = sorted(by_video)
            if not vids:
                continue
            video_precision = np.asarray([np.mean([r['precision'] for r in by_video[v]]) for v in vids])
            video_hit = np.asarray([np.mean([r['any_hit'] for r in by_video[v]]) for v in vids])
            video_recall = np.asarray([np.mean([r['sampled_recall'] for r in by_video[v]]) for v in vids])
            result = dict(videos=len(vids), questions=len(selected),
                          macro_video_precision=float(video_precision.mean()),
                          macro_video_hit_rate=float(video_hit.mean()),
                          macro_video_sampled_recall=float(video_recall.mean()),
                          micro_precision=float(np.mean([r['precision'] for r in selected])),
                          micro_hit_rate=float(np.mean([r['any_hit'] for r in selected])))
            if len(vids) >= 2 and bootstrap:
                ids = rng.integers(0, len(vids), size=(bootstrap, len(vids)))
                for key, values in [('precision', video_precision), ('hit_rate', video_hit), ('sampled_recall', video_recall)]:
                    lo, hi = np.quantile(values[ids].mean(axis=1), [0.025, 0.975])
                    result[f'macro_video_{key}_bootstrap95'] = [float(lo), float(hi)]
            summary[question_type][method] = result
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, default=Path('data/annotations/youcook2.json'))
    p.add_argument('--videos', type=Path, default=Path('data/raw/youcook2'))
    p.add_argument('--output', type=Path, default=Path('results/stage14'))
    p.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--test-fraction', type=float, default=0.2)
    p.add_argument('--max-videos', type=int, default=None, help='Pilot only: caps selected videos after split')
    p.add_argument('--budget', type=int, default=12)
    p.add_argument('--step', type=float, default=2.)
    p.add_argument('--max-frames', type=int, default=2000)
    p.add_argument('--resize', type=int, default=768)
    p.add_argument('--clip-batch', type=int, default=16)
    p.add_argument('--clip-model', default='openai/clip-vit-base-patch32')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--question-types', nargs='+', choices=['localization', 'successor'], default=['localization', 'successor'])
    args = p.parse_args()
    if not 0 < args.test_fraction < 1 or args.budget < 1 or args.step <= 0 or args.clip_batch < 1 or args.bootstrap < 0:
        p.error('Invalid test-fraction, budget, step, clip-batch, or bootstrap')
    if args.max_videos is not None and args.max_videos < 1:
        p.error('--max-videos must be positive')
    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    database = json.loads(args.annotations.read_text(encoding='utf-8'))['database']
    all_paths = sorted(p for p in args.videos.glob('*.mp4') if p.stem in database and database[p.stem].get('subset') == 'validation')
    chosen = [p for p in all_paths if args.split == 'all' or split_for_video(p.stem, args.seed, args.test_fraction) == args.split]
    if args.max_videos is not None:
        chosen = chosen[:args.max_videos]
    if not chosen:
        raise SystemExit('No annotated validation MP4s in this split. Download more videos or use --split all for a smoke test.')
    folder = args.output / args.split
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol='TEMPO Stage 14 silver-label retrieval-only',
                    annotations=str(args.annotations), videos=str(args.videos), split=args.split,
                    seed=args.seed, test_fraction=args.test_fraction, budget=args.budget,
                    step=args.step, clip_model=args.clip_model, question_types=args.question_types,
                    available_validation_videos=len(all_paths), selected_videos=[p.stem for p in chosen],
                    max_videos=args.max_videos, completed=[], failed=[],
                    caveats=['Questions and intervals are annotation-derived, not human validated',
                             'Next-event question refers to the next annotated event, not necessarily next visible action',
                             'Retrieval overlap is not answer correctness',
                             'Zero sampled frames in an annotated interval are excluded from aggregate metrics',
                             'Split is by video ID; test split must not be used for iterative tuning'])
    all_rows = []
    for i, video in enumerate(chosen, 1):
        print(f'[{i}/{len(chosen)}] {video.stem}', flush=True)
        try:
            rows, metadata = evaluate_video(video, database[video.stem]['annotations'], args, device)
            all_rows.extend(rows)
            manifest['completed'].append(dict(video_id=video.stem, **metadata))
        except Exception as e:
            manifest['failed'].append(dict(video_id=video.stem, error=str(e)))
            print(f'  FAILED: {e}', flush=True)
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not all_rows:
        raise SystemExit('No results; inspect manifest.json')
    with (folder / 'per_question.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = dict(config={k: v for k, v in manifest.items() if k not in ('completed', 'failed')},
                   completed_videos=len(manifest['completed']), failed_videos=len(manifest['failed']),
                   question_count=len(all_rows) // len(METHODS),
                   zero_sampled_reference_question_count=sum(r['evaluable'] == 0 for r in all_rows if r['method'] == 'frame'),
                   metrics=summarize(all_rows, args.bootstrap, args.seed))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('completed_videos', 'failed_videos', 'question_count', 'zero_sampled_reference_question_count', 'metrics')}, indent=2))
    print('Saved:', folder)

if __name__ == '__main__':
    main()
