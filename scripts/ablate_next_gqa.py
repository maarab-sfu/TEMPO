"""M3.5 fixed-budget retrieval ablations. Run from project root.

Uses the M3.4 evaluator helpers without changing application code.
Baseline is read from the existing M3.4 journal; only variants invoke Qwen.
"""
import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluate_next_gqa import (load_selection, correct_letter, parse_choice,
    citation_metrics, mc_prompt, summarize)

VARIANTS = ('option_query', 'coverage')


def evidence_hit(timestamps, intervals, tolerance):
    return any(float(a)-tolerance <= float(t) <= float(b)+tolerance
               for t in timestamps for a,b in intervals)


def select_coverage(metadata, anchors, budget=8):
    """Preserve 2 question-only CLIP anchors; fill with evenly distributed time samples.

    Operates on indexed frames (not source video frames). Stable tie-breaking.
    """
    if budget < len(anchors):
        raise ValueError('budget smaller than anchor count')
    ordered = sorted(metadata, key=lambda f: (float(f['timestamp']), str(f['frame_id'])))
    if not ordered:
        return []
    by_path = {str(f['image_path']): f for f in ordered}
    selected = {}
    for rank, anchor in enumerate(anchors, 1):
        frame = by_path.get(str(anchor['image_path']))
        if frame is not None:
            item = dict(frame)
            item.update(evidence_role='semantic_anchor', anchor_rank=rank,
                        similarity=anchor.get('similarity'))
            selected[str(item['image_path'])] = item
    if len(selected) > budget:
        raise ValueError('too many anchors')
    if len(selected) == budget:
        return sorted(selected.values(), key=lambda f: f['timestamp'])
    # Farthest-point sampling in time, seeded by anchors; ensures coverage and
    # no duplicate frames. On empty seeds start with both endpoints.
    remaining = [f for f in ordered if str(f['image_path']) not in selected]
    start, end = float(ordered[0]['timestamp']), float(ordered[-1]['timestamp'])
    while remaining and len(selected) < budget:
        existing_times = [float(f['timestamp']) for f in selected.values()]
        def priority(f):
            t = float(f['timestamp'])
            distance = min(abs(t-x) for x in existing_times) if existing_times else min(t-start, end-t)
            return (distance, -t)
        candidate = max(remaining, key=priority)
        item = dict(candidate)
        item.update(evidence_role='temporal_context', anchor_rank=None, similarity=None)
        selected[str(item['image_path'])] = item
        remaining.remove(candidate)
    return sorted(selected.values(), key=lambda f: f['timestamp'])


def retrieval_query(row, variant):
    if variant == 'option_query':
        # All options, never the correct answer label or the gold field.
        options = '\n'.join(str(row[f'a{i}']) for i in range(5))
        return f"{row['question']}\nPossible answers:\n{options}"
    return str(row['question'])


def retrieve_frames(pipeline, row, index_root, variant):
    from src.storage.index_store import IndexStore
    from src.retrieval.temporal_retriever import TemporalRetriever
    retriever, _ = IndexStore(index_root / f"nextgqa_{row['video_id']}").load()
    embedding = pipeline.encoder.encode_text([retrieval_query(row, variant)])
    if variant == 'option_query':
        frames = TemporalRetriever(retriever).search_with_context(
            embedding, top_k=2, context_before=2, context_after=2, max_frames=8)
    elif variant == 'coverage':
        anchors = retriever.search(embedding, top_k=2)
        frames = select_coverage(retriever.metadata, anchors, budget=8)
    else:
        raise ValueError(variant)
    if not frames:
        raise RuntimeError('No frames retrieved')
    return frames


def evaluate_variant(pipeline, row, index_root, variant, tolerance):
    from src.reasoning.vlm import VisionLanguageModel
    start = time.perf_counter()
    frames = retrieve_frames(pipeline, row, index_root, variant)
    retrieval_seconds = time.perf_counter()-start
    if pipeline.vlm is None:
        pipeline.vlm = VisionLanguageModel()
    result = pipeline.vlm.answer_video_frames_grounded(frames, mc_prompt(row))
    cm = citation_metrics(frames, result['cited_frame_ids'], row['evidence_intervals'], tolerance)
    return dict(video_id=str(row['video_id']), qid=str(row['qid']), type=row['type'],
        variant=variant, status='ok', gold_option=correct_letter(row),
        pred_option=parse_choice(result['answer']), raw_answer=result['answer'],
        correct=parse_choice(result['answer']) == correct_letter(row),
        citation_parse_ok=bool(result['citation_parse_ok']),
        grounding_status=result['grounding_status'], cited_frame_ids=result['cited_frame_ids'],
        retrieved_frame_ids=[f['frame_id'] for f in frames],
        retrieved_timestamps=[float(f['timestamp']) for f in frames],
        evidence_intervals=row['evidence_intervals'],
        retrieval_hit=evidence_hit([f['timestamp'] for f in frames], row['evidence_intervals'], tolerance),
        retrieval_seconds=round(retrieval_seconds, 4), **cm)


def read_journal(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                records[(str(rec['video_id']), str(rec['qid']))] = rec
    return records


def metrics(records, tolerance):
    good = [r for r in records if r.get('status') == 'ok']
    result = summarize(records)
    result['retrieval_hit_rate'] = (sum(evidence_hit(r['retrieved_timestamps'],
        r['evidence_intervals'], tolerance) for r in good)/len(good)) if good else None
    available = [r for r in good if evidence_hit(r['retrieved_timestamps'], r['evidence_intervals'], tolerance)]
    result['citation_hit_given_retrieval_hit'] = (sum(bool(r['any_citation_in_gt']) for r in available)/len(available)) if available else None
    result['retrieval_hits'] = len(available)
    return result


def write_csv(path, records):
    columns = ['variant','video_id','qid','type','status','gold_option','pred_option',
        'correct','raw_answer','retrieval_hit','retrieved_timestamps','retrieved_frame_ids',
        'evidence_intervals','cited_frame_ids','citation_timestamps','citation_count',
        'citation_hits','citation_precision_temporal','any_citation_in_gt','citation_valid',
        'citation_parse_ok','retrieval_seconds','latency_seconds','error']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(r[k]) if isinstance(r.get(k), (dict,list)) else r.get(k, '') for k in columns})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, default=Path('data/datasets/next_gqa/manifests/selection.json'))
    ap.add_argument('--index-root', type=Path, default=Path('data/indexes'))
    ap.add_argument('--baseline', type=Path, default=Path('evaluation/next_gqa/results/seed42_n100_tol1.jsonl'))
    ap.add_argument('--output-dir', type=Path, default=Path('evaluation/next_gqa/ablations'))
    ap.add_argument('--variant', choices=(*VARIANTS, 'all'), default='all')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tolerance', type=float, default=1.0)
    ap.add_argument('--retry-errors', action='store_true')
    args = ap.parse_args()
    if args.limit < 0 or args.tolerance < 0: ap.error('limit and tolerance must be nonnegative')
    selection, questions = load_selection(args.selection)
    if args.limit: questions = questions[:args.limit]
    if not args.baseline.exists(): ap.error(f'Baseline journal not found: {args.baseline}')
    baseline = read_journal(args.baseline)
    keys = [(str(r['video_id']), str(r['qid'])) for r in questions]
    missing = [k for k in keys if k not in baseline or baseline[k].get('status') != 'ok']
    if missing: ap.error(f'Baseline missing {len(missing)} successful questions (e.g. {missing[:3]})')
    # Baseline records are only comparable at the same temporal tolerance.
    if args.tolerance != 1.0:
        ap.error('Baseline was evaluated at 1.0s tolerance; use --tolerance 1.0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = VARIANTS if args.variant == 'all' else (args.variant,)
    pipeline = None
    outputs = {'baseline': [dict(baseline[k], variant='baseline') for k in keys]}
    for variant in variants:
        name = f"seed{selection.get('seed','unknown')}_n{len(questions)}_{variant}_tol{args.tolerance:g}"
        journal = args.output_dir / f'{name}.jsonl'
        existing = read_journal(journal)
        for i, (row, key) in enumerate(zip(questions, keys), 1):
            if key in existing and (existing[key].get('status') == 'ok' or not args.retry_errors):
                print(f'[{variant} {i}/{len(questions)}] SKIP {key}', flush=True)
                continue
            print(f'[{variant} {i}/{len(questions)}] {key}', flush=True)
            start = time.perf_counter()
            try:
                if pipeline is None:
                    from src.pipeline.video_rag import VideoRAGPipeline
                    pipeline = VideoRAGPipeline()
                rec = evaluate_variant(pipeline, row, args.index_root, variant, args.tolerance)
                print(f"  answer={rec['pred_option']} gold={rec['gold_option']} retrieval_hit={rec['retrieval_hit']} citation_hit={rec['any_citation_in_gt']}", flush=True)
            except Exception as exc:
                rec = dict(video_id=key[0], qid=key[1], type=row['type'], variant=variant,
                           status='error', error=f'{type(exc).__name__}: {exc}')
                print('  ERROR:', rec['error'], flush=True)
            rec['latency_seconds'] = round(time.perf_counter()-start, 3)
            rec['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
            with journal.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False)+'\n')
            existing[key] = rec
        ordered = [existing[k] for k in keys if k in existing]
        outputs[variant] = ordered
        write_csv(args.output_dir / f'{name}.csv', ordered)
    summary = {v: metrics(records, args.tolerance) for v, records in outputs.items()}
    summary['_protocol'] = {'dataset':'NExT-GQA', 'split':'val', 'seed':selection.get('seed'),
        'questions':len(questions), 'tolerance_seconds':args.tolerance,
        'frame_budget':8, 'baseline':'existing M3.4 results, no re-inference',
        'option_query':'all five options included in CLIP retrieval text; 2 anchors + neighbors',
        'coverage':'2 question-only CLIP anchors + farthest-in-time indexed frames',
        'answer_prompt':'unchanged M3.4 five-option MC prompt',
        'citation':'unchanged M3.3 two-stage attribution'}
    path = args.output_dir / f"seed{selection.get('seed','unknown')}_n{len(questions)}_comparison.json"
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Comparison:', path)

if __name__ == '__main__':
    main()
