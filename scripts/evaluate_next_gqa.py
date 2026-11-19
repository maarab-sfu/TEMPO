"""NExT-GQA MC benchmark: question-only retrieval, option-conditioned VLM, cited temporal alignment.

Run: python -m scripts.evaluate_next_gqa --limit 2
     python -m scripts.evaluate_next_gqa
"""
import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OPTIONS = 'ABCDE'


def correct_letter(row):
    target = str(row['answer']).strip()
    matches = [OPTIONS[i] for i in range(5) if target.casefold() == str(row[f'a{i}']).strip().casefold()]
    if len(matches) != 1:
        raise ValueError(f"qid {row['qid']}: answer does not uniquely match options: {target!r}")
    return matches[0]


def parse_choice(answer):
    """Conservative option parsing; ambiguous/unparseable outputs are incorrect."""
    s = answer.strip()
    patterns = [r'^\s*(?:option|choice|answer)\s*[:\-]?\s*\(?([A-E])\)?\s*[.!]?\s*$',
                r'^\s*\(?([A-E])\)?\s*[.!]?\s*$',
                r'^\s*(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?\(?([A-E])\)?\s*[.!]?\s*$']
    for pat in patterns:
        match = re.fullmatch(pat, s, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def citation_metrics(frames, cited_ids, intervals, tolerance=1.0):
    by_id = {str(f['frame_id']): float(f['timestamp']) for f in frames}
    ids = [str(i) for i in cited_ids]
    valid = len(ids) == len(set(ids)) and all(i in by_id for i in ids)
    timestamps = [by_id[i] for i in ids if i in by_id]
    hits = [any(float(a)-tolerance <= t <= float(b)+tolerance for a,b in intervals) for t in timestamps]
    return {'citation_valid': valid, 'citation_count': len(ids),
            'citation_timestamps': timestamps, 'citation_hits': sum(hits),
            'citation_precision_temporal': sum(hits)/len(hits) if hits else None,
            'any_citation_in_gt': any(hits) if hits else False,
            'all_citations_in_gt': all(hits) if hits else False}


def load_selection(path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    rows = obj['questions']
    if not isinstance(rows, list):
        raise ValueError('selection.json questions must be a list')
    return obj, rows


def mc_prompt(row):
    options = '\n'.join(f'{OPTIONS[i]}. {row[f"a{i}"]}' for i in range(5))
    return (f"{row['question']}\n\nChoose the single best answer from the options below.\n"
            f"{options}\n\nReply with ONLY one letter: A, B, C, D, or E.")


def evaluate_one(pipeline, row, index_root, tolerance):
    from src.storage.index_store import IndexStore
    from src.retrieval.temporal_retriever import TemporalRetriever
    from src.reasoning.vlm import VisionLanguageModel
    video_id = str(row['video_id'])
    index_dir = index_root / f'nextgqa_{video_id}'
    if not index_dir.is_dir():
        raise FileNotFoundError(f'Missing index: {index_dir}')
    retriever, _ = IndexStore(index_dir).load()
    # Retrieval uses ONLY the original question, never candidate answer text.
    query_embedding = pipeline.encoder.encode_text([row['question']])
    frames = TemporalRetriever(retriever).search_with_context(
        query_embedding, top_k=2, context_before=2, context_after=2, max_frames=8)
    if not frames:
        raise RuntimeError('No frames retrieved')
    if pipeline.vlm is None:
        pipeline.vlm = VisionLanguageModel()
    result = pipeline.vlm.answer_video_frames_grounded(frames, mc_prompt(row))
    prediction = parse_choice(result['answer'])
    gold = correct_letter(row)
    intervals = row['evidence_intervals']
    cm = citation_metrics(frames, result['cited_frame_ids'], intervals, tolerance)
    return {'video_id':video_id, 'qid':str(row['qid']), 'type':row['type'],
            'question':row['question'], 'gold_option':gold, 'pred_option':prediction,
            'raw_answer':result['answer'], 'correct':prediction == gold,
            'citation_parse_ok':bool(result['citation_parse_ok']),
            'grounding_status':result['grounding_status'],
            'cited_frame_ids':result['cited_frame_ids'],
            'evidence_intervals':intervals,
            'retrieved_timestamps':[float(f['timestamp']) for f in frames],
            'retrieved_frame_ids':[f['frame_id'] for f in frames], **cm}


def summarize(records):
    completed = [r for r in records if r.get('status') == 'ok']
    correct = sum(bool(r['correct']) for r in completed)
    cited = [r for r in completed if r['citation_count'] > 0]
    all_citation_count = sum(r['citation_count'] for r in cited)
    all_hits = sum(r['citation_hits'] for r in cited)
    by_type = {}
    for typ in sorted(set(r['type'] for r in completed)):
        subset = [r for r in completed if r['type'] == typ]
        by_type[typ] = {'n':len(subset), 'correct':sum(r['correct'] for r in subset),
                        'accuracy':sum(r['correct'] for r in subset)/len(subset)}
    latencies = sorted(float(r['latency_seconds']) for r in completed)
    def pct(p):
        if not latencies: return None
        return latencies[round((len(latencies)-1)*p)]
    return {'completed':len(completed), 'failed':sum(r.get('status') == 'error' for r in records),
            'accuracy':correct/len(completed) if completed else None,
            'correct':correct, 'by_type':by_type,
            'citation_coverage':len(cited)/len(completed) if completed else None,
            'citation_valid_rate':sum(r['citation_valid'] for r in completed)/len(completed) if completed else None,
            'temporal_citation_precision':all_hits/all_citation_count if all_citation_count else None,
            'any_temporal_hit_rate':sum(r['any_citation_in_gt'] for r in completed)/len(completed) if completed else None,
            'joint_correct_and_temporally_grounded_rate':sum(r['correct'] and r['any_citation_in_gt'] for r in completed)/len(completed) if completed else None,
            'mean_latency_seconds':sum(latencies)/len(latencies) if latencies else None,
            'p50_latency_seconds':pct(.5), 'p95_latency_seconds':pct(.95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, default=Path('data/datasets/next_gqa/manifests/selection.json'))
    ap.add_argument('--index-root', type=Path, default=Path('data/indexes'))
    ap.add_argument('--output-dir', type=Path, default=Path('evaluation/next_gqa/results'))
    ap.add_argument('--limit', type=int, default=0, help='First N questions, 0 means all')
    ap.add_argument('--tolerance', type=float, default=1.0, help='Temporal match tolerance in seconds')
    ap.add_argument('--retry-errors', action='store_true')
    args = ap.parse_args()
    if args.tolerance < 0: ap.error('tolerance must be nonnegative')
    selection, questions = load_selection(args.selection)
    if args.limit: questions = questions[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Separate outputs for pilot and full benchmark; safe to resume each.
    run_name = f"seed{selection.get('seed','unknown')}_n{len(questions)}_tol{args.tolerance:g}"
    journal = args.output_dir / f'{run_name}.jsonl'
    existing = {}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[(str(rec['video_id']), str(rec['qid']))] = rec
    pipeline = None
    for i, row in enumerate(questions, 1):
        key = (str(row['video_id']), str(row['qid']))
        if key in existing and (existing[key]['status'] == 'ok' or not args.retry_errors):
            print(f'[{i}/{len(questions)}] SKIP {key} ({existing[key]["status"]})', flush=True)
            continue
        print(f'[{i}/{len(questions)}] {key}', flush=True)
        start = time.perf_counter()
        try:
            if pipeline is None:
                from src.pipeline.video_rag import VideoRAGPipeline
                pipeline = VideoRAGPipeline()
            rec = evaluate_one(pipeline, row, args.index_root, args.tolerance)
            rec['status'] = 'ok'
            print(f"  predicted={rec['pred_option']} gold={rec['gold_option']} correct={rec['correct']}", flush=True)
        except Exception as exc:
            rec = {'video_id':key[0], 'qid':key[1], 'type':row['type'], 'status':'error',
                   'error':f'{type(exc).__name__}: {exc}'}
            print(f'  ERROR: {rec["error"]}', flush=True)
        rec['latency_seconds'] = round(time.perf_counter()-start, 3)
        rec['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        with journal.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
        existing[key] = rec
    ordered = [existing[(str(r['video_id']), str(r['qid']))] for r in questions if (str(r['video_id']), str(r['qid'])) in existing]
    csv_path = args.output_dir / f'{run_name}.csv'
    columns = ['video_id','qid','type','status','gold_option','pred_option','correct','raw_answer',
               'citation_parse_ok','citation_valid','citation_count','citation_hits',
               'citation_precision_temporal','any_citation_in_gt','all_citations_in_gt',
               'grounding_status','cited_frame_ids','citation_timestamps','evidence_intervals',
               'retrieved_timestamps','retrieved_frame_ids','latency_seconds','error']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rec in ordered:
            writer.writerow({k:json.dumps(rec[k]) if isinstance(rec.get(k), (list,dict)) else rec.get(k,'') for k in columns})
    summary = summarize(ordered)
    summary.update({'dataset':'NExT-GQA', 'split':'val', 'seed':selection.get('seed'),
                    'selected_questions':len(questions), 'tolerance_seconds':args.tolerance,
                    'sampling_interval_seconds':2.0,
                    'retrieval':'CLIP question-only top_k=2 context_before=2 context_after=2 max_frames=8',
                    'answer_mode':'multiple choice letter, two-stage attribution'})
    summary_path = args.output_dir / f'{run_name}_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Results: {csv_path}\nSummary: {summary_path}\nResume journal: {journal}')


if __name__ == '__main__':
    main()
