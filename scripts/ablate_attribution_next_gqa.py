"""M3.6: reattribute frozen M3.4 answers to the exact same retrieved frames.

No retrieval or answer regeneration. Candidate-wise and two-frame window strategies
are evaluated independently and checkpointed per (video_id, qid, variant).
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluate_next_gqa import citation_metrics, load_selection, mc_prompt


def load_baseline(path):
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('status') == 'ok':
            rows[(str(r['video_id']), str(r['qid']))] = r
    return rows


def evidence_frames(index_root, record):
    from src.storage.index_store import IndexStore
    retriever, _ = IndexStore(index_root / f"nextgqa_{record['video_id']}").load()
    by_id = {str(f['frame_id']): f for f in retriever.metadata}
    ids = [str(i) for i in record['retrieved_frame_ids']]
    if len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
        raise ValueError('Baseline retrieved frame IDs missing or duplicated in index')
    frames = [by_id[i] for i in ids]
    baseline_ts = [float(t) for t in record['retrieved_timestamps']]
    if len(frames) != len(baseline_ts) or any(abs(float(f['timestamp'])-t) > .02 for f,t in zip(frames, baseline_ts)):
        raise ValueError('Index metadata timestamps do not match frozen baseline')
    return frames


def select_citations(vlm, frames, question, frozen_answer, variant, max_citations=3):
    """Return cited IDs; calls existing Qwen attribution only, not answer generation."""
    from src.reasoning.vlm import parse_attribution_response
    if variant == 'candidate':
        groups = [[f] for f in frames]
    elif variant == 'window':
        groups = [frames[i:i+2] for i in range(0, len(frames), 2)]
    else:
        raise ValueError(variant)
    cited = []
    valid_parses = 0
    for group in groups:
        raw = vlm.answer_video_frames(group, question, max_new_tokens=96, attribution_answer=frozen_answer)
        parsed = parse_attribution_response(raw, sorted(group, key=lambda f: f['timestamp']), frozen_answer)
        valid_parses += bool(parsed['citation_parse_ok'])
        for frame_id in parsed['cited_frame_ids']:
            if frame_id not in cited:
                cited.append(frame_id)
    # Capping citations prevents candidate-wise attribution from flooding results.
    # This is chronological selection, NOT a confidence ranking.
    return cited[:max_citations], valid_parses, len(groups)


def summary(records):
    ok = [r for r in records if r.get('status') == 'ok']
    n = len(ok)
    citations = sum(r['citation_count'] for r in ok)
    hits = sum(r['citation_hits'] for r in ok)
    # Interval recall: a GT interval counts if >=1 cited frame overlaps it.
    total_intervals = sum(r['gt_interval_count'] for r in ok)
    matched_intervals = sum(r['gt_intervals_hit'] for r in ok)
    return {'completed': n, 'failed': sum(r.get('status')=='error' for r in records),
            'citation_coverage': sum(r['citation_count']>0 for r in ok)/n if n else None,
            'temporal_citation_precision': hits/citations if citations else None,
            'any_temporal_hit_rate': sum(r['any_citation_in_gt'] for r in ok)/n if n else None,
            'joint_correct_and_temporally_grounded_rate': sum(r['correct'] and r['any_citation_in_gt'] for r in ok)/n if n else None,
            'temporal_interval_recall': matched_intervals/total_intervals if total_intervals else None,
            'mean_attribution_latency_seconds': sum(r['attribution_latency_seconds'] for r in ok)/n if n else None,
            'total_attribution_calls': sum(r['attribution_calls'] for r in ok)}


def interval_recall(timestamps, intervals, tolerance):
    hits = sum(any(float(a)-tolerance <= t <= float(b)+tolerance for t in timestamps) for a,b in intervals)
    return hits, len(intervals)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, default=Path('evaluation/next_gqa/results/seed42_n100_tol1.jsonl'))
    p.add_argument('--selection', type=Path, default=Path('data/datasets/next_gqa/manifests/selection.json'))
    p.add_argument('--index-root', type=Path, default=Path('data/indexes'))
    p.add_argument('--output-dir', type=Path, default=Path('evaluation/next_gqa/attribution'))
    p.add_argument('--variant', choices=['candidate','window','both'], default='both')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--tolerance', type=float, default=1.0)
    p.add_argument('--max-citations', type=int, default=3)
    p.add_argument('--retry-errors', action='store_true')
    args=p.parse_args()
    if args.tolerance < 0 or args.max_citations < 1 or args.limit < 0: p.error('Invalid tolerance, max citations, or limit')
    selection, questions = load_selection(args.selection)
    if args.limit: questions = questions[:args.limit]
    baseline=load_baseline(args.baseline)
    variants=['candidate','window'] if args.variant=='both' else [args.variant]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vlm=None
    for variant in variants:
        stem=f"seed{selection.get('seed','unknown')}_n{len(questions)}_{variant}_tol{args.tolerance:g}_cap{args.max_citations}"
        journal=args.output_dir/f'{stem}.jsonl'
        existing={}
        if journal.exists():
            for line in journal.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    rec=json.loads(line)
                    existing[(str(rec['video_id']),str(rec['qid']))]=rec
        for i,row in enumerate(questions,1):
            key=(str(row['video_id']),str(row['qid']))
            if key in existing and (existing[key]['status']=='ok' or not args.retry_errors):
                print(f'[{variant} {i}/{len(questions)}] SKIP {key}',flush=True)
                continue
            print(f'[{variant} {i}/{len(questions)}] {key}',flush=True)
            try:
                base=baseline.get(key)
                if base is None: raise ValueError('Missing successful M3.4 baseline row')
                frames=evidence_frames(args.index_root,base)
                if vlm is None:
                    from src.reasoning.vlm import VisionLanguageModel
                    vlm=VisionLanguageModel()
                start=time.perf_counter()
                ids,valid_count,call_count=select_citations(vlm,frames,mc_prompt(row),base['raw_answer'],variant,args.max_citations)
                latency=time.perf_counter()-start
                cm=citation_metrics(frames,ids,row['evidence_intervals'],args.tolerance)
                intervals_hit,intervals_n=interval_recall(cm['citation_timestamps'],row['evidence_intervals'],args.tolerance)
                rec={'video_id':key[0],'qid':key[1],'type':row['type'],'status':'ok',
                     'variant':variant,'correct':base['correct'],'gold_option':base['gold_option'],
                     'pred_option':base['pred_option'],'raw_answer':base['raw_answer'],
                     'retrieved_frame_ids':base['retrieved_frame_ids'],
                     'retrieved_timestamps':base['retrieved_timestamps'],
                     'evidence_intervals':row['evidence_intervals'],
                     'cited_frame_ids':ids,'attribution_calls':call_count,
                     'valid_attribution_calls':valid_count,'attribution_latency_seconds':round(latency,3),
                     'gt_intervals_hit':intervals_hit,'gt_interval_count':intervals_n,**cm}
                print(f"  citations={len(ids)} hits={cm['citation_hits']} calls={call_count} time={latency:.1f}s",flush=True)
            except Exception as exc:
                rec={'video_id':key[0],'qid':key[1],'variant':variant,'status':'error',
                     'error':f'{type(exc).__name__}: {exc}'}
                print('  ERROR:',rec['error'],flush=True)
            rec['completed_at_utc']=datetime.now(timezone.utc).isoformat()
            with journal.open('a',encoding='utf-8') as f:
                f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            existing[key]=rec
        ordered=[existing[k] for row in questions if (k:=(str(row['video_id']),str(row['qid']))) in existing]
        report=summary(ordered)
        report.update({'variant':variant,'questions':len(questions),'tolerance_seconds':args.tolerance,
                       'max_citations':args.max_citations,'answer_source':'frozen M3.4 baseline',
                       'attribution':'per-frame' if variant=='candidate' else 'nonoverlapping 2-frame windows'})
        (args.output_dir/f'{stem}_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        columns=['video_id','qid','type','status','variant','correct','gold_option','pred_option',
                 'cited_frame_ids','citation_timestamps','citation_count','citation_hits',
                 'citation_precision_temporal','any_citation_in_gt','gt_intervals_hit','gt_interval_count',
                 'attribution_calls','valid_attribution_calls','attribution_latency_seconds','error']
        with (args.output_dir/f'{stem}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
            for rec in ordered:
                w.writerow({c:json.dumps(rec[c]) if isinstance(rec.get(c),(list,dict)) else rec.get(c,'') for c in columns})
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
