"""M3.7: zero-inference selective citation policy analysis of frozen M3.6 CSVs.

No confidence scores were saved by M3.6. Policies operate on recorded citation order,
which is chronological, NOT model confidence. GT-based oracle is diagnostic only.
"""
import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path


def decode_list(value):
    if isinstance(value, list):
        return value
    if value is None or not str(value).strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f'Expected list: {value!r}')
    return parsed


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    records = {}
    for row in rows:
        if row['status'] != 'ok':
            raise ValueError(f'Non-success row in {path}: {row.get("qid")}')
        key = (str(row['video_id']), str(row['qid']))
        if key in records:
            raise ValueError(f'Duplicate question {key} in {path}')
        ids = decode_list(row['cited_frame_ids'])
        timestamps = [float(x) for x in decode_list(row['citation_timestamps'])]
        if len(ids) != len(timestamps):
            raise ValueError(f'Mismatched citation IDs/timestamps for {key}')
        records[key] = {'key': key, 'correct': str(row['correct']).lower() == 'true',
                        'ids': ids, 'timestamps': timestamps}
    return records


def load_ground_truth(path):
    gt = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('status') != 'ok':
                continue
            key = (str(row['video_id']), str(row['qid']))
            if key in gt:
                raise ValueError(f'Duplicate baseline question {key}')
            gt[key] = {'intervals': row['evidence_intervals'], 'correct': bool(row['correct']),
                       'ids': row['cited_frame_ids'], 'timestamps': row['citation_timestamps']}
    return gt


def is_hit(timestamp, intervals, tolerance):
    return any(float(start)-tolerance <= float(timestamp) <= float(end)+tolerance
               for start, end in intervals)


def choose(record, policy, intervals, tolerance):
    pairs = list(zip(record['ids'], record['timestamps']))
    if policy == 'first1': return pairs[:1]
    if policy == 'first2': return pairs[:2]
    if policy == 'last1': return pairs[-1:]
    if policy == 'last2': return pairs[-2:]
    if policy == 'all3': return pairs
    if policy == 'oracle1':
        # Diagnostic upper bound: uses GT, NEVER deploy or compare as a valid method.
        return [next((p for p in pairs if is_hit(p[1], intervals, tolerance)), pairs[0])] if pairs else []
    raise ValueError(policy)


def evaluate(records, ground_truth, policy, tolerance):
    if set(records) != set(ground_truth):
        missing = sorted(set(ground_truth) - set(records))
        extra = sorted(set(records) - set(ground_truth))
        raise ValueError(f'Question key mismatch: missing={missing[:5]}, extra={extra[:5]}')
    output = []
    for key in sorted(records):
        row, gt = records[key], ground_truth[key]
        if row['correct'] != gt['correct']:
            raise ValueError(f'Answer correctness mismatch for {key}')
        selected = choose(row, policy, gt['intervals'], tolerance)
        hits = sum(is_hit(ts, gt['intervals'], tolerance) for _, ts in selected)
        intervals_hit = sum(any(float(start)-tolerance <= ts <= float(end)+tolerance
                                for _, ts in selected) for start, end in gt['intervals'])
        output.append({'video_id': key[0], 'qid': key[1], 'correct': row['correct'],
                       'cited_frame_ids': [fid for fid, _ in selected],
                       'citation_timestamps': [ts for _, ts in selected],
                       'citation_count': len(selected), 'citation_hits': hits,
                       'any_temporal_hit': hits > 0, 'gt_interval_count': len(gt['intervals']),
                       'gt_intervals_hit': intervals_hit})
    n = len(output)
    total_citations = sum(r['citation_count'] for r in output)
    total_hits = sum(r['citation_hits'] for r in output)
    total_intervals = sum(r['gt_interval_count'] for r in output)
    report = {'n': n, 'citation_coverage': sum(r['citation_count'] > 0 for r in output)/n if n else None,
              'temporal_citation_precision': total_hits/total_citations if total_citations else None,
              'any_temporal_hit_rate': sum(r['any_temporal_hit'] for r in output)/n if n else None,
              'joint_correct_and_temporally_grounded_rate': sum(r['correct'] and r['any_temporal_hit'] for r in output)/n if n else None,
              'temporal_interval_recall': sum(r['gt_intervals_hit'] for r in output)/total_intervals if total_intervals else None,
              'mean_citations_per_question': total_citations/n if n else None,
              'citation_count': total_citations, 'citation_hits': total_hits}
    return report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('evaluation/next_gqa/results/seed42_n100_tol1.jsonl'))
    parser.add_argument('--candidate', type=Path, default=Path('evaluation/next_gqa/attribution/seed42_n100_candidate_tol1_cap3.csv'))
    parser.add_argument('--window', type=Path, default=Path('evaluation/next_gqa/attribution/seed42_n100_window_tol1_cap3.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation/next_gqa/selective_citations'))
    parser.add_argument('--tolerance', type=float, default=1.0)
    args = parser.parse_args()
    if args.tolerance < 0: parser.error('Tolerance must be nonnegative')
    gt = load_ground_truth(args.baseline)
    if not gt: parser.error('No successful baseline rows')
    variants = {'baseline': {key: {'correct': r['correct'], 'ids': r['ids'], 'timestamps': r['timestamps']} for key, r in gt.items()},
                'candidate': read_rows(args.candidate), 'window': read_rows(args.window)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant, records in variants.items():
        results[variant] = {}
        policies = ['all3'] if variant == 'baseline' else ['all3', 'first1', 'first2', 'last1', 'last2', 'oracle1']
        for policy in policies:
            metrics, rows = evaluate(records, gt, policy, args.tolerance)
            results[variant][policy] = metrics
            path = args.output_dir / f'{variant}_{policy}.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as handle:
                columns = list(rows[0]) if rows else []
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k,v in row.items()})
    report = {'protocol': {'tolerance_seconds': args.tolerance, 'n_questions': len(gt),
                           'answer_source': 'frozen M3.4', 'inference_calls': 0,
                           'warning': 'M3.6 citations are chronological, not confidence-ranked; oracle1 uses ground truth and is NOT deployable.'},
              'results': results}
    path = args.output_dir / 'comparison.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {path}')
    for variant, policies in results.items():
        for policy, metrics in policies.items():
            print(f'{variant:10s} {policy:8s} precision={metrics["temporal_citation_precision"]:.3f} '
                  f'hit={metrics["any_temporal_hit_rate"]:.3f} '
                  f'joint={metrics["joint_correct_and_temporally_grounded_rate"]:.3f} '
                  f'citations/q={metrics["mean_citations_per_question"]:.2f}')


if __name__ == '__main__':
    main()
