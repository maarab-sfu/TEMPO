"""M3.4: Collect grounded VideoMind answers from an existing running API.

Run from repository root: python -m scripts.evaluate_grounding --questions evaluation/grounding_questions.csv
No model changes, no reindexing, and no third-party dependencies.
"""
import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

QUESTION_FIELDS = ('question_id', 'video_id', 'question', 'category', 'reference_answer', 'reference_notes')
REVIEW_FIELDS = ('answer_correct', 'citation_support', 'review_notes')
RESULT_FIELDS = QUESTION_FIELDS + ('answer', 'grounding_status', 'citation_parse_ok',
    'cited_frame_ids', 'retrieved_frame_ids', 'retrieved_timestamps', 'evidence_json',
    'citation_valid', 'latency_seconds', 'error') + REVIEW_FIELDS


def validate_questions(rows):
    seen = set()
    for line, row in enumerate(rows, 2):
        for key in ('question_id', 'video_id', 'question'):
            if not row.get(key, '').strip():
                raise ValueError(f'Line {line}: missing {key}')
        if row['question_id'] in seen:
            raise ValueError(f"Line {line}: duplicate question_id {row['question_id']}")
        seen.add(row['question_id'])
    return rows


def citation_validity(cited, frames):
    """None if no citations; otherwise True only if all IDs belong to retrieved frames."""
    if not cited:
        return None
    ids = {f['frame_id'] for f in frames}
    return all(type(c) is int and c in ids for c in cited)


def request_json(url, payload=None, timeout=240):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=body,
        headers={'Content-Type': 'application/json'} if body is not None else {},
        method='POST' if body is not None else 'GET')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def collect(rows, base_url, timeout, out_path):
    base_url = base_url.rstrip('/')
    # Fail before starting if the API is unreachable.
    request_json(base_url + '/health', timeout=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for number, question in enumerate(rows, 1):
            started = time.perf_counter()
            row = {key: question.get(key, '') for key in QUESTION_FIELDS}
            row.update({key: '' for key in RESULT_FIELDS if key not in row})
            try:
                result = request_json(
                    base_url + '/api/videos/' + question['video_id'] + '/query',
                    {'question': question['question'], 'top_k': 2,
                     'context_before': 2, 'context_after': 2}, timeout=timeout)
                frames = result.get('frames', [])
                cited = result.get('cited_frame_ids', [])
                valid = citation_validity(cited, frames)
                row.update(answer=result.get('answer', ''),
                    grounding_status=result.get('grounding_status', ''),
                    citation_parse_ok=result.get('citation_parse_ok', ''),
                    cited_frame_ids=json.dumps(cited),
                    retrieved_frame_ids=json.dumps([f['frame_id'] for f in frames]),
                    retrieved_timestamps=json.dumps([f['timestamp'] for f in frames]),
                    evidence_json=json.dumps(frames, ensure_ascii=False),
                    citation_valid='' if valid is None else str(valid))
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                row['error'] = str(exc)
            row['latency_seconds'] = round(time.perf_counter() - started, 3)
            writer.writerow(row)
            handle.flush()
            print(f"[{number}/{len(rows)}] {question['question_id']}: "
                  f"{'ERROR ' + row['error'] if row['error'] else row['grounding_status']}")
    print(f'Wrote {out_path}')


def score(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty results file')
    valid_answers = [r for r in rows if not r.get('error')]
    reviewed = [r for r in valid_answers if r.get('answer_correct', '').strip() in ('0', '1')
                and r.get('citation_support', '').strip() in ('0', '1', 'NA')]
    print(f'Questions: {len(rows)}; successful: {len(valid_answers)}; fully reviewed: {len(reviewed)}')
    if not reviewed:
        print('No completed human reviews yet. Fill answer_correct and citation_support in the CSV.')
        return
    accuracy = sum(r['answer_correct'].strip() == '1' for r in reviewed) / len(reviewed)
    cited = [r for r in reviewed if r.get('cited_frame_ids', '').strip() not in ('', '[]')]
    support = [r for r in cited if r['citation_support'].strip() in ('0', '1')]
    print(f'Human answer accuracy: {accuracy:.1%} ({len(reviewed)} reviewed)')
    print(f'Citation coverage: {len(cited)/len(reviewed):.1%} ({len(cited)}/{len(reviewed)})')
    if cited:
        print('Citation validity (among cited): '
              f"{sum(r['citation_valid'] == 'True' for r in cited)/len(cited):.1%}")
    if support:
        print(f"Human citation support (among assessed cited answers): "
              f"{sum(r['citation_support'].strip() == '1' for r in support)/len(support):.1%} "
              f"({len(support)} assessed)")
    else:
        print('No cited answers assessed for visual support.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=Path('evaluation/grounding_questions.csv'))
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--score', type=Path, help='Score a manually reviewed result CSV instead of collecting')
    args = parser.parse_args()
    if args.score:
        score(args.score)
        return
    with args.questions.open(encoding='utf-8-sig', newline='') as handle:
        rows = validate_questions(list(csv.DictReader(handle)))
    if not rows:
        raise ValueError('Question CSV has no rows; add real questions with existing video IDs')
    filename = datetime.now(timezone.utc).strftime('grounding_%Y%m%d_%H%M%S_utc.csv')
    collect(rows, args.base_url, args.timeout, args.output or Path('evaluation/results') / filename)

if __name__ == '__main__':
    main()
