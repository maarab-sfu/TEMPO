"""Select a reproducible NExT-GQA grounded validation subset; no downloads."""
import argparse
import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', type=Path, default=Path('data/datasets/next_gqa/annotations'))
    p.add_argument('--output', type=Path, default=Path('data/datasets/next_gqa/manifests'))
    p.add_argument('--videos', type=int, default=20)
    p.add_argument('--questions', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    with (args.annotations / 'val.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    ground = json.loads((args.annotations / 'gsub_val.json').read_text(encoding='utf-8'))
    mapping = json.loads((args.annotations / 'map_vid_vidorID.json').read_text(encoding='utf-8'))
    by_video = defaultdict(list)
    for row in rows:
        vid = str(row['video_id']).strip()
        qid = str(row['qid']).strip()
        info = ground.get(vid, {})
        intervals = info.get('location', {}).get(qid, [])
        if not intervals or vid not in mapping:
            continue
        if not all(isinstance(i, list) and len(i) == 2 and i[1] >= i[0] for i in intervals):
            continue
        row = dict(row)
        row['evidence_intervals'] = intervals
        row['source_path'] = str(mapping[vid])
        row['duration_seconds'] = info.get('duration')
        by_video[vid].append(row)
    rng = random.Random(args.seed)
    vids = list(by_video)
    rng.shuffle(vids)
    # Favor videos with enough annotated questions while retaining seeded diversity.
    vids = sorted(vids, key=lambda vid: -len(by_video[vid]))[:args.videos]
    rng.shuffle(vids)
    selected = []
    queues = {v: list(by_video[v]) for v in vids}
    for v in vids:
        rng.shuffle(queues[v])
    while len(selected) < args.questions:
        progressed = False
        for vid in vids:
            if len(selected) >= args.questions:
                break
            if queues[vid]:
                selected.append(queues[vid].pop())
                progressed = True
        if not progressed:
            break
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': 'NExT-GQA', 'split': 'val', 'seed': args.seed,
                'requested_videos': args.videos, 'requested_questions': args.questions,
                'selected_video_count': len({r['video_id'] for r in selected}),
                'selected_question_count': len(selected), 'questions': selected}
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    fields = ['video_id', 'qid', 'type', 'question', 'answer', 'a0', 'a1', 'a2', 'a3', 'a4', 'source_path', 'duration_seconds', 'evidence_intervals']
    with (args.output / 'questions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in selected:
            w.writerow({k: json.dumps(r[k]) if k == 'evidence_intervals' else r.get(k, '') for k in fields})
    unique = sorted({(r['video_id'], r['source_path']) for r in selected})
    with (args.output / 'required_videos.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video_id', 'archive_relative_path'])
        w.writerows(unique)
    print(f'Eligible grounded videos: {len(by_video)}; eligible questions: {sum(map(len, by_video.values()))}')
    print(f'Selected: {len(unique)} videos, {len(selected)} questions')
    print('Question types:', dict(Counter(r['type'] for r in selected)))
    print('Saved:', args.output.resolve())
    if len(selected) < args.questions:
        print('NOTE: fewer eligible questions than requested; increase --videos or reduce --questions.')

if __name__ == '__main__':
    main()
