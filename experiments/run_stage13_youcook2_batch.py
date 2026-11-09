"""Stage 13: annotation-derived YouCook2 multi-video pilot.

Run from TEMPO repository root. By default evaluates retrieval only.
The generated questions are silver-standard, not human-verified QA.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def make_questions(annotations, limit):
    events = []
    for a in annotations:
        seg = a.get('segment')
        sentence = str(a.get('sentence', '')).strip().rstrip('.')
        if not sentence or not isinstance(seg, (list, tuple)) or len(seg) != 2:
            continue
        start, end = map(float, seg)
        if 0 <= start < end:
            events.append((start, end, sentence, str(a.get('id', len(events)))))
    events.sort(key=lambda e: (e[0], e[1], e[3]))
    questions = []
    for i, (start, end, sentence, eid) in enumerate(events[:limit]):
        questions.append(dict(id=f'loc_{eid}', question=f'When does this action happen: {sentence}? Give an approximate interval in seconds.', answer=f'{start:.1f}–{end:.1f} seconds', evidence_start_seconds=start, evidence_end_seconds=end, question_type='localization'))
        if i + 1 < len(events):
            ns, ne, next_sentence, nid = events[i + 1]
            if ns >= end:
                questions.append(dict(id=f'next_{eid}_{nid}', question=f'What happens after the action: {sentence}?', answer=next_sentence + '.', evidence_start_seconds=ns, evidence_end_seconds=ne, question_type='next_annotated_event', anchor_start_seconds=start, anchor_end_seconds=end))
    return questions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, default=Path('data/annotations/youcook2.json'))
    p.add_argument('--videos', type=Path, default=Path('data/raw/youcook2'))
    p.add_argument('--output', type=Path, default=Path('results/stage13/batch'))
    p.add_argument('--max-videos', type=int, default=5)
    p.add_argument('--max-events', type=int, default=3)
    p.add_argument('--evidence-frames', type=int, default=12)
    p.add_argument('--with-vlm', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if min(args.max_videos, args.max_events, args.evidence_frames) < 1:
        p.error('max-videos, max-events and evidence-frames must be positive')
    database = json.loads(args.annotations.read_text(encoding='utf-8'))['database']
    paths = sorted(v for v in args.videos.glob('*.mp4') if v.stem in database)
    validation = [v for v in paths if database[v.stem].get('subset') == 'validation']
    if validation:
        paths = validation
    paths = paths[:args.max_videos]
    if not paths:
        p.error('No annotated MP4s found')
    qdir, rdir = args.output / 'questions', args.output / 'predictions'
    qdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for video in paths:
        questions = make_questions(database[video.stem].get('annotations', []), args.max_events)
        if not questions:
            continue
        qpath = qdir / (video.stem + '.json')
        qpath.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding='utf-8')
        row = dict(video_id=video.stem, subset=database[video.stem].get('subset'), questions=len(questions), status='prepared')
        manifest.append(row)
        if not args.prepare_only:
            cmd = [sys.executable, '-m', 'experiments.evaluate_video_rag_stage13_event', '--video', str(video), '--questions', str(qpath), '--output', str(rdir), '--evidence-frames', str(args.evidence_frames), '--event-frames', str(args.evidence_frames)]
            if not args.with_vlm:
                cmd.append('--retrieval-only')
            print(f'Running {video.stem} ({len(questions)} questions)', flush=True)
            result = subprocess.run(cmd, check=False)
            row['status'] = 'completed' if result.returncode == 0 else 'failed'
            row['returncode'] = result.returncode
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Manifest: {args.output / "manifest.json"}; videos: {len(manifest)}')
    print('NOTE: silver-standard questions; retrieval-only predictions have no answer-accuracy score.')


if __name__ == '__main__':
    main()
