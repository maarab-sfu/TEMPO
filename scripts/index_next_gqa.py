"""Index the selected NExT-GQA videos with VideoMind's existing pipeline.

Run: python -m scripts.index_next_gqa
"""
import argparse
import csv
import json
from pathlib import Path


def read_manifest(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'video_id' not in rows[0]:
        raise ValueError(f'Missing video_id column or empty manifest: {path}')
    ids = [row['video_id'].strip() for row in rows]
    if any(not video_id.isdigit() for video_id in ids):
        raise ValueError('Manifest contains a nonnumeric or blank video_id')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate video_id in manifest')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/datasets/next_gqa'))
    parser.add_argument('--index-root', type=Path, default=Path('data/indexes'))
    parser.add_argument('--sampling-interval', type=float, default=2.0)
    parser.add_argument('--max-resolution', type=int, default=640)
    parser.add_argument('--limit', type=int, default=None, help='Index first N videos for a smoke test')
    args = parser.parse_args()
    if args.sampling_interval <= 0 or args.max_resolution <= 0:
        parser.error('Sampling interval and resolution must be positive')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    manifest = args.root / 'manifests' / 'required_videos.csv'
    video_ids = read_manifest(manifest)
    if args.limit is not None:
        video_ids = video_ids[:args.limit]

    from src.pipeline.video_rag import VideoRAGPipeline
    pipeline = None
    status = {'indexed': [], 'skipped': [], 'missing': [], 'failed': []}
    for n, video_id in enumerate(video_ids, 1):
        source = args.root / 'videos' / f'{video_id}.mp4'
        target = args.index_root / f'nextgqa_{video_id}'
        print(f'[{n}/{len(video_ids)}] {video_id}', flush=True)
        if target.is_dir():
            print('  SKIP: index directory already exists (not revalidated)')
            status['skipped'].append(video_id)
            continue
        if target.exists():
            print('  ERROR: index path exists but is not a directory')
            status['failed'].append({'video_id': video_id, 'error': 'index path is not a directory'})
            continue
        if not source.is_file():
            print(f'  MISSING: {source}')
            status['missing'].append(video_id)
            continue
        if pipeline is None:
            pipeline = VideoRAGPipeline()
        try:
            result = pipeline.index_video(source, target, sampling_interval=args.sampling_interval,
                                          max_resolution=args.max_resolution)
            print(f"  OK: {result['frame_count']} frames")
            status['indexed'].append(video_id)
        except Exception as exc:
            print(f'  FAILED: {type(exc).__name__}: {exc}')
            status['failed'].append({'video_id': video_id, 'error': str(exc)})

    print('\nSummary:', json.dumps({k: len(v) for k, v in status.items()}))
    if status['failed'] or status['missing']:
        print(json.dumps({'missing': status['missing'], 'failed': status['failed']}, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
