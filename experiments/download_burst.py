"""Download BURST annotations and select reproducible validation sequences.

Run from the video-rag project root:
    python -m experiments.download_burst --count 10

Only annotations are downloaded automatically. Official TAO frame archives are
separate and may require downloading through MOTChallenge.
"""
import argparse
import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
import truststore

truststore.inject_into_ssl()

ANNOTATIONS_URL = 'https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip'


def download_annotations(root: Path):
    annotations_dir = root / 'data' / 'annotations' / 'burst'
    annotation_file = annotations_dir / 'val' / 'all_classes.json'
    if annotation_file.exists():
        print(f'Annotations already available: {annotation_file}')
        return annotation_file

    archive = root / 'data' / 'downloads' / 'burst_annotations.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f'Downloading official annotations: {ANNOTATIONS_URL}')
        try:
            req = urllib.request.Request(ANNOTATIONS_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as response, archive.open('wb') as out:
                shutil.copyfileobj(response, out)
        except Exception as exc:
            archive.unlink(missing_ok=True)
            raise RuntimeError('Annotation download failed. Download annotations.zip from the BURST official repository link and place it at ' + str(archive)) from exc

    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f'Invalid ZIP archive: {archive}')

    annotations_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        # Archives may contain an enclosing annotations/ directory.
        for member in zf.infolist():
            if member.is_dir():
                continue
            p = Path(member.filename.replace('\\', '/'))
            parts = p.parts
            if 'annotations' in parts:
                parts = parts[parts.index('annotations') + 1:]
            if not parts or parts[0] not in {'train', 'val', 'test', 'info'}:
                continue
            target = annotations_dir.joinpath(*parts)
            if not target.resolve().is_relative_to(annotations_dir.resolve()):
                raise RuntimeError('Unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)

    if not annotation_file.exists():
        raise RuntimeError(f'Could not find val/all_classes.json after extracting {archive}. Inspect archive layout.')
    return annotation_file


def choose_sequences(annotation_file: Path, count: int, min_tracks: int, min_frames: int):
    with annotation_file.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    sequences = payload['sequences']
    scored = []
    for seq in sequences:
        paths = seq.get('annotated_image_paths', [])
        segments = seq.get('segmentations', [])
        track_ids = seq.get('track_category_ids', {})
        if len(paths) != len(segments):
            continue
        visible = [frame for frame in segments if len(frame) >= 2]
        if len(track_ids) < min_tracks or len(visible) < min_frames:
            continue
        scored.append((len(visible), len(track_ids), seq))
    scored.sort(key=lambda x: (-x[0], -x[1], str(x[2].get('seq_name', ''))))
    selected = [seq for _, _, seq in scored[:count]]
    return selected, len(sequences), len(scored)


def write_manifest(root: Path, selected: list):
    outdir = root / 'data' / 'annotations' / 'burst'
    manifest = outdir / 'selected_val_sequences.csv'
    paths_file = outdir / 'selected_val_image_paths.txt'
    fields = ['id', 'seq_name', 'dataset', 'fps', 'width', 'height', 'num_tracks', 'num_annotated_frames', 'num_multitrack_frames', 'first_image_path']
    with manifest.open('w', newline='', encoding='utf-8') as f, paths_file.open('w', encoding='utf-8') as p:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seq in selected:
            annotated = seq.get('annotated_image_paths', [])
            writer.writerow({
                'id': seq.get('id', ''),
                'seq_name': seq.get('seq_name', ''),
                'dataset': seq.get('dataset', ''),
                'fps': seq.get('fps', ''),
                'width': seq.get('width', ''),
                'height': seq.get('height', ''),
                'num_tracks': len(seq.get('track_category_ids', {})),
                'num_annotated_frames': len(annotated),
                'num_multitrack_frames': sum(len(x) >= 2 for x in seq.get('segmentations', [])),
                'first_image_path': annotated[0] if annotated else '',
            })
            for path in annotated:
                p.write(path + '\n')
    return manifest, paths_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--min-tracks', type=int, default=3)
    parser.add_argument('--min-multitrack-frames', type=int, default=5)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    root = args.root.resolve()
    annotations = download_annotations(root)
    selected, total, eligible = choose_sequences(annotations, args.count, args.min_tracks, args.min_multitrack_frames)
    if not selected:
        raise RuntimeError('No sequences match the selection criteria. Try lowering --min-tracks or --min-multitrack-frames.')
    manifest, paths = write_manifest(root, selected)
    print(f'Validation sequences: {total}; eligible: {eligible}; selected: {len(selected)}')
    print(f'Manifest: {manifest}')
    print(f'Image paths: {paths}')
    print('\nSelected sequences:')
    for seq in selected:
        print(f"  {seq.get('dataset')}/{seq.get('seq_name')}: {len(seq.get('track_category_ids', {}))} tracks, {len(seq.get('annotated_image_paths', []))} annotated frames")
    print('\nNext: acquire the official TAO frame archive(s) containing these sequences. No frames were downloaded by this script.')


if __name__ == '__main__':
    main()
