"""Selectively fetch annotated Charades frames from TAO-Amodal's remote ZIP.

Install: pip install remotezip huggingface_hub requests
Run: python -m experiments.extract_burst_charades_remote --inspect
Then: python -m experiments.extract_burst_charades_remote --limit 2
Finally: python -m experiments.extract_burst_charades_remote
"""
import argparse
import csv
import json
import shutil
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_url, get_token
from remotezip import RemoteZip

REPO = 'chengyenhsieh/TAO-Amodal'
ARCHIVE = 'frames/val/Charades.zip'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--inspect', action='store_true', help='Only inspect ZIP filenames and match selected frames; download no images')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of image files to extract (for testing)')
    args = parser.parse_args()
    root = args.root.resolve()
    annotation_dir = root / 'data' / 'annotations' / 'burst'
    manifest = annotation_dir / 'selected_val_sequences.csv'
    annotation_file = annotation_dir / 'val' / 'all_classes.json'
    if not manifest.is_file() or not annotation_file.is_file():
        raise FileNotFoundError('Run experiments.download_burst first')

    with manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    wanted = {r['seq_name'] for r in rows if r.get('dataset', '').lower() == 'charades'}
    if not wanted:
        raise RuntimeError('Manifest contains no Charades sequences')
    with annotation_file.open(encoding='utf-8') as f:
        sequences = json.load(f)['sequences']
    selected = [s for s in sequences if s.get('dataset', '').lower() == 'charades' and s.get('seq_name') in wanted]
    # Match only the annotated frame filenames, not every frame in a sequence.
    required = {}
    for seq in selected:
        sid = seq['seq_name']
        for raw_path in seq.get('annotated_image_paths', []):
            p = PurePosixPath(raw_path.replace('\\', '/'))
            required[(sid, p.name)] = raw_path
    print(f'Selected Charades sequences: {sorted(wanted)}')
    print(f'Annotated frame requests: {len(required)}')

    url = hf_hub_url(repo_id=REPO, filename=ARCHIVE, repo_type='dataset')
    token = get_token()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    outdir = root / 'data' / 'raw' / 'burst' / 'frames' / 'val' / 'Charades'

    with RemoteZip(url, headers=headers) as rz:
        infos = [i for i in rz.infolist() if not i.is_dir()]
        print(f'ZIP entries: {len(infos)}')
        print('First 8 archive entries:')
        for info in infos[:8]:
            print(' ', info.filename)

        matches = {}
        for info in infos:
            p = PurePosixPath(info.filename.replace('\\', '/'))
            parts = p.parts
            for sid in wanted:
                if sid in parts and (sid, p.name) in required:
                    key = (sid, p.name)
                    matches.setdefault(key, []).append(info)
        ambiguous = {k: v for k, v in matches.items() if len(v) != 1}
        if ambiguous:
            print(f'WARNING: {len(ambiguous)} ambiguous matches; they will be skipped.')
        unique = {k: v[0] for k, v in matches.items() if len(v) == 1}
        print(f'Unambiguous matching frames: {len(unique)} / {len(required)}')
        missing = list(required.keys() - unique.keys())
        if missing:
            print('First 10 unmatched (sequence, filename):', missing[:10])
        if args.inspect:
            print('Inspection only: no images downloaded.')
            return
        if not unique:
            raise RuntimeError('No matching frames. Inspect archive paths and annotation paths before downloading.')

        done = 0
        for (sid, name), info in sorted(unique.items()):
            if args.limit is not None and done >= args.limit:
                break
            dest = outdir / sid / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == info.file_size:
                print('Exists:', dest)
            else:
                tmp = dest.with_suffix(dest.suffix + '.part')
                try:
                    with rz.open(info.filename) as src, tmp.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    if tmp.stat().st_size != info.file_size:
                        raise IOError(f'Incorrect size for {info.filename}')
                    tmp.replace(dest)
                finally:
                    tmp.unlink(missing_ok=True)
                print('Saved:', dest)
            done += 1
        print(f'Extracted/verified {done} annotated frames to {outdir}')
        print('Note: paths were flattened to Charades/<sequence_id>/<image_name>.')


if __name__ == '__main__':
    main()
