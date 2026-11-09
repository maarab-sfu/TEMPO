"""Selectively extract annotated BURST frames from TAO-Amodal LaSOT/HACS ZIPs.

From project root:
  python -m experiments.extract_burst_hard_negative_frames --inspect
  python -m experiments.extract_burst_hard_negative_frames --limit 2
  python -m experiments.extract_burst_hard_negative_frames

Requires: pip install remotezip huggingface_hub requests
Only the five explicitly chosen sequences are downloaded by default.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath

from huggingface_hub import get_token, hf_hub_url
from remotezip import RemoteZip

REPO = 'chengyenhsieh/TAO-Amodal'
SELECTED = {
    'LaSOT': ['volleyball-13', 'person-17', 'sheep-19'],
    'HACS': [
        'Grooming_horse_v_ANnFXWVJ_x0_scene_0_0-5143',
        'Playing_badminton_v_PnA8TaQeCSk_scene_0_0-1002',
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--inspect', action='store_true', help='Inspect remote ZIP paths only; do not download images')
    ap.add_argument('--limit', type=int, default=None, help='Max number of images to extract in total')
    args = ap.parse_args()
    root = args.root.resolve()
    ann_path = root / 'data/annotations/burst/val/all_classes.json'
    with ann_path.open(encoding='utf-8') as f:
        data = json.load(f)
    seqs = {(s['dataset'], s['seq_name']): s for s in data['sequences']}
    token = get_token()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    total_saved = 0
    total_missing = 0
    for dataset, names in SELECTED.items():
        required = {}
        for name in names:
            key = (dataset, name)
            if key not in seqs:
                raise KeyError(f'Missing annotation sequence: {dataset}/{name}')
            s = seqs[key]
            for raw in s['annotated_image_paths']:
                filename = PurePosixPath(raw.replace('\\', '/')).name
                required[(name, filename)] = raw
        archive = f'frames/val/{dataset}.zip'
        url = hf_hub_url(repo_id=REPO, filename=archive, repo_type='dataset')
        print(f'\n{dataset}: {len(names)} sequences, {len(required)} requested frames; remote {archive}', flush=True)
        with RemoteZip(url, headers=headers) as rz:
            infos = [i for i in rz.infolist() if not i.is_dir()]
            print(f'ZIP entries: {len(infos)}; sample paths:', flush=True)
            for i in infos[:6]:
                print(' ', i.filename)
            matches = defaultdict(list)
            wanted = set(names)
            for info in infos:
                p = PurePosixPath(info.filename.replace('\\', '/'))
                parts = set(p.parts[:-1])
                for name in wanted & parts:
                    key = (name, p.name)
                    if key in required:
                        matches[key].append(info)
            unique = {k: values[0] for k, values in matches.items() if len(values) == 1}
            ambiguous = [k for k, values in matches.items() if len(values) != 1]
            missing = sorted(set(required) - set(unique))
            print(f'Matched {len(unique)}/{len(required)}; ambiguous {len(ambiguous)}; missing {len(missing)}')
            if missing:
                print('Missing examples:', missing[:10])
            if ambiguous:
                print('Ambiguous examples:', ambiguous[:10])
            total_missing += len(missing) + len(ambiguous)
            if args.inspect:
                continue
            if not unique:
                print(f'No matches in {dataset}. Skipping; inspect paths before changing matching logic.')
                continue
            outdir = root / 'data/raw/burst/frames/val' / dataset
            for (name, filename), info in sorted(unique.items()):
                if args.limit is not None and total_saved >= args.limit:
                    break
                dest = outdir / name / filename
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
                total_saved += 1
        if args.limit is not None and total_saved >= args.limit:
            break
    print(f'\nExtracted/verified {total_saved} images; unmatched/ambiguous requests: {total_missing}')
    if args.inspect:
        print('Inspection only: no images downloaded.')


if __name__ == '__main__':
    main()
