"""DAVIS 2017 val: first-frame GT box prompted SAM 2.1, official J/F metrics.

Runs all objects jointly, saves label PNGs, and scores non-prompt frames.
The benchmark uses full DAVIS 480p frames and never re-prompts from GT.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


def object_boxes(label_map: np.ndarray) -> dict[int, list[float]]:
    """Return xyxy exclusive boxes for all DAVIS object IDs (0=bg, 255=void)."""
    boxes = {}
    for obj_id in np.unique(label_map):
        obj_id = int(obj_id)
        if obj_id in (0, 255):
            continue
        ys, xs = np.where(label_map == obj_id)
        if len(xs):
            boxes[obj_id] = [float(xs.min()), float(ys.min()),
                             float(xs.max() + 1), float(ys.max() + 1)]
    return boxes


def compose_logits(object_ids, logits, shape):
    """Resolve overlapping SAM masks using positive logit confidence."""
    if len(object_ids) == 0:
        return np.zeros(shape, dtype=np.uint8)
    scores = np.stack([
        logit.squeeze().detach().float().cpu().numpy()
        for logit in logits
    ], axis=0)
    if scores.shape[1:] != tuple(shape):
        raise ValueError(f"Mask dimensions {scores.shape[1:]} != image {shape}")
    winner = np.argmax(scores, axis=0)
    highest = np.max(scores, axis=0)
    result = np.zeros(shape, dtype=np.uint8)
    for i, obj_id in enumerate(object_ids):
        result[(winner == i) & (highest > 0)] = int(obj_id)
    return result


def list_sequences(root: Path, subset: str, limit: int) -> list[str]:
    split = root / 'ImageSets' / '2017' / f'{subset}.txt'
    if not split.is_file():
        raise FileNotFoundError(f"Missing DAVIS split: {split}")
    names = [s.strip() for s in split.read_text().splitlines() if s.strip()]
    return names[:limit] if limit else names


def load_gt(root: Path, sequence: str, stem: str) -> np.ndarray:
    path = root / 'Annotations' / '480p' / sequence / f'{stem}.png'
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path), dtype=np.uint8)


def score_sequence(root: Path, sequence: str, predictions: Path):
    """Official DAVIS J/F implementations; ignore first and last frames as DAVIS does."""
    try:
        from davis2017.metrics import db_eval_iou, db_eval_boundary
    except ImportError as exc:
        raise RuntimeError('Install official davis2017-evaluation package (see README_M410.md)') from exc
    frame_paths = sorted((root / 'JPEGImages' / '480p' / sequence).glob('*.jpg'))
    if len(frame_paths) < 3:
        raise ValueError(f'{sequence}: fewer than 3 frames')
    first_gt = load_gt(root, sequence, frame_paths[0].stem)
    ids = sorted(object_boxes(first_gt))
    rows = []
    for frame in frame_paths[1:-1]:
        gt = load_gt(root, sequence, frame.stem)
        pred_path = predictions / sequence / f'{frame.stem}.png'
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        pred = np.asarray(Image.open(pred_path), dtype=np.uint8)
        if gt.shape != pred.shape:
            raise ValueError(f'{sequence}/{frame.stem}: mask shape mismatch')
        void = gt == 255
        for obj_id in ids:
            j = float(db_eval_iou(gt == obj_id, pred == obj_id, void_pixels=void))
            f = float(db_eval_boundary(gt == obj_id, pred == obj_id, void_pixels=void))
            rows.append({'sequence': sequence, 'frame': frame.stem,
                         'object_id': obj_id, 'J': j, 'F': f, 'JF': (j + f) / 2})
    return rows


def summarize(rows):
    """Macro-average by object, then sequence; never weight long clips more."""
    from collections import defaultdict
    per_object = defaultdict(list)
    for row in rows:
        per_object[(row['sequence'], row['object_id'])].append(row)
    objects = []
    for (sequence, obj_id), records in sorted(per_object.items()):
        objects.append({'sequence': sequence, 'object_id': obj_id,
                        **{key: float(np.mean([r[key] for r in records]))
                           for key in ('J', 'F', 'JF')}})
    if not objects:
        return {'objects': 0, 'frames_scored': 0, 'J': None, 'F': None, 'JF': None}
    return {'objects': len(objects), 'frames_scored': len(rows),
            **{key: float(np.mean([o[key] for o in objects]))
               for key in ('J', 'F', 'JF')}}


def run_sequence(root, sequence, output, predictor, device, overwrite=False):
    import torch
    image_dir = root / 'JPEGImages' / '480p' / sequence
    images = sorted(image_dir.glob('*.jpg'))
    if not images:
        raise FileNotFoundError(image_dir)
    annotation = load_gt(root, sequence, images[0].stem)
    boxes = object_boxes(annotation)
    if not boxes:
        raise ValueError(f'{sequence}: no objects in first GT mask')
    first_size = np.asarray(Image.open(images[0])).shape[:2]
    if annotation.shape != first_size:
        raise ValueError(f'{sequence}: GT/image dimensions differ')
    save_dir = output / 'predictions' / sequence
    if not overwrite and all((save_dir / f'{p.stem}.png').is_file() for p in images):
        return {'sequence': sequence, 'objects': len(boxes), 'frames': len(images),
                'inference_seconds': None, 'cached': True}
    save_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    state = predictor.init_state(video_path=str(image_dir), offload_video_to_cpu=True)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
    written = set()
    try:
        with torch.inference_mode(), context:
            for obj_id, box in boxes.items():
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=obj_id,
                    box=np.asarray(box, dtype=np.float32))
            # DAVIS first-frame annotation is a prompt, not a prediction.
            Image.fromarray(annotation).save(save_dir / f'{images[0].stem}.png')
            written.add(0)
            for frame_idx, ids, logits in predictor.propagate_in_video(state):
                if frame_idx >= len(images):
                    raise ValueError('SAM returned frame index beyond image list')
                if frame_idx == 0:
                    continue
                labels = compose_logits(list(ids), logits, first_size)
                Image.fromarray(labels).save(save_dir / f'{images[frame_idx].stem}.png')
                written.add(frame_idx)
    finally:
        predictor.reset_state(state)
    # Missing outputs are an error, never silently interpreted as background.
    expected = set(range(len(images)))
    if written != expected:
        raise RuntimeError(f'{sequence}: missing SAM outputs: {sorted(expected - written)}')
    return {'sequence': sequence, 'objects': len(boxes), 'frames': len(images),
            'inference_seconds': round(time.perf_counter() - start, 3), 'cached': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--davis-root', type=Path, default=Path('data/datasets/davis2017/DAVIS'))
    parser.add_argument('--subset', default='val', choices=['val'])
    parser.add_argument('--limit', type=int, default=1, help='0=all validation sequences')
    parser.add_argument('--model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output', type=Path, default=Path('outputs/m410_davis_sam2_tiny'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be >= 0')
    sequences = list_sequences(args.davis_root, args.subset, args.limit)
    if not sequences:
        parser.error('No sequences found')
    from sam2.build_sam import build_sam2_video_predictor_hf
    predictor = build_sam2_video_predictor_hf(args.model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    timings = []
    for i, sequence in enumerate(sequences, 1):
        print(f'[{i}/{len(sequences)}] {sequence}', flush=True)
        timing = run_sequence(args.davis_root, sequence, args.output, predictor,
                              args.device, args.overwrite)
        timings.append(timing)
        (args.output / 'timings.json').write_text(json.dumps(timings, indent=2))
    rows = []
    for sequence in sequences:
        rows.extend(score_sequence(args.davis_root, sequence, args.output / 'predictions'))
    with (args.output / 'per_frame.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sequence', 'frame', 'object_id', 'J', 'F', 'JF'])
        writer.writeheader()
        writer.writerows(rows)
    summary = {'model': args.model, 'subset': args.subset,
               'sequence_count': len(sequences), 'sequences': sequences,
               'initialization': 'first-frame GT-derived box per object',
               'prompt_frames_excluded': True, 'last_frames_excluded': True,
               'evaluation': 'official davis2017.metrics J/F',
               'aggregation': 'mean over frames per object, then mean over objects',
               'metrics': summarize(rows), 'timings': timings}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
