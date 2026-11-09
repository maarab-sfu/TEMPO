"""Load and visualize BURST ground-truth instance masks.

Run from the project root:
    python -m experiments.burst_ground_truth --sequence DHVLP --max-frames 8
    python -m experiments.burst_ground_truth --all --max-frames 3

Requires: pip install pycocotools pillow numpy
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / 'data/annotations/burst/val/all_classes.json'
DEFAULT_FRAMES = ROOT / 'data/raw/burst/frames/val/Charades'
DEFAULT_OUTPUT = ROOT / 'results/burst/ground_truth'


def load_sequences(annotation_path: Path) -> tuple[dict, list]:
    with annotation_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    categories = {int(x['id']): x['name'] for x in data['categories']}
    sequences = [s for s in data['sequences'] if s.get('dataset', '').lower() == 'charades']
    return categories, sequences


def decode_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    rle = annotation['rle']
    if isinstance(rle, str):
        encoded = {'size': [height, width], 'counts': rle.encode('utf-8')}
    elif isinstance(rle, dict):
        encoded = dict(rle)
        if isinstance(encoded.get('counts'), str):
            encoded['counts'] = encoded['counts'].encode('utf-8')
    else:
        raise TypeError(f'Unexpected RLE type: {type(rle)}')
    mask = mask_utils.decode(encoded)
    if mask.ndim != 2 or mask.shape != (height, width):
        raise ValueError(f'Mask shape {mask.shape} != annotation dimensions {(height, width)}')
    return mask.astype(bool)


def find_frame(frames_root: Path, seq_name: str, annotated_path: str) -> Path:
    # Archive was extracted into Charades/<sequence>/<filename>.
    image_name = Path(annotated_path.replace('\\', '/')).name
    return frames_root / seq_name / image_name


def color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id + 12345)
    return tuple(int(x) for x in rng.integers(65, 240, size=3))


def visualize(image: Image.Image, objects: list[dict]) -> Image.Image:
    original = np.asarray(image.convert('RGB'), dtype=np.uint8)
    blended = original.astype(np.float32).copy()
    for obj in objects:
        blended[obj['mask']] = (0.58 * blended[obj['mask']] +
                                0.42 * np.array(color_for_id(obj['track_id'])))
    out = Image.fromarray(np.uint8(np.clip(blended, 0, 255)))
    draw = ImageDraw.Draw(out)
    for obj in objects:
        ys, xs = np.where(obj['mask'])
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = map(int, (xs.min(), ys.min(), xs.max(), ys.max()))
        color = color_for_id(obj['track_id'])
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"ID {obj['track_id']} | {obj['category']}"
        bbox = draw.textbbox((0, 0), label)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y0 - text_h - 8)
        draw.rectangle((x0, label_y, min(out.width - 1, x0 + text_w + 8),
                        label_y + text_h + 6), fill=(0, 0, 0))
        draw.text((x0 + 4, label_y + 3), label, fill=(255, 255, 255))
    return out


def process_sequence(sequence: dict, categories: dict, frames_root: Path,
                     output_root: Path, max_frames: int | None) -> dict:
    seq_name = sequence['seq_name']
    paths = sequence['annotated_image_paths']
    segmentations = sequence['segmentations']
    if len(paths) != len(segmentations):
        raise ValueError(f'{seq_name}: {len(paths)} annotated paths but {len(segmentations)} segmentation entries')

    h, w = int(sequence['height']), int(sequence['width'])
    category_by_track = {int(k): int(v) for k, v in sequence['track_category_ids'].items()}
    output_dir = output_root / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    n_masks = 0
    n_empty = 0
    n_processed = 0
    for frame_idx, (image_path, frame_masks) in enumerate(zip(paths, segmentations)):
        if max_frames is not None and n_processed >= max_frames:
            break
        local_image = find_frame(frames_root, seq_name, image_path)
        if not local_image.is_file():
            raise FileNotFoundError(f'Missing annotated frame: {local_image}')
        with Image.open(local_image) as im:
            image = im.convert('RGB')
        if image.size != (w, h):
            raise ValueError(f'{local_image}: image {image.size}, annotation {(w, h)}')
        objects = []
        for raw_id, annotation in frame_masks.items():
            track_id = int(raw_id)
            mask = decode_mask(annotation, h, w)
            if not mask.any():
                n_empty += 1
                continue
            cat_id = category_by_track.get(track_id)
            objects.append({'track_id': track_id, 'category': categories.get(cat_id, f'category_{cat_id}'), 'mask': mask})
            counts[track_id] += 1
            n_masks += 1
        overlay = visualize(image, objects)
        overlay.save(output_dir / f'{Path(local_image).stem}_overlay.jpg', quality=92)
        n_processed += 1
    return {'sequence': seq_name, 'frames_visualized': n_processed,
            'decoded_nonempty_masks': n_masks, 'empty_masks': n_empty,
            'distinct_visible_track_ids': len(counts),
            'track_ids': sorted(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='DHVLP', help='BURST Charades sequence ID')
    parser.add_argument('--all', action='store_true', help='Process all eight downloaded Charades sequences')
    parser.add_argument('--max-frames', type=int, default=8,
                        help='Frames per sequence; use 0 for all annotated frames')
    parser.add_argument('--annotations', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--frames', type=Path, default=DEFAULT_FRAMES)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    categories, sequences = load_sequences(args.annotations)
    available = {s['seq_name']: s for s in sequences}
    if args.all:
        chosen = [s for s in sequences if (args.frames / s['seq_name']).is_dir()]
    else:
        if args.sequence not in available:
            raise ValueError(f'Unknown Charades sequence: {args.sequence}')
        chosen = [available[args.sequence]]
    if not chosen:
        raise ValueError('No matching downloaded Charades sequences found')
    limit = args.max_frames or None
    summaries = []
    for sequence in chosen:
        result = process_sequence(sequence, categories, args.frames, args.output, limit)
        summaries.append(result)
        print(f"{result['sequence']}: {result['frames_visualized']} frames, "
              f"{result['decoded_nonempty_masks']} masks, "
              f"{result['distinct_visible_track_ids']} track IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'validation_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)
    print('Saved overlays to:', args.output)
    print('Saved summary to:', args.output / 'validation_summary.json')


if __name__ == '__main__':
    main()
