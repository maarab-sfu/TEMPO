"""Render DAVIS indexed-label predictions as RGB masks, overlays and contact sheets."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [(239, 68, 68), (34, 197, 94), (59, 130, 246), (245, 158, 11),
           (168, 85, 247), (20, 184, 166), (236, 72, 153), (132, 204, 22)]


def colorize(mask):
    """Background remains black; nonzero DAVIS instance IDs receive stable colors."""
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for object_id in np.unique(mask):
        if object_id:
            result[mask == object_id] = PALETTE[(int(object_id) - 1) % len(PALETTE)]
    return result


def overlay(frame, mask, alpha=0.5):
    """Blend only foreground pixels, preserving the original background."""
    rgb = np.asarray(frame.convert('RGB'), dtype=np.uint8).copy()
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f'Frame/mask size mismatch: {rgb.shape[:2]} vs {mask.shape}')
    colored = colorize(mask)
    active = mask != 0
    rgb[active] = np.rint((1 - alpha) * rgb[active].astype(float) +
                         alpha * colored[active].astype(float)).astype(np.uint8)
    return Image.fromarray(rgb)


def make_sheet(items, destination, width=320, columns=4):
    if not items:
        return
    thumb_h = round(items[0][1].height * width / items[0][1].width)
    label_h = 25
    rows = math.ceil(len(items) / columns)
    sheet = Image.new('RGB', (columns * width, rows * (thumb_h + label_h)), '#222222')
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(items):
        x = (i % columns) * width
        y = (i // columns) * (thumb_h + label_h)
        sheet.paste(img.resize((width, thumb_h), Image.Resampling.BILINEAR), (x, y + label_h))
        draw.text((x + 8, y + 5), name, fill='white')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def visualize_sequence(davis_root, predictions, output, sequence, stride=10, max_frames=24,
                       alpha=0.5, sheet_columns=4):
    pred_dir = predictions / sequence
    frames_dir = davis_root / 'JPEGImages' / '480p' / sequence
    if not pred_dir.is_dir():
        raise FileNotFoundError(f'Prediction sequence not found: {pred_dir}')
    if not frames_dir.is_dir():
        raise FileNotFoundError(f'DAVIS frames not found: {frames_dir}')
    masks = sorted(pred_dir.glob('*.png'))[::stride][:max_frames]
    if not masks:
        raise ValueError(f'No PNG masks in {pred_dir}')
    mask_out = output / 'colored_masks' / sequence
    overlay_out = output / 'overlays' / sequence
    mask_out.mkdir(parents=True, exist_ok=True)
    overlay_out.mkdir(parents=True, exist_ok=True)
    sheet_items = []
    object_ids = set()
    for mask_file in masks:
        frame_file = frames_dir / (mask_file.stem + '.jpg')
        if not frame_file.is_file():
            raise FileNotFoundError(f'Corresponding frame missing: {frame_file}')
        with Image.open(mask_file) as src:
            mask = np.asarray(src).copy()
        if mask.ndim != 2:
            raise ValueError(f'Expected indexed single-channel mask: {mask_file} ({mask.shape})')
        object_ids.update(int(i) for i in np.unique(mask) if i != 0)
        with Image.open(frame_file) as frame:
            frame = frame.convert('RGB')
            colored = Image.fromarray(colorize(mask))
            blended = overlay(frame, mask, alpha)
        colored.save(mask_out / mask_file.name)
        blended.save(overlay_out / mask_file.name)
        sheet_items.append((mask_file.stem, blended))
    make_sheet(sheet_items, output / 'contact_sheets' / f'{sequence}.jpg', columns=sheet_columns)
    return {'sequence': sequence, 'frames_rendered': len(masks),
            'object_ids_seen': sorted(object_ids)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--davis-root', type=Path, default=Path('data/datasets/davis2017/DAVIS'))
    p.add_argument('--predictions', type=Path,
                   default=Path('outputs/m411_davis_detector_sam2/predictions'))
    p.add_argument('--output', type=Path, default=Path('outputs/m411_davis_detector_sam2/visualizations'))
    p.add_argument('--sequences', nargs='*', help='Sequence names; defaults to all prediction folders')
    p.add_argument('--stride', type=int, default=10, help='Render every Nth mask')
    p.add_argument('--max-frames', type=int, default=24, help='Max sampled frames per sequence')
    p.add_argument('--alpha', type=float, default=0.5, help='Mask overlay opacity')
    p.add_argument('--columns', type=int, default=4)
    args = p.parse_args()
    if args.stride < 1 or args.max_frames < 1 or args.columns < 1 or not 0 <= args.alpha <= 1:
        p.error('stride, max-frames, columns must be positive; alpha must be in [0,1]')
    if not args.predictions.is_dir():
        p.error(f'Predictions directory does not exist: {args.predictions}')
    sequences = args.sequences or sorted(p.name for p in args.predictions.iterdir() if p.is_dir())
    for sequence in sequences:
        result = visualize_sequence(args.davis_root, args.predictions, args.output,
                                    sequence, args.stride, args.max_frames,
                                    args.alpha, args.columns)
        print(f"{sequence}: {result['frames_rendered']} frames, object IDs {result['object_ids_seen']}")
    print(f'Visualizations saved to {args.output}')


if __name__ == '__main__':
    main()
