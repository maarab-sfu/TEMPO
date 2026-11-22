"""M4.11: detector-box initialization for SAM 2.1 on DAVIS 2017.

Ground truth is used ONLY for oracle assignment of detected boxes to object IDs
and evaluation, never to initialize SAM. Missing IDs produce background masks.
This is NOT an autonomous identity-association benchmark.
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

from scripts.benchmark_davis_sam2 import (
    compose_logits, list_sequences, load_gt, object_boxes, score_sequence, summarize,
)

# Sequence-level text queries are manually specified. They are not inferred from
# DAVIS pixel masks. Edit this dictionary to test other text-query policies.
PROMPTS = {
    'bike-packing': 'person . bicycle . backpack .',
    'blackswan': 'black swan . bird .',
    'bmx-trees': 'person . bicycle .',
    'breakdance': 'person . dancer .',
    'camel': 'camel .',
    'car-roundabout': 'car . vehicle .',
    'car-shadow': 'car . vehicle .',
    'cows': 'cow . cattle .',
    'dance-twirl': 'person . dancer .',
    'dog': 'dog .',
    'dogs-jump': 'dog .',
    'drift-chicane': 'car . racing car .',
    'drift-straight': 'car . racing car .',
    'goat': 'goat .',
    'gold-fish': 'fish . goldfish .',
    'horsejump-high': 'horse . rider . person .',
    'india': 'person . motorcycle . vehicle .',
    'judo': 'person .',
    'kite-surf': 'person . kite . surfboard .',
    'lab-coat': 'person . lab coat .',
    'libby': 'dog .',
    'loading': 'person . vehicle .',
    'mbike-trick': 'person . motorcycle .',
    'motocross-jump': 'person . motorcycle .',
    'paragliding-launch': 'person . paraglider . parachute .',
    'parkour': 'person .',
    'pigs': 'pig .',
    'scooter-black': 'person . scooter . motorcycle .',
    'shooting': 'person .',
    'soapbox': 'person . cart . vehicle .',
}


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def assign_detections(gt_boxes, candidates, min_iou):
    """One-to-one GT/detection matching for evaluation only; no GT prompt leakage."""
    from scipy.optimize import linear_sum_assignment
    ids = sorted(gt_boxes)
    if not ids or not candidates:
        return {}, {obj_id: 0.0 for obj_id in ids}
    similarities = np.asarray([
        [box_iou(gt_boxes[obj_id], candidate['box']) for candidate in candidates]
        for obj_id in ids
    ], dtype=np.float64)
    rows, cols = linear_sum_assignment(-similarities)
    matched = {}
    match_ious = {obj_id: 0.0 for obj_id in ids}
    for r, c in zip(rows, cols):
        iou = float(similarities[r, c])
        if iou >= min_iou:
            matched[ids[r]] = candidates[c]['box']
            match_ious[ids[r]] = iou
    return matched, match_ious


def detect_first_frame(processor, detector, image, prompt, threshold, text_threshold, device):
    import torch
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = detector(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)]
    )[0]
    return [
        {'box': [float(x) for x in box], 'score': float(score), 'label': str(label)}
        for box, score, label in zip(
            result['boxes'].cpu().tolist(),
            result['scores'].cpu().tolist(), result['labels']
        )
    ]


def run_sequence(root, sequence, output, predictor, detector, processor, args):
    import torch
    image_dir = root / 'JPEGImages' / '480p' / sequence
    images = sorted(image_dir.glob('*.jpg'))
    if not images:
        raise FileNotFoundError(image_dir)
    gt = load_gt(root, sequence, images[0].stem)
    gt_boxes = object_boxes(gt)
    image = Image.open(images[0]).convert('RGB')
    if gt.shape != (image.height, image.width):
        raise ValueError(f'{sequence}: mismatched GT/image size')
    prompt = PROMPTS[sequence]
    detections = detect_first_frame(
        processor, detector, image, prompt,
        args.threshold, args.text_threshold, args.device
    )
    matched, match_ious = assign_detections(gt_boxes, detections, args.min_match_iou)
    save_dir = output / 'predictions' / sequence
    save_dir.mkdir(parents=True, exist_ok=True)
    # Always rerun inference: don't accidentally reuse an incompatible cache.
    started = time.perf_counter()
    if matched:
        state = predictor.init_state(video_path=str(image_dir), offload_video_to_cpu=True)
        try:
            context = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()
            with torch.inference_mode(), context:
                for obj_id, box in matched.items():
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=obj_id,
                        box=np.asarray(box, dtype=np.float32)
                    )
                produced = set()
                for frame_idx, ids, logits in predictor.propagate_in_video(state):
                    if frame_idx >= len(images):
                        raise ValueError(f'{sequence}: unexpected frame index {frame_idx}')
                    if frame_idx == 0:
                        continue
                    mask = compose_logits(list(ids), logits, gt.shape)
                    Image.fromarray(mask).save(save_dir / f'{images[frame_idx].stem}.png')
                    produced.add(frame_idx)
                missing_frames = set(range(1, len(images))) - produced
                if missing_frames:
                    raise RuntimeError(f'{sequence}: missing SAM frames {sorted(missing_frames)}')
        finally:
            predictor.reset_state(state)
    else:
        # All objects missed: score empty predictions rather than dropping the video.
        for path in images[1:]:
            Image.fromarray(np.zeros(gt.shape, dtype=np.uint8)).save(
                save_dir / f'{path.stem}.png'
            )
    return {
        'sequence': sequence, 'prompt': prompt,
        'gt_objects': len(gt_boxes), 'detected_candidates': len(detections),
        'matched_objects': len(matched),
        'missed_object_ids': sorted(set(gt_boxes) - set(matched)),
        'first_frame_match_iou': {str(k): v for k, v in match_ious.items()},
        'detections': detections, 'matched_boxes': {str(k): v for k, v in matched.items()},
        'inference_seconds': round(time.perf_counter() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--davis-root', type=Path, default=Path('data/datasets/davis2017/DAVIS'))
    parser.add_argument('--limit', type=int, default=1, help='0 means all 30 validation sequences')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--sam-model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--detector-model', default='IDEA-Research/grounding-dino-tiny')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--min-match-iou', type=float, default=0.3)
    parser.add_argument('--output', type=Path, default=Path('outputs/m411_davis_detector_sam2'))
    args = parser.parse_args()
    if args.limit < 0 or not (0 <= args.min_match_iou <= 1):
        parser.error('Invalid limit or match IoU threshold')
    sequences = list_sequences(args.davis_root, 'val', args.limit)
    missing_prompts = sorted(set(sequences) - set(PROMPTS))
    if missing_prompts:
        parser.error(f'Missing sequence prompts: {missing_prompts}')
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2_video_predictor_hf
    print('Loading detector and SAM predictor...', flush=True)
    processor = AutoProcessor.from_pretrained(args.detector_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.detector_model
    ).to(args.device).eval()
    predictor = build_sam2_video_predictor_hf(args.sam_model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sequence in enumerate(sequences, 1):
        print(f'[{index}/{len(sequences)}] {sequence}', flush=True)
        record = run_sequence(
            args.davis_root, sequence, args.output, predictor, detector, processor, args
        )
        records.append(record)
        (args.output / 'detections.json').write_text(json.dumps(records, indent=2))
        print(f"  matched {record['matched_objects']}/{record['gt_objects']}", flush=True)
    rows = []
    for sequence in sequences:
        rows.extend(score_sequence(args.davis_root, sequence, args.output / 'predictions'))
    with (args.output / 'per_frame.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sequence', 'frame', 'object_id', 'J', 'F', 'JF'])
        writer.writeheader()
        writer.writerows(rows)
    matched = sum(r['matched_objects'] for r in records)
    total = sum(r['gt_objects'] for r in records)
    summary = {
        'model': args.sam_model, 'detector': args.detector_model,
        'subset': 'val', 'sequence_count': len(sequences),
        'initialization': 'Grounding DINO boxes with oracle GT-ID assignment',
        'identity_matching_uses_ground_truth': True,
        'sam_prompts_use_ground_truth_boxes': False,
        'per_sequence_text_prompts_manual': True,
        'match_iou_threshold': args.min_match_iou,
        'matched_objects': matched, 'total_objects': total,
        'initialization_recall': matched / total if total else None,
        'evaluation': 'official davis2017.metrics J/F',
        'aggregation': 'mean over frames per object, then mean over objects',
        'metrics': summarize(rows), 'sequences': records,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'sequences'}, indent=2))


if __name__ == '__main__':
    main()
