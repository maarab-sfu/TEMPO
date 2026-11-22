"""M5.1: query-driven Grounding DINO object proposals, independent of GT labels.

CLI entry point: python -m scripts.query_object_grounding
This module deliberately does not alter the production retrieval pipeline.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Proposal:
    timestamp: float
    frame_index: int
    box: list[float]
    label: str
    score: float
    area_fraction: float
    rank_score: float


def normalize_object_phrase(phrase: str) -> str:
    """The requested object is explicit: no unvalidated noun-phrase extraction."""
    phrase = ' '.join(phrase.strip().lower().split()).strip('. ')
    if not phrase or len(phrase) > 120 or any(c in phrase for c in '\r\n'):
        raise ValueError('Provide a short object phrase, e.g. --object "toy car"')
    return phrase


def box_area_fraction(box, width: int, height: int) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0., min(x1, width)), max(0., min(x2, width))
    y1, y2 = max(0., min(y1, height)), max(0., min(y2, height))
    return max(0., x2-x1)*max(0., y2-y1)/max(1, width*height)


def select_proposals(candidates: list[Proposal], top_k: int = 5, min_separation: float = 0.5) -> list[Proposal]:
    """Score-first selection with timestamp diversity; not identity association."""
    if top_k < 1 or min_separation < 0:
        raise ValueError('top_k must be >=1 and min_separation >=0')
    chosen = []
    for p in sorted(candidates, key=lambda x: (-x.rank_score, x.timestamp, x.frame_index)):
        if all(abs(p.timestamp-q.timestamp) >= min_separation for q in chosen):
            chosen.append(p)
        if len(chosen) >= top_k:
            break
    return chosen


def sample_video(path: Path, timestamps: list[float]):
    """Decode requested frames; return actual frame timestamp and index."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise RuntimeError('Video FPS/frame count unavailable')
        seen = set()
        for t in timestamps:
            if not math.isfinite(t) or t < 0:
                continue
            idx = min(count-1, max(0, round(t*fps)))
            if idx in seen:
                continue
            seen.add(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx/fps, idx, Image.fromarray(rgb)
    finally:
        cap.release()


def detect_on_frames(video: Path, object_phrase: str, timestamps: list[float],
                     model_name: str = 'IDEA-Research/grounding-dino-tiny',
                     threshold: float = 0.25, text_threshold: float = 0.25,
                     device: str = 'cuda', min_area: float = 0.0001) -> tuple[list[Proposal], int]:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(model_name)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()
    prompt = normalize_object_phrase(object_phrase) + ' .'
    proposals = []
    frames_seen = 0
    for timestamp, idx, image in sample_video(video, timestamps):
        frames_seen += 1
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = detector(**inputs)
        detections = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold,
            text_threshold=text_threshold, target_sizes=[(image.height, image.width)]
        )[0]
        labels = detections.get('text_labels', detections.get('labels', []))
        for box, score, label in zip(detections['boxes'].cpu().tolist(),
                                     detections['scores'].cpu().tolist(), labels):
            area = box_area_fraction(box, image.width, image.height)
            if area < min_area:
                continue
            score = float(score)
            proposals.append(Proposal(timestamp=round(timestamp, 5), frame_index=idx,
                box=[round(float(v), 3) for v in box], label=str(label),
                score=score, area_fraction=area, rank_score=score))
    return proposals, frames_seen


def parse_timestamps(csv: str) -> list[float]:
    result = [float(x.strip()) for x in csv.split(',') if x.strip()]
    if not result or any(not math.isfinite(x) or x < 0 for x in result):
        raise ValueError('--timestamps must contain nonnegative finite numbers')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--object', dest='object_phrase', required=True,
                   help='Explicit object phrase from the user question (e.g. toy car)')
    p.add_argument('--question', default='', help='Original question, stored for provenance only')
    p.add_argument('--timestamps', default='0,2,4,6,8,10',
                   help='Comma-separated CLIP/FAISS anchor timestamps; defaults to a smoke-test grid')
    p.add_argument('--model', default='IDEA-Research/grounding-dino-tiny')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threshold', type=float, default=0.25)
    p.add_argument('--text-threshold', type=float, default=0.25)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--min-separation', type=float, default=0.5)
    p.add_argument('--output', type=Path, default=Path('outputs/m51_object_grounding/result.json'))
    args = p.parse_args()
    if not args.video.is_file():
        p.error(f'Video not found: {args.video}')
    phrase = normalize_object_phrase(args.object_phrase)
    times = parse_timestamps(args.timestamps)
    proposals, frames_seen = detect_on_frames(args.video, phrase, times, args.model,
        args.threshold, args.text_threshold, args.device)
    selected = select_proposals(proposals, args.top_k, args.min_separation)
    result = {'schema_version': 1, 'status': 'found' if selected else 'not_found',
              'video': str(args.video), 'question': args.question, 'object_phrase': phrase,
              'model': args.model, 'device': args.device,
              'sampling': {'requested_timestamps': times, 'frames_decoded': frames_seen},
              'thresholds': {'box': args.threshold, 'text': args.text_threshold},
              'candidate_count': len(proposals), 'candidates': [asdict(x) for x in proposals],
              'selected': [asdict(x) for x in selected],
              'limitations': ['No object identity linking across frames',
                              'Detection confidence is not calibrated',
                              'No GT-based assignment or evaluation']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"M5.1: {result['status']} | {frames_seen} frames | {len(proposals)} candidates | {len(selected)} selected")
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
