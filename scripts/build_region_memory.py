"""M4.2 experimental region detections + conservative temporal associations."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def sample_positions(frame_count, fps, interval):
    if frame_count <= 0 or fps <= 0 or interval <= 0:
        raise ValueError('Positive frame count, FPS and sampling interval required')
    step = max(1, round(fps * interval))
    return list(range(0, frame_count, step))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter / (aa+bb-inter) if aa+bb-inter else 0.0


def associate(detections, max_gap=4.1, min_iou=0.2):
    """Greedy label/IoU association; track IDs are tentative, not identities."""
    tracks = []
    for d in sorted(detections, key=lambda x: (x['timestamp'], -x['score'])):
        choices = [(iou(d['box'], t['last_box']), i, t) for i,t in enumerate(tracks)
                   if t['label'] == d['label'] and 0 < d['timestamp']-t['end_seconds'] <= max_gap
                   and t['end_seconds'] != d['timestamp']]
        choices = [(score,i,t) for score,i,t in choices if score >= min_iou]
        choices.sort(reverse=True, key=lambda x:x[0])
        used = {o['frame_index'] for o in []}
        if choices and not any(x['frame_index'] == d['frame_index'] for x in choices[0][2]['observations']):
            t = choices[0][2]
        else:
            t = {'track_id':f'tentative_{len(tracks):04d}', 'label':d['label'], 'start_seconds':d['timestamp'],
                 'end_seconds':d['timestamp'], 'last_box':d['box'], 'observations':[]}
            tracks.append(t)
        t['observations'].append({'frame_index':d['frame_index'], 'timestamp':d['timestamp'], 'box':d['box'], 'score':d['score']})
        t['end_seconds'] = d['timestamp']
        t['last_box'] = d['box']
    for t in tracks:
        t.pop('last_box')
        t['observation_count'] = len(t['observations'])
    return tracks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=Path, required=True, help='Original MP4; not the CLIP index')
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--labels', nargs='+', default=['person','knife','garlic','cutting board'])
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--threshold', type=float, default=0.3)
    p.add_argument('--text-threshold', type=float, default=0.25)
    p.add_argument('--max-gap', type=float, default=4.1)
    p.add_argument('--min-iou', type=float, default=0.2)
    p.add_argument('--max-side', type=int, default=800)
    p.add_argument('--contact-frames', type=int, default=8)
    p.add_argument('--model', default='IDEA-Research/grounding-dino-tiny')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    import cv2
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if not a.video.is_file():
        p.error(f'Video not found: {a.video}')
    if not 0 < a.threshold < 1 or not 0 < a.text_threshold < 1:
        p.error('Thresholds must be between 0 and 1')
    output = a.output_dir or a.video.parent / (a.video.stem+'_m42')
    output.mkdir(parents=True, exist_ok=True)
    dest = output/'region_memory_v1.json'
    if dest.exists() and not a.overwrite:
        p.error(f'{dest} exists; pass --overwrite to replace')
    cap = cv2.VideoCapture(str(a.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or count <= 0:
        raise RuntimeError('Cannot decode video metadata/FPS')
    indices = sample_positions(count, fps, a.interval)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device={device}; frames={len(indices)}; source FPS={fps:.5f}; loading {a.model}', flush=True)
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).to(device).eval()
    prompt = '. '.join(a.labels)+'.'
    detections = []
    samples = []
    for k, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            print(f'WARNING: cannot decode frame {frame_idx}; skipping')
            continue
        actual_index = max(0, int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1)
        pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        # Frame index/FPS is a nominal timestamp, not ground-truth event boundary.
        timestamp = actual_index/fps
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        if max(im.size) > a.max_side:
            factor = a.max_side/max(im.size)
            im = im.resize((round(im.width*factor), round(im.height*factor)))
        inputs = processor(images=im, text=prompt, return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
                    threshold=a.threshold, text_threshold=a.text_threshold,
                    target_sizes=[(im.height, im.width)])[0]
        for box, score, label in zip(results['boxes'].cpu().tolist(), results['scores'].cpu().tolist(), results['labels']):
            detections.append({'frame_index':actual_index, 'timestamp':round(timestamp,4),
                               'label':str(label).strip().lower().rstrip('.'),
                               'score':round(float(score),5), 'box':[round(float(v),2) for v in box]})
        samples.append((actual_index, timestamp, im.copy()))
        if (k+1)%10 == 0:
            print(f'Processed {k+1}/{len(indices)} frames', flush=True)
    cap.release()
    tracks = associate(detections, a.max_gap, a.min_iou)
    payload = {'schema_version':1,'kind':'region_detections_tentative_associations',
               'video':a.video.name,'source_fps':fps,'frame_count':count,
               'timestamp_note':'Nominal timestamps = decoded frame index / source FPS; not exact event boundaries. Variable-frame-rate files may require PTS-aware decoding.',
               'sampling_interval_requested_seconds':a.interval,'model':a.model,
               'parameters':{'labels':a.labels,'threshold':a.threshold,'text_threshold':a.text_threshold,
                             'max_gap_seconds':a.max_gap,'min_iou':a.min_iou},
               'limitations':'Detector boxes are predictions; greedy IoU associations are tentative and may switch IDs, especially at 2s sampling.',
               'detections':detections,'tracks':tracks}
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    tmp.replace(dest)
    if samples and a.contact_frames > 0:
        n = min(len(samples), a.contact_frames)
        selected = [samples[round(i*(len(samples)-1)/max(1,n-1))] for i in range(n)]
        thumb_w, thumb_h = 420, 280
        cols = 2
        rows = math.ceil(n/cols)
        sheet = Image.new('RGB',(cols*thumb_w,rows*thumb_h),'#f5f5f5')
        for j,(idx,ts,im) in enumerate(selected):
            scale = min((thumb_w-16)/im.width,(thumb_h-38)/im.height)
            im = im.resize((round(im.width*scale),round(im.height*scale)))
            draw = ImageDraw.Draw(im)
            local = [d for d in detections if d['frame_index']==idx]
            for d in local:
                box = [round(v*scale) for v in d['box']]
                draw.rectangle(box,outline='red',width=3)
                draw.text((box[0]+2,max(0,box[1]-13)),f"{d['label']} {d['score']:.2f}",fill='yellow',stroke_width=1,stroke_fill='black')
            x = (j%cols)*thumb_w+8
            y = (j//cols)*thumb_h+28
            sheet.paste(im,(x,y))
            ImageDraw.Draw(sheet).text((x,y-20),f'frame {idx} / {ts:.2f}s / {len(local)} detections',fill='black')
        sheet.save(output/'contact_sheet.jpg',quality=90)
    print(json.dumps({'frames_sampled':len(samples),'detections':len(detections),
                      'labels':dict(Counter(d['label'] for d in detections)),
                      'tentative_tracks':len(tracks),'multi_observation_tracks':sum(t['observation_count']>1 for t in tracks),
                      'output':str(dest),'contact_sheet':str(output/'contact_sheet.jpg')},indent=2))

if __name__ == '__main__':
    main()
