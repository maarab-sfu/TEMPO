"""M5.2: propagate one Grounding DINO proposal with SAM2 (no oracle ID)."""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path


def eligible(candidates, max_area=0.35, min_score=0.25):
    return sorted((c for c in candidates if min_score <= float(c['score']) and
                   0 < float(c['area_fraction']) <= max_area and
                   len(c['box']) == 4 and c['box'][2] > c['box'][0] and
                   c['box'][3] > c['box'][1]),
                  key=lambda c: (-float(c['score']), float(c['timestamp'])))


def scale_to_frame(box, source_width, source_height, target_width, target_height):
    sx, sy = target_width/source_width, target_height/source_height
    return [float(box[0])*sx, float(box[1])*sy,
            float(box[2])*sx, float(box[3])*sy]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--candidate-index', type=int, default=0,
                   help='Index in filtered, confidence-sorted proposals; each run tracks ONE instance')
    p.add_argument('--max-area', type=float, default=0.35)
    p.add_argument('--min-score', type=float, default=0.25)
    p.add_argument('--before', type=float, default=4)
    p.add_argument('--after', type=float, default=4)
    p.add_argument('--sample-fps', type=float, default=5)
    p.add_argument('--max-side', type=int, default=640)
    p.add_argument('--model', default='facebook/sam2.1-hiera-tiny')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, default=Path('outputs/m52_grounded_tracking'))
    a=p.parse_args()
    if not 0 < a.max_area <= 1 or a.before < 0 or a.after < 0 or a.sample_fps <= 0:
        p.error('Invalid filter/window arguments')
    from scripts.experiment_sam2_tracking import decode_window, propagate, mask_bbox
    import cv2, numpy as np
    data=json.loads(a.detections.read_text(encoding='utf-8'))
    candidates=eligible(data.get('candidates',[]),a.max_area,a.min_score)
    if not candidates:
        p.error('No eligible detection proposals; try adjusting --max-area or --min-score')
    if not 0 <= a.candidate_index < len(candidates):
        p.error(f'--candidate-index must be in [0,{len(candidates)-1}]')
    seed=candidates[a.candidate_index]
    video=Path(data['video'])
    if not video.is_file():
        p.error(f'Video not found: {video}')
    frames, indices, seed_pos, fps, w, h=decode_window(
        video, float(seed['timestamp']),a.before,a.after,a.sample_fps,a.max_side)
    box=scale_to_frame(seed['box'],w,h,frames[seed_pos].shape[1],frames[seed_pos].shape[0])
    with tempfile.TemporaryDirectory(prefix='videomind_m52_') as tmp:
        fwd=propagate(frames,seed_pos,box,a.model,a.device,Path(tmp),False)
        bwd=propagate(frames,seed_pos,box,a.model,a.device,Path(tmp),True)
    masks={**bwd,**fwd}
    a.output.mkdir(parents=True,exist_ok=True)
    overlay_dir=a.output/'overlays'; overlay_dir.mkdir(exist_ok=True)
    rows=[]
    for i,(frame,frame_index) in enumerate(zip(frames,indices)):
        mask=masks.get(i)
        bbox=mask_bbox(mask) if mask is not None else None
        pixels=int(mask.sum()) if mask is not None else 0
        rows.append({'frame_index':frame_index,'timestamp':round(frame_index/fps,5),
                     'mask_pixels':pixels,'box':bbox})
        vis=frame.copy()
        if mask is not None and pixels:
            vis[mask]=(vis[mask].astype(np.float32)*.65 +
                       np.array([0,210,0],dtype=np.float32)*.35).astype(np.uint8)
        if bbox is not None:
            cv2.rectangle(vis,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
        cv2.imwrite(str(overlay_dir/f'{i:05d}.jpg'),vis)
    output={'schema_version':1,'status':'tracked','video':str(video),
            'object_phrase':data.get('object_phrase'),'seed':seed,
            'candidate_index':a.candidate_index,'eligible_candidate_count':len(candidates),
            'model':a.model,'seed_box_rescaled':box,'frames':rows,
            'limitations':['One detection proposal per run; no multi-instance identity linking',
                           'Masks not scored against GT','Missing masks can mean occlusion or exit']}
    (a.output/'trajectory.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(f"M5.2: {len(rows)} frames; {sum(r['mask_pixels']>0 for r in rows)} nonempty masks")
    print(f"Output: {a.output/'trajectory.json'}")

if __name__=='__main__':main()
