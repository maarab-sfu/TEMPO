"""Build experimental object-crop CLIP index from cached M4.5 tracks (no DINO rerun)."""
import argparse
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from src.retrieval.object_hybrid import load_tracks,build_candidates,save_index


def build(video,tracks,output,min_observations=3,min_score=.35):
    from src.embeddings.clip_encoder import CLIPEncoder
    candidates=build_candidates(load_tracks(tracks),min_observations,min_score)
    if not candidates:raise ValueError('No candidates passed filters')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise FileNotFoundError(video)
    encoder=CLIPEncoder()
    kept=[];vectors=[]
    with tempfile.TemporaryDirectory(prefix='videomind_m46_') as temp:
        # One seek per distinct frame, reuse decoded images for multiple boxes.
        by_frame={}
        for c in candidates:by_frame.setdefault(c['frame_index'],[]).append(c)
        for frame_idx,items in sorted(by_frame.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
            ok,frame=cap.read()
            if not ok:continue
            h,w=frame.shape[:2]
            paths=[];valid=[]
            for c in items:
                x1,y1,x2,y2=c['box'];x1=max(0,int(x1));y1=max(0,int(y1));x2=min(w,int(x2));y2=min(h,int(y2))
                if x2-x1<8 or y2-y1<8:continue
                path=Path(temp)/f'{frame_idx}_{len(paths)}.jpg'
                Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)).save(path)
                paths.append(str(path));valid.append(c)
            if paths:
                arr=np.asarray(encoder.encode_images(paths,batch_size=16),dtype=np.float32)
                kept.extend(valid);vectors.extend(arr)
    cap.release()
    if not kept:raise ValueError('No valid crop images; check box coordinate scale')
    save_index(kept,np.stack(vectors),output,Path(video).name)
    return {'indexed_crops':len(kept),'unique_tracks':len(set(c['track_id'] for c in kept)),'output':str(output)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--tracks',required=True);p.add_argument('--output',required=True)
    p.add_argument('--min-observations',type=int,default=3);p.add_argument('--min-score',type=float,default=.35)
    a=p.parse_args();import json
    print(json.dumps(build(a.video,a.tracks,a.output,a.min_observations,a.min_score),indent=2))
if __name__=='__main__':main()
