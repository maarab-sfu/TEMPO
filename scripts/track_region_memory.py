"""M4.3: post-process M4.2 detections into appearance-aware tentative tracks.

No detector inference. All timestamps remain nominal sampled-frame timestamps.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    aa = max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb = max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter > 0 else 0.0


def nms(detections, threshold=0.65):
    """Class-wise NMS within a frame; never suppress distinct labels."""
    kept = []
    for label in sorted(set(d['label'] for d in detections)):
        group = sorted((d for d in detections if d['label']==label), key=lambda d:-d['score'])
        for d in group:
            if all(iou(d['box'], k['box']) < threshold for k in kept if k['label']==label):
                kept.append(d)
    return kept


def hist_feature(frame, box):
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = [int(round(x)) for x in box]
    x1,x2 = max(0,min(w,x1)), max(0,min(w,x2))
    y1,y2 = max(0,min(h,y1)), max(0,min(h,y2))
    if x2<=x1 or y2<=y1: return None
    crop = frame[y1:y2,x1:x2]
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    feat = cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256])
    feat = cv2.normalize(feat,feat).flatten()
    return feat


def appearance_similarity(a,b):
    if a is None or b is None: return 0.0
    return max(0.0,min(1.0,float(cv2.compareHist(a.astype('float32'),b.astype('float32'),cv2.HISTCMP_CORREL)+1)/2))


def center_distance(a,b,diag):
    ac=((a[0]+a[2])/2,(a[1]+a[3])/2)
    bc=((b[0]+b[2])/2,(b[1]+b[3])/2)
    return math.dist(ac,bc)/max(1,diag)


def associate_frames(frame_detections, diag, max_gap=6.2, max_cost=0.62, min_appearance=0.48):
    """One-to-one Hungarian association per timestamp; no same-frame identity reuse."""
    tracks=[]
    for timestamp, ds in sorted(frame_detections.items()):
        active=[t for t in tracks if 0 < timestamp-t['end_seconds'] <= max_gap]
        costs=np.full((len(active),len(ds)),1e6,dtype=float)
        for i,t in enumerate(active):
            for j,d in enumerate(ds):
                if t['label'] != d['label']: continue
                app=appearance_similarity(t['_feature'],d.get('_feature'))
                overlap=iou(t['_box'],d['box'])
                dist=center_distance(t['_box'],d['box'],diag)
                # Appearance can bridge a large camera/subject movement; spatial
                # gating prevents implausible matches when color alone is similar.
                if app < min_appearance or dist > 0.70: continue
                costs[i,j]=0.48*(1-app)+0.32*(1-overlap)+0.20*min(1,dist/0.50)
        matches=[]
        if costs.size:
            rows,cols=linear_sum_assignment(costs)
            matches=[(i,j) for i,j in zip(rows,cols) if costs[i,j] <= max_cost]
        matched=set()
        for i,j in matches:
            t,d=active[i],ds[j]
            t['observations'].append(clean_observation(d))
            t['end_seconds']=timestamp
            t['_box']=d['box']; t['_feature']=d.get('_feature')
            matched.add(j)
        for j,d in enumerate(ds):
            if j in matched: continue
            tracks.append({'track_id':f"{d['label'].replace(' ','_')}_{len(tracks)+1:04d}",
                           'label':d['label'],'start_seconds':timestamp,'end_seconds':timestamp,
                           '_box':d['box'],'_feature':d.get('_feature'),
                           'observations':[clean_observation(d)]})
    for t in tracks:
        t.pop('_box');t.pop('_feature')
        t['observation_count']=len(t['observations'])
    return tracks


def clean_observation(d):
    return {k:d[k] for k in ('frame_index','timestamp','box','score')}


def read_frame(cap, frame_index, max_side):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_index))
    ok,frame=cap.read()
    if not ok: raise RuntimeError(f'Cannot decode frame {frame_index}')
    h,w=frame.shape[:2]
    if max(h,w)>max_side:
        s=max_side/max(h,w)
        frame=cv2.resize(frame,(round(w*s),round(h*s)))
    return frame


def draw_sheet(cap, tracks, output, max_side, count):
    from PIL import Image,ImageDraw
    by_frame=defaultdict(list)
    for t in tracks:
        for o in t['observations']:
            by_frame[o['frame_index']].append((t['track_id'],t['label'],o['box']))
    indices=sorted(by_frame)
    if not indices or count<=0: return
    selected=list(dict.fromkeys(indices[round(i*(len(indices)-1)/max(1,min(count,len(indices))-1))]
                                for i in range(min(count,len(indices)))))
    tw,th=440,290
    sheet=Image.new('RGB',(tw*2,th*math.ceil(len(selected)/2)),'white')
    for k,idx in enumerate(selected):
        frame=read_frame(cap,idx,max_side)
        h,w=frame.shape[:2]
        scale=min((tw-16)/w,(th-36)/h)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((round(w*scale),round(h*scale)))
        dr=ImageDraw.Draw(im)
        for tid,label,box in by_frame[idx]:
            x1,y1,x2,y2=[int(round(x*scale)) for x in box]
            hue=(sum(ord(c)*(i+1) for i,c in enumerate(tid))*73)%360
            import colorsys
            color=tuple(int(255*c) for c in colorsys.hsv_to_rgb(hue/360,0.85,0.95))
            dr.rectangle((x1,y1,x2,y2),outline=color,width=3)
            dr.text((x1,max(0,y1-13)),tid,fill='white',stroke_width=2,stroke_fill='black')
        x=(k%2)*tw+8;y=(k//2)*th+26
        sheet.paste(im,(x,y))
        ImageDraw.Draw(sheet).text((x,y-19),f'frame {idx} / {len(by_frame[idx])} tracked detections',fill='black')
    sheet.save(output,quality=92)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory',type=Path,required=True,help='Existing M4.2 region_memory_v1.json')
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--exclude-labels',nargs='*',default=['porch'])
    p.add_argument('--nms-iou',type=float,default=0.65)
    p.add_argument('--max-gap',type=float,default=6.2)
    p.add_argument('--max-cost',type=float,default=0.62)
    p.add_argument('--min-appearance',type=float,default=0.48)
    p.add_argument('--max-side',type=int,default=800,help='Must match M4.2 --max-side')
    p.add_argument('--contact-frames',type=int,default=8)
    a=p.parse_args()
    data=json.loads(a.memory.read_text(encoding='utf-8'))
    if data.get('schema_version')!=1 or 'detections' not in data:
        p.error('Expected M4.2 schema_version=1 with detections')
    if not a.video.is_file():p.error(f'Video not found: {a.video}')
    if not (0<a.nms_iou<=1 and a.max_gap>0 and 0<=a.max_cost<=1 and 0<=a.min_appearance<=1):
        p.error('Invalid association parameters')
    output=a.output_dir or a.memory.parent/'m43'
    output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(a.video))
    if not cap.isOpened():raise RuntimeError('Cannot open video')
    grouped=defaultdict(list)
    for d in data['detections']:
        if d['label'] not in a.exclude_labels:
            grouped[d['frame_index']].append(dict(d))
    filtered={i:nms(ds,a.nms_iou) for i,ds in grouped.items()}
    by_time=defaultdict(list)
    dimensions=None
    for idx,ds in sorted(filtered.items()):
        frame=read_frame(cap,idx,a.max_side)
        h,w=frame.shape[:2];dimensions=(w,h)
        for d in ds:
            d['_feature']=hist_feature(frame,d['box'])
            by_time[d['timestamp']].append(d)
    diag=math.hypot(*(dimensions or (800,450)))
    tracks=associate_frames(by_time,diag,a.max_gap,a.max_cost,a.min_appearance)
    raw=data.get('tracks',[])
    summary={'source_detections':len(data['detections']),
             'detections_after_exclusions_and_nms':sum(map(len,filtered.values())),
             'excluded_labels':a.exclude_labels,
             'm42_tracks_all_labels':len(raw),
             'm42_tracks_comparable_labels':sum(t['label'] not in a.exclude_labels for t in raw),
             'm43_tracks':len(tracks),
             'm43_multi_observation_tracks':sum(t['observation_count']>1 for t in tracks),
             'm43_singletons':sum(t['observation_count']==1 for t in tracks),
             'm43_mean_observations_per_track':round(sum(t['observation_count'] for t in tracks)/max(1,len(tracks)),2),
             'm43_tracks_by_label':dict(Counter(t['label'] for t in tracks))}
    payload={'schema_version':1,'kind':'appearance_aware_tentative_tracks',
             'source_memory':str(a.memory),'video':a.video.name,
             'timestamp_note':data.get('timestamp_note'),
             'parameters':{'max_gap':a.max_gap,'max_cost':a.max_cost,'min_appearance':a.min_appearance,
                           'nms_iou':a.nms_iou,'max_side':a.max_side,'exclude_labels':a.exclude_labels},
             'limitations':'Track IDs are tentative. Histogram appearance is not a learned re-ID embedding. Fragmentation counts are proxies, not ground-truth ID switches. Detections may be false positives.',
             'summary':summary,'tracks':tracks}
    dest=output/'tracks_v1.json'
    temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2),encoding='utf-8');temp.replace(dest)
    if a.contact_frames>0:draw_sheet(cap,tracks,output/'tracking_contact_sheet.jpg',a.max_side,a.contact_frames)
    cap.release()
    print(json.dumps({'output':str(dest),'contact_sheet':str(output/'tracking_contact_sheet.jpg'),**summary},indent=2))

if __name__=='__main__':main()
