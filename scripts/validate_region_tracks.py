"""M4.4: compare tentative region tracks and generate targeted manual identity review.

Requires only Python stdlib for reports; OpenCV optional for --video review sheets.
This is a validation tool, not an automatic MOT metric or a track-merging step.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def area(b):
    return max(0., b[2]-b[0])*max(0., b[3]-b[1])


def overlap(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    aa,bb=area(a),area(b)
    return (inter/(aa+bb-inter) if aa+bb>inter else 0., inter/min(aa,bb) if min(aa,bb)>0 else 0.)


def flatten(doc):
    out=[]
    for tr in doc['tracks']:
        for ob in tr['observations']:
            out.append(dict(track_id=tr['track_id'],label=tr['label'],frame_index=ob['frame_index'],timestamp=ob['timestamp'],box=ob['box'],score=ob['score']))
    return out


def analyze(doc, name):
    obs=flatten(doc)
    by_frame=defaultdict(list)
    for ob in obs: by_frame[ob['frame_index']].append(ob)
    tracks=doc['tracks']
    frames=int(round(max((o['timestamp'] for o in obs),default=0)/ (0.5 if name=='0p5s' else 2)))+1
    # Infer actual sampling grid from all frame indices; avoids assuming every frame has a detection.
    frame_ids=sorted(by_frame)
    diffs=[b-a for a,b in zip(frame_ids,frame_ids[1:]) if b>a]
    step=Counter(diffs).most_common(1)[0][0] if diffs else 1
    expected=(max(frame_ids)//step+1) if frame_ids else 0
    by_label=defaultdict(list)
    for t in tracks:by_label[t['label']].append(t)
    conflicts=[]
    for f,items in by_frame.items():
        for i,a in enumerate(items):
            for b in items[i+1:]:
                if a['label']!=b['label'] or a['track_id']==b['track_id']:continue
                iou,contain=overlap(a['box'],b['box'])
                if iou>=0.4 or contain>=0.75:
                    conflicts.append(dict(run=name,frame_index=f,timestamp=a['timestamp'],label=a['label'],track_a=a['track_id'],track_b=b['track_id'],iou=round(iou,3),intersection_over_smaller=round(contain,3),review_reason='overlapping_same_label'))
    jumps=[]
    for t in tracks:
        o=sorted(t['observations'],key=lambda x:x['frame_index'])
        for a,b in zip(o,o[1:]):
            dt=b['timestamp']-a['timestamp']
            if dt<=0:continue
            ac=((a['box'][0]+a['box'][2])/2,(a['box'][1]+a['box'][3])/2)
            bc=((b['box'][0]+b['box'][2])/2,(b['box'][1]+b['box'][3])/2)
            dist=math.dist(ac,bc)
            diag=max(1,math.hypot(a['box'][2]-a['box'][0],a['box'][3]-a['box'][1]))
            speed=dist/diag/dt
            if speed>=0.8:
                jumps.append(dict(run=name,frame_index=b['frame_index'],timestamp=b['timestamp'],label=t['label'],track_a=t['track_id'],track_b='',iou='',intersection_over_smaller='',review_reason='fast_normalized_motion',motion_box_diagonals_per_sec=round(speed,3)))
    dominant={}
    for label,ts in by_label.items():
        t=max(ts,key=lambda x:len(x['observations']))
        dominant[label]={'track_id':t['track_id'],'observations':len(t['observations']),'sampled_frame_coverage':round(len(set(o['frame_index'] for o in t['observations']))/expected,4) if expected else 0}
    return {'run':name,'sampled_frames_inferred':expected,'sampling_step_frames':step,'tracks':len(tracks),'observations':len(obs),'multi_observation_tracks':sum(len(t['observations'])>1 for t in tracks),'singleton_tracks':sum(len(t['observations'])==1 for t in tracks),'tracks_by_label':dict(Counter(t['label'] for t in tracks)),'dominant_by_label':dominant,'overlap_conflict_events':len(conflicts),'overlap_conflict_frames':len(set(x['frame_index'] for x in conflicts)),'suspicious_motion_events':len(jumps),'limitations':'Heuristic flags are not confirmed ID switches or false detections; manual visual labels required.'}, conflicts+jumps, obs


def write_csv(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def make_review_sheet(video, candidates, out, max_items=32):
    try:import cv2
    except ImportError:raise SystemExit('Install opencv-python to use --video; JSON/CSV reports work without it')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise SystemExit(f'Could not open video: {video}')
    selected=candidates[:max_items]
    if not selected:return
    from PIL import Image,ImageDraw
    panels=[]
    for c in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(c['frame_index']))
        ok,frame=cap.read()
        if not ok:continue
        h,w=frame.shape[:2]
        # Source M4.3 boxes are in a max-side-800 resized coordinate system.
        factor=min(1.,800/max(h,w))
        canvas=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((640,360))
        draw=ImageDraw.Draw(canvas)
        sx=640/(w*factor);sy=360/(h*factor)
        for ob in c.get('detections',[]):
            b=ob['box'];xy=(b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy)
            draw.rectangle(xy,outline='#ffb000',width=3)
            draw.text((max(0,xy[0]),max(0,xy[1]-12)),ob['track_id'],fill='#ffb000')
        draw.rectangle((0,338,640,360),fill='#111111')
        draw.text((5,343),f"{c['run']} | t={c['timestamp']:.2f}s | {c['reason']}",fill='white')
        panels.append(canvas)
    cap.release()
    if not panels:return
    cols=2;rows=math.ceil(len(panels)/cols)
    sheet=Image.new('RGB',(cols*640,rows*360),'white')
    for i,p in enumerate(panels):sheet.paste(p,((i%cols)*640,(i//cols)*360))
    sheet.save(out,quality=88)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks-2s',required=True,type=Path)
    p.add_argument('--tracks-0p5s',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path('data/reports/m44'))
    p.add_argument('--video',type=Path,help='Optional original video for visual review contact sheets')
    p.add_argument('--review-limit',type=int,default=32)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    all_flags=[]; summaries=[]; review_rows=[]
    for name,path in [('2s',args.tracks_2s),('0p5s',args.tracks_0p5s)]:
        doc=json.loads(path.read_text(encoding='utf-8'))
        summary,flags,obs=analyze(doc,name);summaries.append(summary);all_flags+=flags
        by_frame=defaultdict(list)
        for ob in obs:by_frame[ob['frame_index']].append(ob)
        # Prioritize overlap conflicts, motion anomalies, and sparse manual spot-checks.
        chosen={x['frame_index']:x['review_reason'] for x in flags if x['label']=='person'}
        for x in flags:
            if x['frame_index'] not in chosen:chosen[x['frame_index']]=x['review_reason']
        all_frames=sorted(by_frame)
        for f in all_frames[::max(1,len(all_frames)//12)]:chosen.setdefault(f,'routine_spot_check')
        for f,reason in sorted(chosen.items()):
            items=by_frame[f]
            for ob in items:
                if ob['label'] not in ('person','chair','toy car'):continue
                review_rows.append(dict(run=name,frame_index=f,timestamp=ob['timestamp'],track_id=ob['track_id'],predicted_label=ob['label'],score=ob['score'],box=json.dumps(ob['box']),review_reason=reason,physical_object_id='',detection_correct='',notes=''))
        if args.video:
            candidates=[dict(run=name,frame_index=f,timestamp=by_frame[f][0]['timestamp'],reason=reason,detections=by_frame[f]) for f,reason in sorted(chosen.items(),key=lambda kv:(kv[1]=='routine_spot_check',kv[0]))]
            make_review_sheet(args.video,candidates,args.out/f'review_{name}.jpg',args.review_limit)
    (args.out/'comparison.json').write_text(json.dumps({'runs':summaries},indent=2),encoding='utf-8')
    write_csv(args.out/'suspicious_events.csv',all_flags,['run','frame_index','timestamp','label','track_a','track_b','iou','intersection_over_smaller','review_reason','motion_box_diagonals_per_sec'])
    write_csv(args.out/'manual_review.csv',review_rows,['run','frame_index','timestamp','track_id','predicted_label','score','box','review_reason','physical_object_id','detection_correct','notes'])
    print(json.dumps({'output':str(args.out),'runs':summaries,'suspicious_events':len(all_flags),'manual_review_rows':len(review_rows)},indent=2))

if __name__=='__main__':main()
