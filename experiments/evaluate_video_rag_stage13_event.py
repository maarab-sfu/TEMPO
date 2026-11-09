"""TEMPO Stage 13: frame vs temporal-window vs expanded event-context Video RAG.

Place at experiments/evaluate_video_rag_stage13.py and run from repository root:
 python -m experiments.evaluate_video_rag_stage13 --video path/to/video.mp4 --questions questions.json

Requires: torch, transformers, pillow, opencv-python, numpy; optional faiss-cpu.
CLIP: openai/clip-vit-base-patch32; VLM: Qwen/Qwen2.5-VL-3B-Instruct.
Does not require BURST oracle inputs, tracked objects, or prior experiment files.
"""
from __future__ import annotations
import argparse, json, math, re, time
from pathlib import Path
import numpy as np


def load_questions(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data,dict): data=data.get('questions',[])
    if not isinstance(data,list) or not data: raise ValueError('Questions JSON must be a nonempty list or {"questions": [...]}')
    result=[]
    for j,q in enumerate(data):
        if not isinstance(q,dict) or not isinstance(q.get('question'),str): raise ValueError(f'Question {j} must have a question string')
        result.append({'id':str(q.get('id',j)), 'question':q['question'], 'answer':q.get('answer'), 'evidence_seconds':q.get('evidence_seconds'), 'evidence_start_seconds':q.get('evidence_start_seconds'), 'evidence_end_seconds':q.get('evidence_end_seconds')})
    if len({q['id'] for q in result})!=len(result): raise ValueError('Duplicate question IDs')
    return result


def sample_video(path, step, max_frames, resize):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {path}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps<=0: raise RuntimeError('Video FPS unavailable')
    duration=count/fps
    times=np.arange(0,duration,step,dtype=np.float64)
    if len(times)>max_frames: raise ValueError(f'{len(times)} sampled frames exceeds --max-frames {max_frames}; increase --step or --max-frames')
    frames=[]; actual=[]
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
        ok,frame=cap.read()
        if not ok: continue
        h,w=frame.shape[:2]
        scale=min(1.,resize/max(h,w))
        if scale<1: frame=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        actual.append(float(t))
    cap.release()
    if not frames: raise RuntimeError('No frames decoded')
    return frames,np.array(actual,dtype=np.float64),duration


def clip_embeddings(frames,questions,model_name,batch_size,device):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    processor=CLIPProcessor.from_pretrained(model_name)
    model=CLIPModel.from_pretrained(model_name).to(device).eval()
    images=[]
    with torch.inference_mode():
        for start in range(0,len(frames),batch_size):
            batch=processor(images=frames[start:start+batch_size],return_tensors='pt',padding=True)
            batch={k:v.to(device) for k,v in batch.items()}
            feats=model.get_image_features(**batch)
            if not isinstance(feats,torch.Tensor): feats=feats.pooler_output
            feats=torch.nn.functional.normalize(feats.float(),dim=-1)
            images.append(feats.cpu().numpy())
        queries=[]
        for start in range(0,len(questions),batch_size):
            batch=processor(text=questions[start:start+batch_size],return_tensors='pt',padding=True,truncation=True)
            batch={k:v.to(device) for k,v in batch.items()}
            feats=model.get_text_features(**batch)
            if not isinstance(feats,torch.Tensor): feats=feats.pooler_output
            feats=torch.nn.functional.normalize(feats.float(),dim=-1)
            queries.append(feats.cpu().numpy())
    del model
    if device.startswith('cuda'): torch.cuda.empty_cache()
    return np.concatenate(images),np.concatenate(queries)


def retrieve(scores,times,k,mode,radius):
    """Frame: top-k independent. Temporal: best window by mean query similarity.
    Both modes use exactly k frames, ordered chronologically, from the same sampled pool.
    Window mode enforces contiguous sample indices (radius only used as max allowed span in seconds).
    """
    n=len(scores);k=min(k,n)
    if mode=='frame':
        order=sorted(range(n),key=lambda i:(-float(scores[i]),i))[:k]
    else:
        # Temporal mode: rank contiguous windows using mean score; never compare
        # an entire long window against a single frame's score.
        choices=[]
        for start in range(n-k+1):
            end=start+k-1
            if radius>0 and times[end]-times[start]>radius+1e-6: continue
            choices.append((float(np.mean(scores[start:start+k])),start))
        if not choices: raise ValueError('No eligible temporal window. Increase --window-max-span or reduce --evidence-frames')
        best=max(choices,key=lambda x:(x[0],-x[1]))[1]
        order=list(range(best,best+k))
    return sorted(order)



def event_window(scores, times, k, question):
    """Anchor on CLIP's strongest matching frame, then expose a longer event context.

    For 'after' questions, keep the anchor and following frames; otherwise center
    the window on the anchor. This is an event-context baseline, NOT event boundary
    detection and NOT guaranteed to include the next semantic action.
    """
    n=len(scores)
    k=min(k,n)
    anchor=int(np.argmax(scores))
    if re.search(r"\b(after|next|following|subsequently)\b", question, re.I):
        start=min(anchor,n-k)
    else:
        start=max(0,min(anchor-k//2,n-k))
    return list(range(start,start+k)), anchor


def interval_metrics(indices,times,q):
    start=q.get('evidence_start_seconds')
    end=q.get('evidence_end_seconds')
    if start is None or end is None:
        return None
    start,end=float(start),float(end)
    if start>end: raise ValueError(f'Invalid evidence interval for {q["id"]}')
    selected=[float(times[i]) for i in indices]
    hits=sum(start<=t<=end for t in selected)
    return {'inside_count':hits,'selected_count':len(selected),
            'frame_precision':round(hits/len(selected),4),
            'any_hit':bool(hits),'reference_interval_seconds':[start,end]}


def load_vlm(model_name,device):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor=AutoProcessor.from_pretrained(model_name)
    kwargs={'torch_dtype':torch.float16 if device.startswith('cuda') else torch.float32}
    if device.startswith('cuda'):kwargs['device_map']='auto'
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,**kwargs).eval()
    if not device.startswith('cuda'):model=model.to(device)
    return model,processor


def answer(model,processor,question,frames,times,indices,max_new_tokens):
    import torch
    content=[{'type':'text','text':('Answer the question using only the supplied video frames, in chronological order. '
             'Frames are sparse observations; do not invent actions between them. '
             'For WHEN questions, answer with approximate video timestamps in seconds; give a start-end interval only if both boundaries are visible. '
             'For AFTER questions, identify an action visible later than the referenced action; if not shown, say evidence is insufficient. '
             'If the evidence is insufficient, say so. Question: '+question)}]
    for idx in indices:
        content.append({'type':'text','text':f'Video time {times[idx]:.2f} seconds:'})
        content.append({'type':'image','image':frames[idx]})
    messages=[{'role':'user','content':content}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    chosen=[frames[i] for i in indices]
    inputs=processor(text=[prompt],images=chosen,padding=True,return_tensors='pt')
    device=next(model.parameters()).device
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    decoded=processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return decoded[0].strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/stage13'))
    p.add_argument('--step',type=float,default=2.)
    p.add_argument('--max-frames',type=int,default=2000)
    p.add_argument('--resize',type=int,default=768)
    p.add_argument('--evidence-frames',type=int,default=4)
    p.add_argument('--event-frames',type=int,default=12,help='Number of frames for expanded event context (default 12); set equal to --evidence-frames for budget-matched comparison')
    p.add_argument('--window-max-span',type=float,default=0.,help='0 means no span restriction')
    p.add_argument('--clip-model',default='openai/clip-vit-base-patch32')
    p.add_argument('--vlm-model',default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--clip-batch',type=int,default=16)
    p.add_argument('--max-new-tokens',type=int,default=160)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--retrieval-only',action='store_true',help='Skip VLM; validate retrieval first')
    args=p.parse_args()
    import torch
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    if args.step<=0 or args.evidence_frames<1 or args.event_frames<1 or args.clip_batch<1:raise ValueError('step, evidence-frames and clip-batch must be positive')
    qs=load_questions(args.questions)
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    frames,times,duration=sample_video(args.video,args.step,args.max_frames,args.resize)
    print(f'Sampled {len(frames)} frames over {duration:.1f}s. Computing CLIP embeddings...',flush=True)
    img,txt=clip_embeddings(frames,[q['question'] for q in qs],args.clip_model,args.clip_batch,device)
    scores=txt@img.T
    rows=[]
    for j,q in enumerate(qs):
        for mode in ('frame','temporal','event'):
            if mode=='event':
                indices,anchor=event_window(scores[j],times,args.event_frames,q['question'])
            else:
                indices=retrieve(scores[j],times,args.evidence_frames,mode,args.window_max_span)
                anchor=None
            row={'question_id':q['id'],'question':q['question'],'reference_answer':q['answer'],
                 'reference_evidence_seconds':q['evidence_seconds'],
                 'reference_evidence_start_seconds':q['evidence_start_seconds'],
                 'reference_evidence_end_seconds':q['evidence_end_seconds'],
                 'mode':mode,'retrieved_indices':indices,
                 'retrieved_seconds':[round(float(times[i]),3) for i in indices],
                 'retrieval_scores':[round(float(scores[j,i]),6) for i in indices],
                 'anchor_seconds':round(float(times[anchor]),3) if anchor is not None else None,
                 'interval_retrieval_metrics':interval_metrics(indices,times,q),
                 'prediction':None}
            rows.append(row)
    if not args.retrieval_only:
        print('Loading Qwen VLM...',flush=True)
        model,processor=load_vlm(args.vlm_model,device)
        for j,row in enumerate(rows):
            print(f'[{j+1}/{len(rows)}] {row["question_id"]} {row["mode"]}',flush=True)
            try:
                row['prediction']=answer(model,processor,row['question'],frames,times,row['retrieved_indices'],args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError('VLM CUDA OOM: try --evidence-frames 2 --resize 448 or --retrieval-only')
    summary={'stage':'13','protocol':'Equal frame budget, frozen CLIP index, frame top-k vs contiguous temporal-window mean similarity; no oracle tracking inputs',
             'video':str(args.video),'duration_seconds':duration,'sampled_frames':len(frames),'sampling_step_seconds':args.step,
             'evidence_frames':min(args.evidence_frames,len(frames)),'event_frames':min(args.event_frames,len(frames)),'window_max_span_seconds':args.window_max_span,
             'clip_model':args.clip_model,'vlm_model':None if args.retrieval_only else args.vlm_model,
             'device':device,'elapsed_seconds':round(time.perf_counter()-start,2),'results':rows,
             'limitations':['Event mode uses a different frame budget unless --event-frames equals --evidence-frames; do not attribute improvements solely to retrieval algorithm', 'Event mode is anchored on top CLIP frame, not detected semantic event boundaries', 'Single video until run on additional videos','No automatic answer grading: inspect or externally score predictions against reference answers','Temporal baseline uses contiguous fixed-length window; may miss distant evidence','Both retrieval methods use CLIP text-frame similarity; no object tracking yet','Sampling may miss short events']}
    out=args.output/(re.sub(r'[^A-Za-z0-9_.-]+','_',args.video.stem)+'_stage13_event.json')
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved:',out,flush=True)
    for row in rows:print(row['question_id'],row['mode'],row['retrieved_seconds'],row['prediction'] or '(retrieval only)')

if __name__=='__main__':main()
