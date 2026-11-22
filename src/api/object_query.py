"""M5.4 opt-in object-aware query endpoint; baseline /query remains unchanged."""
from __future__ import annotations
import json
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'data' / 'indexes'
UPLOAD = ROOT / 'data' / 'uploads'
OUT = ROOT / 'outputs' / 'm54_object_queries'

class ObjectQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    object_phrase: str = Field(min_length=1, max_length=120)
    top_k: int = Field(default=2, ge=1, le=5)
    evidence_budget: int = Field(default=4, ge=1, le=6)
    before: float = Field(default=4, ge=0, le=12)
    after: float = Field(default=4, ge=0, le=12)


def video_source(video_id: str) -> Path:
    if not video_id or any(not(c.isalnum() or c in '-_') for c in video_id):
        raise ValueError('Invalid video ID')
    matches = [p for p in UPLOAD.glob(video_id + '.*') if p.suffix.lower() in {'.mp4','.mov','.avi','.mkv','.webm'}]
    if len(matches) != 1:
        raise FileNotFoundError('Original uploaded video unavailable; re-upload to use object-aware mode')
    return matches[0]


def run_step(module: str, *args: str) -> None:
    completed = subprocess.run([sys.executable, '-m', module, *map(str,args)], cwd=ROOT,
                               capture_output=True, text=True, timeout=900, check=False)
    if completed.returncode:
        raise RuntimeError(f'{module} failed: {(completed.stderr or completed.stdout)[-1400:]}')


def merge_frames(semantic, object_rows, video: Path, work: Path, budget: int = 8):
    """Preserve semantic anchors, add object frames only in remaining slots."""
    import cv2
    semantic = list(semantic)[:budget]
    remaining = budget-len(semantic)
    if remaining <= 0 or not object_rows:
        return semantic, []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('Cannot decode source video')
    selected=[]
    try:
        for row in object_rows:
            if len(selected)>=remaining: break
            t=float(row['timestamp'])
            if any(abs(float(f['timestamp'])-t)<0.35 for f in semantic+selected):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(row['frame_index']))
            ok,img=cap.read()
            if not ok:continue
            dest=work/f'object_{len(selected):02d}.jpg'
            if not cv2.imwrite(str(dest),img):continue
            selected.append({'frame_id':f'O{len(selected)+1}', 'timestamp':t,
                             'image_path':str(dest), 'evidence_role':'object_track',
                             'anchor_rank':None,'similarity':None})
    finally:cap.release()
    return sorted(semantic+selected,key=lambda f:float(f['timestamp'])),selected


def execute_object_query(video_id: str, req: ObjectQueryRequest, pipeline):
    from src.storage.index_store import IndexStore
    from src.retrieval.temporal_retriever import TemporalRetriever
    from src.reasoning.vlm import VisionLanguageModel
    from scripts.track_grounded_object import eligible
    from scripts.select_object_evidence import select_evidence
    idx=INDEX/video_id
    if not (idx/'manifest.json').is_file():raise FileNotFoundError('Index not found')
    video=video_source(video_id)
    retriever,_=IndexStore(idx).load()
    semantic=TemporalRetriever(retriever).search_with_context(
        pipeline.encoder.encode_text([req.question]),top_k=req.top_k,
        context_before=2,context_after=2,max_frames=6)
    if not semantic:raise RuntimeError('No semantic evidence')
    work=OUT/video_id/uuid.uuid4().hex[:12]
    work.mkdir(parents=True,exist_ok=False)
    times=sorted({round(float(f['timestamp']),3) for f in semantic})
    detected=work/'detections.json'; tracked=work/'tracking'; selected_path=work/'evidence.json'
    status='object_not_found'; diagnostic=None; object_rows=[]
    try:
        run_step('scripts.query_object_grounding','--video',str(video),'--question',req.question,
                 '--object',req.object_phrase,'--timestamps',','.join(map(str,times)),
                 '--output',str(detected))
        detection_data=json.loads(detected.read_text(encoding='utf-8'))
        if eligible(detection_data.get('candidates',[])):
            run_step('scripts.track_grounded_object','--detections',str(detected),
                     '--candidate-index','0','--before',str(req.before),'--after',str(req.after),
                     '--output',str(tracked))
            trajectory=json.loads((tracked/'trajectory.json').read_text(encoding='utf-8'))
            selection=select_evidence(trajectory,budget=req.evidence_budget)
            selected_path.write_text(json.dumps(selection,indent=2),encoding='utf-8')
            object_rows=selection.get('evidence',[])
            status='object_evidence_found' if object_rows else 'object_not_visible'
    except (RuntimeError,subprocess.TimeoutExpired) as exc:
        status='object_pipeline_error';diagnostic=str(exc)
    frames,added=merge_frames(semantic,object_rows,video,work)
    if pipeline.vlm is None:pipeline.vlm=VisionLanguageModel()
    # Keep original grounded answer/citation protocol; no new citation parser.
    answer=pipeline.vlm.answer_video_frames_grounded(frames=frames,question=req.question)
    return {'video_id':video_id,'question':req.question,'object_phrase':req.object_phrase,
            'answer':answer['answer'],'cited_frame_ids':answer['cited_frame_ids'],
            'grounding_status':answer['grounding_status'],'citation_parse_ok':answer['citation_parse_ok'],
            'object_status':status,'object_error':diagnostic,'object_evidence_count':len(added),
            'frames':frames,'run_id':work.name,
            'limitations':['Object instance chosen by confidence; not resolved against referring expression',
                           'Temporal frame selection does not verify action semantics',
                           'Object-specific inference can be slow; no persistent object cache']}

@router.post('/api/videos/{video_id}/query-object')
def query_object(video_id: str, request: ObjectQueryRequest):
    # Reuse production GPU lock and lazy pipeline; no concurrent model calls.
    from src.api.main import _query_lock, get_pipeline
    try:
        with _query_lock:
            return execute_object_query(video_id,request,get_pipeline())
    except (ValueError,FileNotFoundError) as exc:
        raise HTTPException(status_code=422,detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500,detail=f'Object-aware query failed: {exc}')
