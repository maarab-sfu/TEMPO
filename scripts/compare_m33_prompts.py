"""Compare original and structured Qwen prompts using identical retrieved frames."""
from pathlib import Path
from src.pipeline.video_rag import VideoRAGPipeline
from src.storage.index_store import IndexStore
from src.retrieval.temporal_retriever import TemporalRetriever
from src.reasoning.vlm import VisionLanguageModel, parse_grounded_response

root = Path('data/indexes')
indexes = sorted(p.parent.parent for p in root.glob('*/frames/metadata.json') if (p.parent.parent/'manifest.json').is_file())
if not indexes:
    raise RuntimeError('No ready index directories found')
print('Available index IDs:', [p.name for p in indexes])
chosen = input('Enter index ID (Enter for first): ').strip()
index_dir = root/chosen if chosen else indexes[0]
if index_dir not in indexes:
    raise ValueError(f'Unknown index ID: {chosen}')
question = input('Question [What is the type of the knife being used in this video?]: ').strip() or 'What is the type of the knife being used in this video?'
pipeline = VideoRAGPipeline()
retriever, _ = IndexStore(index_dir).load()
embedding = pipeline.encoder.encode_text([question])
frames = TemporalRetriever(retriever).search_with_context(embedding, top_k=2, context_before=2, context_after=2, max_frames=8)
print('Evidence:', [(f['frame_id'], f['timestamp']) for f in frames])
vlm = VisionLanguageModel()
print('\n--- Original prompt ---', flush=True)
original = vlm.answer_video_frames(frames, question, structured=False)
print('RAW ORIGINAL:', repr(original), flush=True)
print('\n--- M3.3 structured prompt ---', flush=True)
structured = vlm.answer_video_frames(frames, question, structured=True)
print('RAW STRUCTURED:', repr(structured), flush=True)
print('PARSED:', repr(parse_grounded_response(structured, frames)), flush=True)
