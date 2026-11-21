"""Build experimental semantic presence memory from an existing VideoMind index."""
import argparse
from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.memory.object_memory import build_memory, save_memory
from src.storage.index_store import IndexStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Concrete visually recognizable labels, e.g. knife garlic bowl')
    parser.add_argument('--min-similarity', type=float, default=0.22)
    parser.add_argument('--top-labels', type=int, default=2)
    parser.add_argument('--max-gap-seconds', type=float, default=4.1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output = args.index_dir / 'object_memory_v1.json'
    if output.exists() and not args.overwrite:
        parser.error(f'{output} already exists; pass --overwrite to replace')
    retriever, _ = IndexStore(args.index_dir).load()
    encoder = CLIPEncoder()
    prompts = [f'a video frame showing {label}' for label in args.labels]
    text_vectors = encoder.encode_text(prompts)
    image_vectors = [retriever.index.reconstruct(i) for i in range(retriever.index.ntotal)]
    memory = build_memory(retriever.metadata, image_vectors, args.labels, text_vectors,
                          top_labels=args.top_labels, min_similarity=args.min_similarity,
                          max_gap_seconds=args.max_gap_seconds)
    memory['index_dir_name'] = args.index_dir.name
    save_memory(output, memory)
    print(f'Saved {output}: {len(memory["observations"])} observations, '
          f'{len(memory["episodes"])} episodes')
    print('WARNING: frame-level semantic matches, NOT verified object tracks.')


if __name__ == '__main__':
    main()
