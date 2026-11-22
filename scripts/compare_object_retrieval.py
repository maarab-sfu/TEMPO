"""Compare frame-only and object-augmented retrieval without running a VLM.

This diagnostic does not measure answer correctness or grounding quality.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.object_hybrid import hybrid_select, rank_object_hits, search_index
from src.retrieval.temporal_retriever import TemporalRetriever
from src.storage.index_store import IndexStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--object-index", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--object-slots", type=int, default=2)
    args = parser.parse_args()

    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.object_slots < 0:
        parser.error("--object-slots cannot be negative")
    if not (args.index_dir / "manifest.json").is_file():
        parser.error(f"Frame index manifest not found: {args.index_dir / 'manifest.json'}")
    if not args.object_index.is_file():
        parser.error(f"Object index not found: {args.object_index}")
    return args


def frame_summary(frame: dict[str, Any]) -> dict[str, Any]:
    """Extract evidence fields relevant to the comparison."""
    return {
        "timestamp": frame["timestamp"],
        "role": frame.get("evidence_role"),
        "track": frame.get("object_track_id"),
        "object_label": frame.get("object_label"),
        "object_observation_timestamp": frame.get("object_observation_timestamp"),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    encoder = CLIPEncoder()
    frame_index, _ = IndexStore(str(args.index_dir)).load()
    query_embedding = encoder.encode_text([args.question])

    temporal_retriever = TemporalRetriever(frame_index)
    baseline = temporal_retriever.search_with_context(
        query_embedding,
        top_k=2,
        context_before=2,
        context_after=2,
        max_frames=args.max_frames,
    )

    object_hits = search_index(str(args.object_index), query_embedding, top_k=100)
    eligible_hits = rank_object_hits(object_hits, args.question)

    hybrid = hybrid_select(
        baseline,
        eligible_hits,
        frame_index.metadata,
        max_frames=args.max_frames,
        object_slots=args.object_slots,
    )

    baseline_paths = {str(frame["image_path"]) for frame in baseline}
    hybrid_paths = {str(frame["image_path"]) for frame in hybrid}

    return {
        "question": args.question,
        "baseline": [frame_summary(frame) for frame in baseline],
        "object_hits": object_hits[:10],
        "eligible_object_hits": eligible_hits[:10],
        "hybrid": [frame_summary(frame) for frame in hybrid],
        "added_frames": len(hybrid_paths - baseline_paths),
        "removed_frames": len(baseline_paths - hybrid_paths),
        "retained_frames": len(baseline_paths & hybrid_paths),
    }


def main() -> None:
    args = parse_args()
    report = compare(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
