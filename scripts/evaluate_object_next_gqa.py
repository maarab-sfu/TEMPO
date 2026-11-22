"""M4.7 paired baseline/hybrid NExT-GQA evaluation (experimental).

Reuses the existing benchmark's question selection, prompts and scoring.
Does not modify the production retriever or existing benchmark outputs.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluate_next_gqa import (
    citation_metrics,
    correct_letter,
    load_selection,
    mc_prompt,
    parse_choice,
)
from src.retrieval.object_hybrid import hybrid_select, rank_object_hits, search_index


def frame_identity(frame):
    return str(frame["image_path"])


def retrieval_hit(frames, intervals, tolerance):
    return any(
        float(start) - tolerance <= float(frame["timestamp"]) <= float(end) + tolerance
        for frame in frames
        for start, end in intervals
    )


def classify_question(question, object_labels):
    return "object_routed" if rank_object_hits(
        [{"label": label, "track_id": label, "object_similarity": 0.0}
         for label in object_labels], question
    ) else "fallback"


def select_evidence(retriever, encoder, question, sidecar_path, max_frames, object_slots):
    from src.retrieval.temporal_retriever import TemporalRetriever

    query = encoder.encode_text([question])
    baseline = TemporalRetriever(retriever).search_with_context(
        query, top_k=2, context_before=2, context_after=2, max_frames=max_frames
    )
    if not baseline:
        raise RuntimeError("Frame retrieval returned no evidence")

    hits = []
    eligible = []
    if sidecar_path.is_file():
        hits = search_index(sidecar_path, query, top_k=100)
        eligible = rank_object_hits(hits, question)

    hybrid = hybrid_select(
        baseline, eligible, retriever.metadata,
        max_frames=max_frames, object_slots=object_slots
    )
    baseline_ids = {frame_identity(frame) for frame in baseline}
    hybrid_ids = {frame_identity(frame) for frame in hybrid}
    return {
        "baseline": baseline,
        "hybrid": hybrid,
        "object_hits": len(hits),
        "eligible_hits": len(eligible),
        "added_frames": len(hybrid_ids - baseline_ids),
        "removed_frames": len(baseline_ids - hybrid_ids),
        "sidecar_present": sidecar_path.is_file(),
        "question_group": "object_routed" if eligible else "fallback",
    }


def score_answer(frames, response, row, tolerance):
    choice = parse_choice(response["answer"])
    citations = citation_metrics(
        frames, response["cited_frame_ids"], row["evidence_intervals"], tolerance
    )
    return {
        "prediction": choice,
        "correct": choice == correct_letter(row),
        "retrieval_hit": retrieval_hit(frames, row["evidence_intervals"], tolerance),
        "citation_parse_ok": bool(response["citation_parse_ok"]),
        "grounding_status": response["grounding_status"],
        "cited_frame_ids": response["cited_frame_ids"],
        **citations,
    }


def summarize(records, mode):
    valid = [r for r in records if r.get("status") == "ok"]
    if not valid:
        return {"n": 0}
    scored = [r[mode] for r in valid]
    citations = sum(s["citation_count"] for s in scored)
    hits = sum(s["citation_hits"] for s in scored)
    return {
        "n": len(valid),
        "accuracy": sum(s["correct"] for s in scored) / len(valid),
        "retrieval_hit_rate": sum(s["retrieval_hit"] for s in scored) / len(valid),
        "citation_temporal_hit_rate": sum(s["any_citation_in_gt"] for s in scored) / len(valid),
        "citation_temporal_precision": hits / citations if citations else None,
        "joint_correct_and_temporally_grounded_rate": sum(
            s["correct"] and s["any_citation_in_gt"] for s in scored
        ) / len(valid),
        "mean_answer_latency_seconds": sum(
            r[f"{mode}_answer_seconds"] for r in valid
        ) / len(valid),
    }


def report(records, selection, args):
    complete = [r for r in records if r.get("status") == "ok"]
    groups = defaultdict(list)
    for record in complete:
        groups[record["question_group"]].append(record)

    return {
        "dataset": "NExT-GQA",
        "seed": selection.get("seed"),
        "selected_questions": len(records),
        "completed_pairs": len(complete),
        "failed": len(records) - len(complete),
        "tolerance_seconds": args.tolerance,
        "object_slots": args.object_slots,
        "max_frames": args.max_frames,
        "sidecars_present": sum(r.get("sidecar_present", False) for r in records),
        "changed_retrieval_pairs": sum(r.get("added_frames", 0) > 0 for r in complete),
        "baseline": summarize(complete, "baseline"),
        "hybrid": summarize(complete, "hybrid"),
        "by_question_group": {
            group: {
                "baseline": summarize(items, "baseline"),
                "hybrid": summarize(items, "hybrid"),
            }
            for group, items in sorted(groups.items())
        },
        "notes": [
            "Temporal citation overlap is not proof of semantic grounding.",
            "Question groups use explicit object-label keyword routing, not semantic classification.",
            "Answer latency excludes initial index load and CLIP retrieval.",
            "Pairs run sequentially (baseline first); model warmup may bias latency.",
        ],
    }


def evaluate_pair(pipeline, row, args):
    from src.reasoning.vlm import VisionLanguageModel
    from src.storage.index_store import IndexStore

    video_id = str(row["video_id"])
    index_dir = args.index_root / f"nextgqa_{video_id}"
    retriever, _ = IndexStore(index_dir).load()
    selection = select_evidence(
        retriever, pipeline.encoder, row["question"],
        index_dir / args.sidecar_name, args.max_frames, args.object_slots
    )
    if pipeline.vlm is None:
        pipeline.vlm = VisionLanguageModel()

    result = {
        "video_id": video_id,
        "qid": str(row["qid"]),
        "type": row["type"],
        "question": row["question"],
        "question_group": selection["question_group"],
        "sidecar_present": selection["sidecar_present"],
        "added_frames": selection["added_frames"],
        "removed_frames": selection["removed_frames"],
        "eligible_hits": selection["eligible_hits"],
    }
    for mode in ("baseline", "hybrid"):
        frames = selection[mode]
        start = time.perf_counter()
        response = pipeline.vlm.answer_video_frames_grounded(frames, mc_prompt(row))
        result[f"{mode}_answer_seconds"] = round(time.perf_counter() - start, 3)
        result[mode] = score_answer(frames, response, row, args.tolerance)
        result[f"{mode}_timestamps"] = [float(f["timestamp"]) for f in frames]
    result["status"] = "ok"
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path,
        default=Path("data/datasets/next_gqa/manifests/selection.json")
    )
    parser.add_argument("--index-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/next_gqa/m47"))
    parser.add_argument("--sidecar-name", default="object_m46.npz")
    parser.add_argument("--limit", type=int, default=2, help="0 means all questions")
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--object-slots", type=int, default=2)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.tolerance < 0 or args.max_frames < 1 or args.object_slots < 0:
        parser.error("Invalid negative limit/tolerance/slots or max-frames below 1")
    return args


def main():
    args = parse_args()
    selection, questions = load_selection(args.selection)
    if args.limit:
        questions = questions[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"seed{selection.get('seed', 'unknown')}_n{len(questions)}"
        f"_slots{args.object_slots}_frames{args.max_frames}_tol{args.tolerance:g}"
    )
    journal = args.output_dir / f"{run_id}.jsonl"
    existing = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[(record["video_id"], record["qid"])] = record

    pipeline = None
    for number, row in enumerate(questions, 1):
        key = (str(row["video_id"]), str(row["qid"]))
        if key in existing and (existing[key]["status"] == "ok" or not args.retry_errors):
            print(f"[{number}/{len(questions)}] SKIP {key}", flush=True)
            continue
        print(f"[{number}/{len(questions)}] {key}", flush=True)
        try:
            if pipeline is None:
                from src.pipeline.video_rag import VideoRAGPipeline
                pipeline = VideoRAGPipeline()
            record = evaluate_pair(pipeline, row, args)
            print(
                f"  baseline={record['baseline']['correct']} "
                f"hybrid={record['hybrid']['correct']} "
                f"added={record['added_frames']}",
                flush=True,
            )
        except Exception as exc:
            record = {
                "video_id": key[0], "qid": key[1], "type": row["type"],
                "status": "error", "error": f"{type(exc).__name__}: {exc}"
            }
            print(f"  ERROR: {record['error']}", flush=True)
        record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing[key] = record

    ordered = [
        existing[(str(row["video_id"]), str(row["qid"]))]
        for row in questions
        if (str(row["video_id"]), str(row["qid"])) in existing
    ]
    summary = report(ordered, selection, args)
    output = args.output_dir / f"{run_id}_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {journal}\nSummary: {output}")


if __name__ == "__main__":
    main()
