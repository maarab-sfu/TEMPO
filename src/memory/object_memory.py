"""M4.1 experimental frame-level semantic memory; NOT object detection/tracking."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA_VERSION = 1


def build_memory(metadata, image_vectors, labels, text_vectors, *, top_labels=2, min_similarity=0.22, max_gap_seconds=4.1):
    """Associate CLIP whole-frame label matches into contiguous *presence episodes*.

    Episodes are NOT object identities. Thresholds are experimental and require
    manual calibration; scores are cosine similarities, not probabilities.
    """
    if not labels or len(set(labels)) != len(labels) or any(not str(x).strip() for x in labels):
        raise ValueError('Provide nonempty unique labels')
    if top_labels < 1 or max_gap_seconds <= 0:
        raise ValueError('Invalid top_labels or max_gap_seconds')
    images = np.asarray(image_vectors, dtype=np.float32)
    texts = np.asarray(text_vectors, dtype=np.float32)
    if images.ndim != 2 or texts.shape != (len(labels), images.shape[1]) or len(metadata) != len(images):
        raise ValueError('Embedding/metadata dimensions mismatch')
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError('Nonfinite embeddings')
    def normalize(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    scores = normalize(images) @ normalize(texts).T
    observations = []
    for idx in sorted(range(len(metadata)), key=lambda i: (float(metadata[i]['timestamp']), i)):
        frame = metadata[idx]
        for j in np.argsort(-scores[idx], kind='stable')[:top_labels]:
            score = float(scores[idx, j])
            if score >= min_similarity:
                observations.append({'frame_id': frame['frame_id'], 'timestamp': float(frame['timestamp']),
                                     'label': str(labels[j]), 'similarity': round(score, 6)})
    episodes = []
    by_label = {label: [] for label in labels}
    for obs in observations:
        by_label[obs['label']].append(obs)
    for label in labels:
        current = []
        for obs in by_label[label]:
            if current and obs['timestamp'] - current[-1]['timestamp'] > max_gap_seconds:
                episodes.append(_episode(label, current))
                current = []
            current.append(obs)
        if current:
            episodes.append(_episode(label, current))
    episodes.sort(key=lambda e: (e['start_seconds'], e['label']))
    return {'schema_version': SCHEMA_VERSION, 'kind': 'frame_semantic_presence',
            'limitations': 'Whole-frame CLIP matches; no bounding boxes, identities, or object tracking.',
            'parameters': {'top_labels': top_labels, 'min_similarity': min_similarity,
                           'max_gap_seconds': max_gap_seconds},
            'labels': list(labels), 'observations': observations, 'episodes': episodes}


def _episode(label, observations):
    return {'episode_id': f'episode_{label}_{observations[0]["frame_id"]}',
            'label': label, 'start_seconds': observations[0]['timestamp'],
            'end_seconds': observations[-1]['timestamp'],
            'frame_ids': [o['frame_id'] for o in observations],
            'max_similarity': max(o['similarity'] for o in observations),
            'observation_count': len(observations)}


def save_memory(path, memory):
    """Atomically publish a sidecar JSON, leaving the FAISS index untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.object_memory_', suffix='.tmp', delete=False) as f:
            name = f.name
            json.dump(memory, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
