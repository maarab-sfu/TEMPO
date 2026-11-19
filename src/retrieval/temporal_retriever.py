class TemporalRetriever:
    def __init__(self, faiss_retriever):
        self.retriever = faiss_retriever

    def search_with_context(
        self,
        query_embedding,
        top_k=1,
        context_before=2,
        context_after=2,
        max_frames=None,
    ):
        """Retrieve CLIP anchors and nearby sampled frames, preserving anchors.

        With max_frames set, anchor frames are always retained. Remaining slots
        are assigned to neighbors round-robin across anchors, nearest first.
        Results are returned chronologically; the stored metadata is not changed.
        """
        if top_k < 1 or context_before < 0 or context_after < 0:
            raise ValueError("top_k must be positive and context sizes nonnegative")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")

        anchors = self.retriever.search(query_embedding, top_k=top_k)
        metadata = self.retriever.metadata
        ordered = sorted(enumerate(metadata), key=lambda pair: pair[1]["timestamp"])
        position_by_index = {index: pos for pos, (index, _) in enumerate(ordered)}
        index_by_path = {item["image_path"]: i for i, item in enumerate(metadata)}

        anchor_indices = []
        anchor_details = {}
        for rank, anchor in enumerate(anchors, start=1):
            index = index_by_path.get(anchor["image_path"])
            if index is not None and index not in anchor_indices:
                anchor_indices.append(index)
                anchor_details[index] = {
                    "evidence_role": "semantic_anchor",
                    "anchor_rank": rank,
                    "similarity": anchor.get("similarity"),
                }

        if max_frames is not None and len(anchor_indices) > max_frames:
            raise ValueError(
                "max_frames must be at least the number of distinct retrieved anchors"
            )

        selected = set(anchor_indices)
        # Alternate left/right at each distance; distribute slots across anchors.
        neighbors_by_anchor = []
        for index in anchor_indices:
            position = position_by_index[index]
            neighbors = []
            for distance in range(1, max(context_before, context_after) + 1):
                if distance <= context_before and position - distance >= 0:
                    neighbors.append(ordered[position - distance][0])
                if distance <= context_after and position + distance < len(ordered):
                    neighbors.append(ordered[position + distance][0])
            neighbors_by_anchor.append(neighbors)

        if max_frames is None:
            for neighbors in neighbors_by_anchor:
                selected.update(neighbors)
        else:
            # Round-robin ensures one anchor's context does not monopolize budget.
            offsets = [0] * len(neighbors_by_anchor)
            while len(selected) < max_frames:
                added = False
                for i, neighbors in enumerate(neighbors_by_anchor):
                    while offsets[i] < len(neighbors) and neighbors[offsets[i]] in selected:
                        offsets[i] += 1
                    if offsets[i] < len(neighbors):
                        selected.add(neighbors[offsets[i]])
                        offsets[i] += 1
                        added = True
                    if len(selected) >= max_frames:
                        break
                if not added:
                    break

        results = []
        for index in selected:
            item = metadata[index].copy()
            # Avoid mutating stored metadata or inheriting a stale score.
            item.pop("similarity", None)
            item.update(anchor_details.get(index, {
                "evidence_role": "temporal_context",
                "anchor_rank": None,
                "similarity": None,
            }))
            results.append(item)
        return sorted(results, key=lambda item: item["timestamp"])
