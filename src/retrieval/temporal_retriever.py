class TemporalRetriever:

    def __init__(self, faiss_retriever):
        self.retriever = faiss_retriever

    def search_with_context(
        self,
        query_embedding,
        top_k=1,
        context_before=2,
        context_after=2
    ):
        """
        Retrieve relevant frames and their temporal neighbors.

        Args:
            query_embedding: CLIP text embedding.
            top_k: Number of semantic anchor frames.
            context_before: Number of preceding sampled frames.
            context_after: Number of following sampled frames.
        """

        # Retrieve semantic anchor frames.
        anchors = self.retriever.search(
            query_embedding,
            top_k=top_k
        )

        metadata = self.retriever.metadata

        # Metadata positions correspond to FAISS row positions.
        # Build a chronological view without assuming frame_id
        # values are consecutive.
        ordered = sorted(
            enumerate(metadata),
            key=lambda pair: pair[1]["timestamp"]
        )

        position_by_index = {
            original_index: position
            for position, (original_index, _) in enumerate(ordered)
        }

        selected_indices = set()

        for anchor in anchors:

            # Locate the anchor in the original metadata.
            anchor_index = next(
                (
                    i for i, item in enumerate(metadata)
                    if item["image_path"] == anchor["image_path"]
                ),
                None
            )

            if anchor_index is None:
                continue

            position = position_by_index[anchor_index]

            start = max(
                0,
                position - context_before
            )

            end = min(
                len(ordered),
                position + context_after + 1
            )

            for p in range(start, end):
                original_index, _ = ordered[p]
                selected_indices.add(original_index)

        results = [
            metadata[i].copy()
            for i in selected_indices
        ]

        results.sort(
            key=lambda x: x["timestamp"]
        )

        return results