"""
Clip search index — embeds decisions and searches via Supabase pgvector.

Handles:
1. Indexing: embed a decision's reason+description, store vector in Supabase
2. Searching: embed a query, call Supabase RPC for similarity search

Usage:
    from agent.clip_index import ClipIndex

    index = ClipIndex(config)
    index.index_decision(decision_id, reason, description, metadata)
    results = index.search("man with a bag near the gate", limit=10)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ClipIndex:
    """Embeds and searches clip descriptions via Supabase pgvector."""

    def __init__(self, config):
        self._config = config
        self._embedder = None
        self._client = None

        # Build embedding provider
        from agent.embeddings import build_embedding_provider
        self._embedder = build_embedding_provider(config)
        if self._embedder:
            logger.info(f"ClipIndex: embedding provider = {self._embedder.model_name} ({self._embedder.dimensions}d)")
        else:
            logger.warning("ClipIndex: no embedding provider configured, search disabled")

        # Build Supabase client
        supa_cfg = config.supabase
        if supa_cfg.enabled and supa_cfg.url and supa_cfg.service_key:
            try:
                from supabase import create_client
                self._client = create_client(supa_cfg.url, supa_cfg.service_key)
                logger.info("ClipIndex: connected to Supabase")
            except Exception as e:
                logger.warning(f"ClipIndex: failed to connect to Supabase: {e}")

        # Check if stored embeddings are from a different model — auto re-embed
        if self.is_ready():
            self._check_model_mismatch()

    def is_ready(self) -> bool:
        """Check if both embedding provider and Supabase are available."""
        return self._embedder is not None and self._client is not None

    def _check_model_mismatch(self):
        """If stored embeddings use a different model, re-embed only stale rows in background."""
        try:
            current_model = self._embedder.model_name

            # Count how many rows have a DIFFERENT model (stale embeddings)
            stale = self._client.table("cloud_decision").select(
                "id", count="exact"
            ).eq("alert", True).not_.is_(
                "embedding", "null"
            ).neq("embedding_model", current_model).execute()

            stale_count = stale.count or 0
            if stale_count == 0:
                return  # all embeddings are current, nothing to do

            logger.warning(
                f"ClipIndex: {stale_count} decisions embedded with a different model. "
                f"Auto re-embedding to {current_model} in background..."
            )
            import threading
            threading.Thread(
                target=self._auto_reembed,
                daemon=True,
                name="clip-reembed",
            ).start()

        except Exception as e:
            logger.warning(f"ClipIndex: model mismatch check failed: {e}")

    def _auto_reembed(self):
        """Background re-embed triggered by model mismatch. Only updates stale rows."""
        try:
            stats = self.reembed_stale()
            logger.info(f"ClipIndex: auto re-embed complete — {stats['updated']} updated, {stats['failed']} failed")
        except Exception as e:
            logger.error(f"ClipIndex: auto re-embed failed: {e}")

    def index_decision(self, cloud_decision_id: int, reason: str,
                       description: str = "", **extra_fields) -> bool:
        """
        Embed a decision and update its vector in Supabase.

        Args:
            cloud_decision_id: the row ID in cloud_decision table
            reason: the AI's reason text
            description: the visual description text
            **extra_fields: any other fields to update on the row

        Returns:
            True if embedding was stored, False on failure.
        """
        if not self.is_ready():
            return False

        # Combine reason + description for richer embedding
        text = reason
        if description:
            text = f"{reason}. {description}"

        vector = self._embedder.embed(text)
        if vector is None:
            logger.warning(f"ClipIndex: embedding failed for decision {cloud_decision_id}")
            return False

        try:
            update_data = {
                "embedding": vector,
                "embedding_model": self._embedder.model_name,
                **extra_fields,
            }
            self._client.table("cloud_decision").update(
                update_data
            ).eq("id", cloud_decision_id).execute()

            logger.debug(f"ClipIndex: indexed decision {cloud_decision_id}")
            return True

        except Exception as e:
            logger.warning(f"ClipIndex: failed to store embedding for decision {cloud_decision_id}: {e}")
            return False

    def search(self, query: str, limit: int = 10,
               org_id: str = None, camera_id: str = None,
               after: str = None) -> list[dict]:
        """
        Search for clips matching a natural language query.

        Args:
            query: natural language search text
            limit: max results to return
            org_id: optional org filter
            camera_id: optional camera filter
            after: optional ISO date string filter (only results after this date)

        Returns:
            List of matching decision dicts with similarity scores.
        """
        if not self.is_ready():
            logger.warning("ClipIndex: search called but index not ready")
            return []

        # Embed the query
        if hasattr(self._embedder, 'embed_query'):
            query_vector = self._embedder.embed_query(query)
        else:
            query_vector = self._embedder.embed(query)

        if query_vector is None:
            logger.warning("ClipIndex: failed to embed search query")
            return []

        try:
            result = self._client.rpc("search_clips", {
                "query_text": query,
                "query_embedding": query_vector,
                "match_count": limit,
                "org_filter": org_id,
                "camera_filter": camera_id,
                "after_date": after,
            }).execute()

            if result.data:
                return result.data

            # Fall back to semantic-only if hybrid returns nothing
            result = self._client.rpc("match_clips", {
                "query_embedding": query_vector,
                "match_threshold": 0.3,
                "match_count": limit,
                "org_filter": org_id,
                "camera_filter": camera_id,
                "after_date": after,
            }).execute()

            return result.data or []

        except Exception as e:
            logger.warning(f"ClipIndex: search failed: {e}")
            return []

    def search_semantic_only(self, query: str, limit: int = 10,
                              threshold: float = 0.3,
                              org_id: str = None) -> list[dict]:
        """Pure semantic search — no keyword matching."""
        if not self.is_ready():
            return []

        if hasattr(self._embedder, 'embed_query'):
            query_vector = self._embedder.embed_query(query)
        else:
            query_vector = self._embedder.embed(query)

        if query_vector is None:
            return []

        try:
            result = self._client.rpc("match_clips", {
                "query_embedding": query_vector,
                "match_threshold": threshold,
                "match_count": limit,
                "org_filter": org_id,
            }).execute()
            return result.data or []
        except Exception as e:
            logger.warning(f"ClipIndex: semantic search failed: {e}")
            return []

    def reembed_all(self, batch_size: int = 50) -> dict:
        """
        Re-embed all decisions with the current embedding provider.

        Use this after switching embedding providers to make all vectors
        consistent. Processes in batches to avoid memory/rate limit issues.

        Returns:
            dict with counts: {"total", "updated", "failed", "skipped"}
        """
        if not self.is_ready():
            logger.error("ClipIndex: cannot re-embed — index not ready")
            return {"total": 0, "updated": 0, "failed": 0, "skipped": 0}

        stats = {"total": 0, "updated": 0, "failed": 0, "skipped": 0}
        offset = 0

        while True:
            # Fetch a batch of alert decisions
            result = self._client.table("cloud_decision").select(
                "id, reason, embedding_model"
            ).eq("alert", True).range(offset, offset + batch_size - 1).execute()

            rows = result.data or []
            if not rows:
                break

            stats["total"] += len(rows)

            # Collect texts to embed in batch
            to_embed = []
            for row in rows:
                reason = row.get("reason", "")
                if not reason:
                    stats["skipped"] += 1
                    continue
                to_embed.append((row["id"], reason))

            if to_embed:
                texts = [t[1] for t in to_embed]
                vectors = self._embedder.embed_batch(texts)

                if vectors is None:
                    stats["failed"] += len(to_embed)
                    logger.error(f"ClipIndex: batch embed failed at offset {offset}")
                    break

                for (row_id, _), vector in zip(to_embed, vectors):
                    try:
                        self._client.table("cloud_decision").update({
                            "embedding": vector,
                            "embedding_model": self._embedder.model_name,
                        }).eq("id", row_id).execute()
                        stats["updated"] += 1
                    except Exception as e:
                        stats["failed"] += 1
                        logger.warning(f"ClipIndex: failed to update row {row_id}: {e}")

            offset += batch_size
            logger.info(f"ClipIndex: re-embed progress — {stats['updated']} updated, {stats['failed']} failed, {offset} processed")

        logger.info(f"ClipIndex: re-embed complete — {stats}")
        return stats

    def reembed_stale(self, batch_size: int = 50) -> dict:
        """
        Re-embed only decisions whose embedding_model doesn't match the current provider.

        Skips rows that are already up-to-date. Safe to call repeatedly —
        if everything is current, it does nothing.

        Returns:
            dict with counts: {"total", "updated", "failed", "skipped"}
        """
        if not self.is_ready():
            logger.error("ClipIndex: cannot re-embed — index not ready")
            return {"total": 0, "updated": 0, "failed": 0, "skipped": 0}

        current_model = self._embedder.model_name
        stats = {"total": 0, "updated": 0, "failed": 0, "skipped": 0}
        offset = 0

        while True:
            # Only fetch rows with a different (stale) embedding model
            result = self._client.table("cloud_decision").select(
                "id, reason"
            ).eq("alert", True).not_.is_(
                "embedding", "null"
            ).neq("embedding_model", current_model).range(
                offset, offset + batch_size - 1
            ).execute()

            rows = result.data or []
            if not rows:
                break

            stats["total"] += len(rows)

            to_embed = []
            for row in rows:
                reason = row.get("reason", "")
                if not reason:
                    stats["skipped"] += 1
                    continue
                to_embed.append((row["id"], reason))

            if to_embed:
                texts = [t[1] for t in to_embed]
                vectors = self._embedder.embed_batch(texts)

                if vectors is None:
                    stats["failed"] += len(to_embed)
                    logger.error(f"ClipIndex: stale batch embed failed at offset {offset}")
                    break

                for (row_id, _), vector in zip(to_embed, vectors):
                    try:
                        self._client.table("cloud_decision").update({
                            "embedding": vector,
                            "embedding_model": current_model,
                        }).eq("id", row_id).execute()
                        stats["updated"] += 1
                    except Exception as e:
                        stats["failed"] += 1
                        logger.warning(f"ClipIndex: failed to update row {row_id}: {e}")

            offset += batch_size
            logger.info(f"ClipIndex: stale re-embed progress — {stats['updated']}/{stats['total']}")

        logger.info(f"ClipIndex: stale re-embed complete — {stats}")
        return stats
