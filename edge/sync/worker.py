"""
Sync worker for Eyes of Horus edge device.

Pushes events, decisions, and media to the cloud API when online.
Runs as a background thread (same pattern as EvalAgent).

Built in Phase 5.
"""

# TODO: Implement SyncWorker
# - Background thread that polls for unsynced records (synced_at IS NULL)
# - Uploads media to R2 via cloud API
# - POSTs events/decisions to cloud sync endpoint
# - Marks records as synced (sets synced_at)
# - Sends heartbeat every N seconds
# - Handles offline gracefully (queues until online)
