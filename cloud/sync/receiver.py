"""
Sync receiver for Eyes of Horus.

Receives events, decisions, and media uploads from edge devices.
Writes to Supabase (Postgres) and R2 (media storage).

Routes:
    POST /sync/events — batch event upload
    POST /sync/decisions — batch decision upload
    POST /sync/media — media file upload (snapshots/clips)
    POST /sync/heartbeat — device health update
"""

# TODO: Implement sync endpoints
# - Validate edge device API key
# - Insert events/decisions into Supabase
# - Upload media to R2
# - Update device heartbeat
