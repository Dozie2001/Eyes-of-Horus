"""
Clip upload service — uploads video clips and snapshots to Supabase Storage.

Runs uploads in a background thread so they don't block the alert pipeline.
Falls back gracefully if Supabase is not configured.

Usage:
    from agent.clip_upload import ClipUploader

    uploader = ClipUploader(supabase_url, service_key, bucket="clips")
    url = uploader.upload(local_path, remote_path="org1/cam1/alert_xxx.mp4")
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class ClipUploader:
    """Uploads clips to Supabase Storage."""

    def __init__(self, supabase_url: str, service_key: str,
                 bucket: str = "clips"):
        self._url = supabase_url
        self._key = service_key
        self._bucket = bucket
        self._client = None

        if self.is_configured():
            try:
                from supabase import create_client
                self._client = create_client(supabase_url, service_key)
                logger.info(f"ClipUploader connected to Supabase Storage (bucket: {bucket})")
            except Exception as e:
                logger.warning(f"ClipUploader: Failed to init Supabase client: {e}")

    def is_configured(self) -> bool:
        return bool(self._url) and bool(self._key)

    def upload(self, local_path: str, remote_path: str) -> Optional[str]:
        """
        Upload a file to Supabase Storage synchronously.

        Args:
            local_path: path to local file
            remote_path: path within the bucket (e.g. "org1/cam1/file.mp4")

        Returns:
            Public URL string, or None on failure.
        """
        if not self._client:
            return None

        if not os.path.exists(local_path):
            logger.warning(f"ClipUploader: File not found: {local_path}")
            return None

        content_type = "video/mp4" if local_path.endswith(".mp4") else "image/jpeg"

        try:
            with open(local_path, "rb") as f:
                self._client.storage.from_(self._bucket).upload(
                    file=f,
                    path=remote_path,
                    file_options={"content-type": content_type, "upsert": "false"},
                )

            public_url = self._client.storage.from_(self._bucket).get_public_url(
                remote_path
            )
            logger.debug(f"ClipUploader: Uploaded {remote_path}")
            return public_url

        except Exception as e:
            # Don't fail the alert pipeline if upload fails
            logger.warning(f"ClipUploader: Upload failed for {remote_path}: {e}")
            return None

    def upload_async(self, local_path: str, remote_path: str,
                     callback=None):
        """
        Upload in a background thread. Optionally calls callback(url) when done.

        Args:
            local_path: path to local file
            remote_path: path within the bucket
            callback: optional function(url: str | None) called with the result
        """
        def _do():
            url = self.upload(local_path, remote_path)
            if callback:
                callback(url)

        thread = threading.Thread(target=_do, daemon=True)
        thread.start()
        return thread

    def build_remote_path(self, org_id: str, camera_id: str,
                          filename: str) -> str:
        """Build the standard remote path: org_id/camera_id/filename."""
        return f"{org_id}/{camera_id}/{filename}"
