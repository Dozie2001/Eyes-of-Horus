"""
SAM3 segmentation via Roboflow Inference API.

Supports two backends via config:
- "serverless": Roboflow's hosted SAM3 at serverless.roboflow.com (works on
                any machine, requires ROBOFLOW_API_KEY, needs internet)
- "local":      Self-hosted GPU Docker at localhost:9001 (fully offline,
                requires NVIDIA GPU — used for cloud deployments)

Same endpoints, same request/response format, only the base URL differs.

Two operations:
- segment_by_text(frame, text)   -> list of polygons  (/sam3/concept_segment)
- segment_by_click(frame, point) -> list of polygons  (/sam3/visual_segment)

A polygon is list[list[float]] — [[x, y], [x, y], ...].

Usage:
    seg = RoboflowSegmenter(
        provider="serverless",
        api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    seg.load()
    polygons = seg.segment_by_text(frame, "driveway")
    polygons = seg.segment_by_click(frame, (450, 320))
"""

import base64
import logging
from typing import Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_LOCAL_URL = "http://localhost:9001"
DEFAULT_SERVERLESS_URL = "https://serverless.roboflow.com"


class RoboflowSegmenter:
    """HTTP client for Roboflow SAM3 endpoints (serverless or local Docker)."""

    def __init__(
        self,
        provider: str = "serverless",
        api_key: str = "",
        local_url: str = DEFAULT_LOCAL_URL,
        serverless_url: str = DEFAULT_SERVERLESS_URL,
        confidence_threshold: float = 0.5,
        timeout_seconds: float = 120.0,
    ):
        """
        Args:
            provider: "serverless" | "local"
            api_key: Roboflow API key. Required for serverless. Passed as
                     query param on every request.
            local_url: base URL for self-hosted Docker (when provider=local)
            serverless_url: base URL for Roboflow hosted API
            confidence_threshold: default output_prob_thresh for text prompts
            timeout_seconds: HTTP timeout for inference calls
        """
        self.provider = provider
        self.api_key = api_key
        self.local_url = local_url.rstrip("/")
        self.serverless_url = serverless_url.rstrip("/")
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self._client: Optional[httpx.Client] = None

    # ----- lifecycle -----

    def load(self) -> None:
        """Create the HTTP client and verify provider reachability."""
        self._client = httpx.Client(timeout=self.timeout_seconds)

        base = self._base_url()
        if self.provider == "serverless" and not self.api_key:
            raise ValueError(
                "RoboflowSegmenter: provider=serverless requires an API key. "
                "Set ROBOFLOW_API_KEY in .env."
            )

        # Light health check. For serverless we just try reaching the root
        # (it may 200 or 404, either proves connectivity). For local we
        # hit the / route which the inference server owns.
        try:
            resp = self._client.get(base)
            # 2xx OR 404 both prove the host is alive
            if resp.status_code >= 500:
                raise ConnectionError(
                    f"Roboflow segmenter backend {base} returned {resp.status_code}"
                )
            logger.info(
                f"RoboflowSegmenter loaded (provider={self.provider}, base={base})"
            )
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"RoboflowSegmenter: cannot reach {base}: {e}"
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ----- public API -----

    def segment_by_text(
        self,
        frame: np.ndarray,
        text: str,
        confidence: Optional[float] = None,
    ) -> list[list[list[float]]]:
        """
        Segment regions matching a natural language prompt.

        Args:
            frame: BGR image (numpy array)
            text: concept to find (e.g. "driveway", "gate", "parking lot")
            confidence: override the default output_prob_thresh

        Returns:
            List of polygons. Each polygon is a list of [x, y] points.
            Empty list if nothing matched or the call failed.
        """
        if self._client is None:
            raise RuntimeError("RoboflowSegmenter not loaded. Call load() first.")

        payload = {
            "image": {"type": "base64", "value": self._encode_frame(frame)},
            "prompts": [{"type": "text", "text": text}],
            "output_prob_thresh": (
                confidence if confidence is not None else self.confidence_threshold
            ),
            "format": "polygon",
        }

        data = self._post("/sam3/concept_segment", payload)
        if data is None:
            return []

        polygons: list[list[list[float]]] = []
        for result in data.get("prompt_results", []):
            for pred in result.get("predictions", []):
                polygons.extend(self._parse_masks(pred.get("masks", [])))
        return polygons

    def segment_by_click(
        self,
        frame: np.ndarray,
        point: tuple[float, float],
        positive: bool = True,
    ) -> list[list[list[float]]]:
        """
        Segment the object under a click point.

        Args:
            frame: BGR image
            point: (x, y) — where the user clicked
            positive: True to include the region; False to exclude

        Returns:
            List of polygons (typically 1 — the object under the click).
        """
        if self._client is None:
            raise RuntimeError("RoboflowSegmenter not loaded. Call load() first.")

        payload = {
            "image": {"type": "base64", "value": self._encode_frame(frame)},
            "prompts": [
                {
                    "points": [
                        {
                            "x": float(point[0]),
                            "y": float(point[1]),
                            "positive": bool(positive),
                        }
                    ]
                }
            ],
            "format": "polygon",
        }

        data = self._post("/sam3/visual_segment", payload)
        if data is None:
            return []

        polygons: list[list[list[float]]] = []
        for pred in data.get("predictions", []):
            polygons.extend(self._parse_masks(pred.get("masks", [])))
        return polygons

    # ----- internals -----

    def _base_url(self) -> str:
        if self.provider == "local":
            return self.local_url
        if self.provider == "serverless":
            return self.serverless_url
        raise ValueError(
            f"Unknown RoboflowSegmenter provider: '{self.provider}'. "
            "Use 'serverless' or 'local'."
        )

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        """POST to a SAM3 endpoint and return the JSON body, or None on failure."""
        url = f"{self._base_url()}{path}"
        params = {"api_key": self.api_key} if self.api_key else None
        try:
            resp = self._client.post(url, params=params, json=payload)  # type: ignore[union-attr]
        except httpx.HTTPError as e:
            logger.error(f"RoboflowSegmenter {path} failed: {e}")
            return None

        if resp.status_code != 200:
            logger.error(
                f"RoboflowSegmenter {path} returned {resp.status_code}: "
                f"{resp.text[:300]}"
            )
            return None

        try:
            return resp.json()
        except ValueError:
            logger.error(f"RoboflowSegmenter {path} returned non-JSON body")
            return None

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode a BGR frame as base64 JPEG."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _parse_masks(masks) -> list[list[list[float]]]:
        """
        Normalize the mask field from a SAM3 prediction into a list of
        polygons, where each polygon is a list of [x, y] pairs.

        The API may return masks in different shapes depending on the
        request. Common formats:
            [[[x,y], [x,y], ...]]              single polygon
            [[[x,y], ...], [[x,y], ...]]       multiple polygons
            [[x,y], [x,y], ...]                single polygon, one level deep
        """
        if not masks:
            return []

        # Single polygon (list of points) vs. list-of-polygons.
        # Distinguish by inspecting the first element.
        first = masks[0]

        # If the first element looks like a point [x, y] (two numbers),
        # treat the whole thing as ONE polygon.
        if (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and all(isinstance(v, (int, float)) for v in first)
        ):
            return [[[float(p[0]), float(p[1])] for p in masks]]

        # Otherwise, assume list of polygons.
        out: list[list[list[float]]] = []
        for poly in masks:
            if not poly:
                continue
            out.append([[float(p[0]), float(p[1])] for p in poly])
        return out
