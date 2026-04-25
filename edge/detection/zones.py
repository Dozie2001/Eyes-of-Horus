"""
Zone checking for StangWatch.

Zones are named regions on a camera view. They can be:
- POLYGON: closed region defined by 3+ points. A detection is "in" the zone
          if its anchor (default: bbox centroid) falls inside the polygon.
- LINE:    directed line segment with start + end. A detection "crosses" the
          line when it is seen on the opposite side from the previous frame.
          Line zones require track IDs (stateful) and are handled via
          supervision.LineZone.

Zone dict schema (stored inside CameraProfile.zones_json):
    {
        "name": "front_gate",                   required
        "zone_type": "polygon" | "line",        defaults to "polygon"
        "points": [[x, y], ...],                required — 3+ for polygon, 2 for line

        # optional rule fields (read by the AI agent for severity reasoning)
        "severity_override": "low" | "medium" | "high",
        "active_hours_start": "HH:MM",
        "active_hours_end": "HH:MM",
        "allowed_object_types": ["person", "vehicle"],
        "alert_on_entry": bool,
        "alert_on_dwell_seconds": float,
    }

Polygon-in-point tests use cv2.pointPolygonTest (battle-tested, already a
dependency). Line crossings use supervision.LineZone which handles state.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


ZONE_TYPE_POLYGON = "polygon"
ZONE_TYPE_LINE = "line"


def _normalize_points(points):
    """Return a numpy int32 array of [[x, y], ...] or None if invalid."""
    if not points:
        return None
    try:
        arr = np.array(points, dtype=np.int32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def _zone_type(zone: dict) -> str:
    """Infer zone type. Defaults to polygon if not specified."""
    t = zone.get("zone_type")
    if t in (ZONE_TYPE_POLYGON, ZONE_TYPE_LINE):
        return t
    return ZONE_TYPE_POLYGON


def point_in_polygon(point, polygon) -> bool:
    """Check whether (x, y) is inside the given polygon.

    Args:
        point: (x, y) tuple
        polygon: list of [x, y] points OR a numpy int32 array

    Returns:
        True if point is strictly inside or on the edge of the polygon.
    """
    if isinstance(polygon, np.ndarray):
        poly = polygon
    else:
        poly = _normalize_points(polygon)
    if poly is None or len(poly) < 3:
        return False

    # cv2.pointPolygonTest returns >0 inside, <0 outside, =0 on edge.
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


class ZoneChecker:
    """
    Stateful zone checker for a single camera.

    Holds parsed polygons and supervision.LineZone instances. Call update()
    once per frame with the current tracked detections (as sv.Detections)
    to refresh line-crossing state, then query check_point() or
    check_detection() for zone membership.

    Usage:
        checker = ZoneChecker(zones, frame_resolution_wh=(1920, 1080))
        # per frame:
        checker.update(sv_detections)
        hits = checker.check_point((cx, cy))
        # hits = [{"name": "gate", "zone_type": "polygon", "triggered": True,
        #          "severity_override": "high", ...}]
    """

    def __init__(self, zones: list[dict], frame_resolution_wh: tuple[int, int] | None = None):
        """
        Args:
            zones: list of zone dicts (see schema at top of this file)
            frame_resolution_wh: (width, height) of the camera frame.
                Required only if you plan to use supervision.PolygonZone
                trigger() — not needed for cv2.pointPolygonTest checks.
        """
        self.frame_resolution_wh = frame_resolution_wh
        self._polygon_zones: list[dict] = []  # {name, points (np array), rules}
        self._line_zones: list[dict] = []     # {name, line (sv.LineZone), rules}

        for zone in zones or []:
            name = zone.get("name", "unnamed")
            zt = _zone_type(zone)
            rules = self._extract_rules(zone)

            if zt == ZONE_TYPE_POLYGON:
                pts = _normalize_points(zone.get("points", []))
                if pts is None or len(pts) < 3:
                    logger.warning(
                        f"ZoneChecker: polygon zone '{name}' skipped — "
                        f"needs at least 3 points, got {0 if pts is None else len(pts)}"
                    )
                    continue
                self._polygon_zones.append(
                    {"name": name, "points": pts, "rules": rules}
                )

            elif zt == ZONE_TYPE_LINE:
                pts = zone.get("points", [])
                if not pts or len(pts) < 2:
                    logger.warning(
                        f"ZoneChecker: line zone '{name}' skipped — "
                        f"needs 2 points, got {len(pts)}"
                    )
                    continue
                try:
                    import supervision as sv
                    start = sv.Point(x=int(pts[0][0]), y=int(pts[0][1]))
                    end = sv.Point(x=int(pts[1][0]), y=int(pts[1][1]))
                    line = sv.LineZone(start=start, end=end)
                    self._line_zones.append(
                        {"name": name, "line": line, "rules": rules,
                         "last_in": 0, "last_out": 0}
                    )
                except Exception as e:
                    logger.warning(f"ZoneChecker: line zone '{name}' init failed: {e}")

    @staticmethod
    def _extract_rules(zone: dict) -> dict:
        """Pick out the rule fields from a zone dict."""
        return {
            "severity_override": zone.get("severity_override"),
            "active_hours_start": zone.get("active_hours_start"),
            "active_hours_end": zone.get("active_hours_end"),
            "allowed_object_types": zone.get("allowed_object_types"),
            "alert_on_entry": zone.get("alert_on_entry", False),
            "alert_on_dwell_seconds": zone.get("alert_on_dwell_seconds"),
        }

    def has_zones(self) -> bool:
        return bool(self._polygon_zones) or bool(self._line_zones)

    def update(self, sv_detections) -> list[dict]:
        """
        Feed the current frame's tracked detections to line zones so
        they can track in/out crossings.

        Args:
            sv_detections: supervision.Detections with tracker_id set

        Returns:
            List of line-crossing events emitted this frame, e.g.
            [{"name": "perimeter", "direction": "in", "rules": {...}}]
        """
        events: list[dict] = []
        if not self._line_zones or sv_detections is None:
            return events

        for lz in self._line_zones:
            line = lz["line"]
            try:
                line.trigger(sv_detections)
            except Exception as e:
                logger.debug(f"LineZone '{lz['name']}' trigger failed: {e}")
                continue

            # Detect new crossings since last frame
            new_in = int(line.in_count) - lz["last_in"]
            new_out = int(line.out_count) - lz["last_out"]
            if new_in > 0:
                events.append({
                    "name": lz["name"],
                    "direction": "in",
                    "count": new_in,
                    "rules": lz["rules"],
                })
            if new_out > 0:
                events.append({
                    "name": lz["name"],
                    "direction": "out",
                    "count": new_out,
                    "rules": lz["rules"],
                })
            lz["last_in"] = int(line.in_count)
            lz["last_out"] = int(line.out_count)

        return events

    def check_point(self, point: tuple[float, float]) -> list[dict]:
        """Return rule-enriched info for every polygon zone containing the point.

        Args:
            point: (x, y) — typically the centroid of a detection's bbox.

        Returns:
            List of dicts with name, zone_type, and all rule fields.
        """
        hits: list[dict] = []
        for pz in self._polygon_zones:
            if point_in_polygon(point, pz["points"]):
                hits.append({
                    "name": pz["name"],
                    "zone_type": ZONE_TYPE_POLYGON,
                    **pz["rules"],
                })
        return hits

    def check_bbox(self, bbox: list[float]) -> list[dict]:
        """Convenience — same as check_point but takes a bbox [x1, y1, x2, y2]."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return self.check_point((cx, cy))

    def zone_names(self) -> list[str]:
        """Names of all zones this checker knows about (polygon + line)."""
        return [pz["name"] for pz in self._polygon_zones] + \
               [lz["name"] for lz in self._line_zones]
