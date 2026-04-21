"""
Event tracking for StangWatch.
Tracks people across frames using ByteTrack and emits factual events.

The EventTracker does NOT judge what is suspicious — it records facts
(positions, durations, objects, movement). The AI agent decides what matters.

Events emitted:
    appeared        — new person detected
    departed        — person gone for N seconds
    returned        — person came back after departing
    companion       — new person appeared near an existing one
    objects_changed — objects near a person changed
    track_summary   — periodic factual snapshot of an active track (every N seconds)
"""

from datetime import datetime, time
import math

# --- Event type constants ---
EVENT_APPEARED = "appeared"
EVENT_DEPARTED = "departed"
EVENT_RETURNED = "returned"
EVENT_COMPANION = "companion"
EVENT_OBJECTS_CHANGED = "objects_changed"
EVENT_TRACK_SUMMARY = "track_summary"


class TrackedPerson:
    """
    Represents one tracked person across frames.
    Stores position history and factual measurements.
    """

    def __init__(self, track_id, bbox, timestamp):
        self.track_id = track_id
        self.first_seen = timestamp
        self.last_seen = timestamp

        # Position history: list of (x, y) center points
        self.positions = [self._center(bbox)]
        self.current_bbox = bbox

        # Objects detected near this person (from YOLO multi-class)
        self.nearby_objects = set()

        # Frames since last detection (for departure detection)
        self.frames_missing = 0

    def _center(self, bbox):
        """Center point of bounding box [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def update(self, bbox, timestamp):
        """Called each frame this person is detected."""
        self.last_seen = timestamp
        self.current_bbox = bbox
        self.positions.append(self._center(bbox))
        self.frames_missing = 0

        # Keep only last 300 positions (~10 seconds at 30fps)
        if len(self.positions) > 300:
            self.positions = self.positions[-300:]

    def duration_seconds(self):
        """How long this person has been tracked."""
        return (self.last_seen - self.first_seen).total_seconds()

    def recent_movement(self, last_n=30):
        """
        Average pixel distance per frame over the last N positions.
        Low value = standing still. High value = walking/running.
        """
        if len(self.positions) < 2:
            return 0.0

        recent = self.positions[-last_n:]
        total_distance = 0.0

        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            total_distance += math.sqrt(dx * dx + dy * dy)

        return total_distance / (len(recent) - 1)

    def total_distance(self):
        """Total pixels traveled since first seen."""
        total = 0.0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def position_spread(self):
        """
        Max distance from centroid of all positions.
        Low spread = staying in one spot. High spread = roaming the area.
        """
        if len(self.positions) < 2:
            return 0.0

        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        max_dist = 0.0
        for x, y in self.positions:
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if d > max_dist:
                max_dist = d
        return max_dist


class EventTracker:
    """
    Tracks people across frames and emits factual events.
    No judgments — the AI agent decides what's suspicious.

    Usage:
        from events.bus import event_bus

        tracker = EventTracker(event_bus)
        event_bus.on("track_summary", lambda data: print(data))
        tracker.update(detections, timestamp)
    """

    def __init__(self, event_bus, quiet_hours=None, departure_seconds=3.0,
                 companion_distance=200, summary_interval=60.0, camera_id="cam1",
                 zones=None):
        """
        Args:
            event_bus: pyee EventEmitter instance
            quiet_hours: dict {"start": "22:00", "end": "06:00"} or None
            departure_seconds: seconds without seeing a person before marking departed
            companion_distance: max pixels between two people to trigger COMPANION
            summary_interval: seconds between periodic track summaries
            camera_id: identifier for this camera (included in all events)
            zones: list of zone dicts from camera profile (see detection/zones.py
                for the schema). Builds a ZoneChecker internally for queries.
        """
        self.bus = event_bus
        self.camera_id = camera_id
        self.quiet_hours = quiet_hours
        self.departure_seconds = departure_seconds
        self.companion_distance = companion_distance
        self.summary_interval = summary_interval

        from detection.zones import ZoneChecker
        self._zone_checker = ZoneChecker(zones or [])

        # Active tracks: track_id → TrackedPerson
        self.tracks = {}

        # Recently departed (kept briefly for RETURNED detection)
        self.departed_tracks = {}

    def refresh_zones(self, zones):
        """Hot-reload zone definitions without restarting the pipeline."""
        from detection.zones import ZoneChecker
        self._zone_checker = ZoneChecker(zones or [])

        # Last summary emission time per track
        self._last_summary_time = {}

    def update(self, detections, timestamp):
        """
        Process one frame of detections.
        Emits events via the bus for lifecycle changes and periodic summaries.
        """
        seen_track_ids = set()

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            seen_track_ids.add(track_id)
            bbox = det["bbox"]
            nearby_objects = set(det.get("nearby_objects", []))

            if track_id in self.tracks:
                self._update_existing(track_id, bbox, nearby_objects, timestamp)

            elif track_id in self.departed_tracks:
                self._handle_returned(track_id, bbox, timestamp)

            else:
                self._handle_new(track_id, bbox, timestamp)

        # Check for departures
        self._check_departures(seen_track_ids, timestamp)

        # Emit periodic track summaries
        self._check_summaries(timestamp)

        # Clean up old departed tracks
        self._cleanup_departed(timestamp)

    def _update_existing(self, track_id, bbox, nearby_objects, timestamp):
        """Update an existing tracked person. Check for object changes only."""
        person = self.tracks[track_id]
        old_objects = person.nearby_objects.copy()

        person.update(bbox, timestamp)

        # Object changes are factual — emit when the set changes
        if nearby_objects and nearby_objects != old_objects:
            person.nearby_objects = nearby_objects
            event_data = self._build_track_summary(person, timestamp)
            event_data["objects_before"] = list(old_objects)
            event_data["objects_after"] = list(nearby_objects)
            self.bus.emit(EVENT_OBJECTS_CHANGED, event_data)

    def _handle_returned(self, track_id, bbox, timestamp):
        """Handle a person who was departed but came back."""
        person = self.departed_tracks.pop(track_id)
        person.update(bbox, timestamp)
        person.frames_missing = 0
        self.tracks[track_id] = person

        # Reset summary timer for returned track
        self._last_summary_time[track_id] = timestamp

        self.bus.emit(EVENT_RETURNED, self._build_track_summary(person, timestamp))

    def _handle_new(self, track_id, bbox, timestamp):
        """Handle a brand new person."""
        person = TrackedPerson(track_id, bbox, timestamp)
        self.tracks[track_id] = person

        # Set initial summary time so first summary comes after one interval
        self._last_summary_time[track_id] = timestamp

        self.bus.emit(EVENT_APPEARED, self._build_track_summary(person, timestamp))

        # Check for COMPANION — is this new person near an existing one?
        for other_id, other in self.tracks.items():
            if other_id == track_id:
                continue
            if other.frames_missing > 10:
                continue
            dist = self._distance(person.positions[-1], other.positions[-1])
            if dist < self.companion_distance:
                event_data = self._build_track_summary(person, timestamp)
                event_data["near_track_id"] = other_id
                self.bus.emit(EVENT_COMPANION, event_data)
                break

    def _check_summaries(self, timestamp):
        """Emit periodic track summaries for active tracks."""
        for track_id, person in self.tracks.items():
            if person.frames_missing > 10:
                continue  # skip ghost tracks not currently visible

            last = self._last_summary_time.get(track_id)
            if last is None or (timestamp - last).total_seconds() >= self.summary_interval:
                self.bus.emit(EVENT_TRACK_SUMMARY, self._build_track_summary(person, timestamp))
                self._last_summary_time[track_id] = timestamp

    def _check_departures(self, seen_track_ids, timestamp):
        """Mark people as departed if they haven't been seen for departure_seconds."""
        for track_id in list(self.tracks.keys()):
            if track_id not in seen_track_ids:
                person = self.tracks[track_id]
                person.frames_missing += 1

                seconds_missing = (timestamp - person.last_seen).total_seconds()

                if seconds_missing >= self.departure_seconds:
                    self.bus.emit(EVENT_DEPARTED, self._build_track_summary(person, timestamp))
                    self.departed_tracks[track_id] = self.tracks.pop(track_id)

                    # Clean up summary timer
                    self._last_summary_time.pop(track_id, None)

    def _cleanup_departed(self, timestamp, max_age=60):
        """Remove departed tracks older than max_age seconds."""
        for track_id in list(self.departed_tracks.keys()):
            person = self.departed_tracks[track_id]
            if (timestamp - person.last_seen).total_seconds() > max_age:
                del self.departed_tracks[track_id]

    def _build_track_summary(self, person, timestamp):
        """Build a rich factual summary of a track's current state."""
        # Find companion tracks (other active tracks within distance)
        companions = []
        for other_id, other in self.tracks.items():
            if other_id == person.track_id:
                continue
            if other.frames_missing > 10:
                continue
            if self._distance(person.positions[-1], other.positions[-1]) < self.companion_distance:
                companions.append(other_id)

        zones = self._zone_checker.check_point(person.positions[-1])

        return {
            "track_id": person.track_id,
            "camera_id": self.camera_id,
            "timestamp": timestamp.isoformat(),
            "first_seen": person.first_seen.isoformat(),
            "duration_seconds": round(person.duration_seconds(), 1),
            "bbox": person.current_bbox,
            "avg_movement_30f": round(person.recent_movement(30), 2),
            "avg_movement_150f": round(person.recent_movement(150), 2),
            "total_distance": round(person.total_distance(), 1),
            "position_spread": round(person.position_spread(), 1),
            "nearby_objects": list(person.nearby_objects),
            "is_quiet_hours": self.is_quiet_hours(timestamp),
            "companion_track_ids": companions,
            "frames_tracked": len(person.positions),
            "zones": zones,  # list[dict] — name + rule fields per zone
        }

    def is_quiet_hours(self, timestamp):
        """Check if the given timestamp falls within configured quiet hours."""
        if self.quiet_hours is None:
            return False

        current = timestamp.time()
        start = time.fromisoformat(self.quiet_hours["start"])
        end = time.fromisoformat(self.quiet_hours["end"])

        # Handle overnight ranges (e.g., 22:00 to 06:00)
        if start <= end:
            return start <= current <= end
        else:
            return current >= start or current <= end

    def _distance(self, pos1, pos2):
        """Euclidean distance between two (x, y) points."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
