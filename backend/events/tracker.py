"""
Event tracking for StangWatch.
Tracks people across frames using ByteTrack, detects state changes,
and emits events via the event bus.

The EventTracker doesn't know about alerts, databases, or dashboards.
It just emits events. Other components subscribe to what they care about.
"""

from datetime import datetime, time
import math

# --- Event type constants ---
EVENT_APPEARED = "appeared"
EVENT_LOITERING = "loitering"
EVENT_MOVING = "moving"
EVENT_COMPANION = "companion"
EVENT_DEPARTED = "departed"
EVENT_OBJECTS_CHANGED = "objects_changed"
EVENT_RETURNED = "returned"

# --- Internal track states ---
STATE_ACTIVE = "active"
STATE_STATIONARY = "stationary"
STATE_DEPARTED = "departed"


class TrackedPerson:
    """
    Represents one tracked person across frames.
    Stores their history so we can detect state changes.
    """

    def __init__(self, track_id, bbox, timestamp):
        self.track_id = track_id
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.state = STATE_ACTIVE

        # Position history: list of (x, y) center points
        self.positions = [self._center(bbox)]
        self.current_bbox = bbox

        # Objects detected near this person (from YOLO multi-class)
        self.nearby_objects = set()

        # Frames since last detection (for departure detection)
        self.frames_missing = 0

        # Whether we've already emitted a loitering event for this threshold
        self.loiter_alerted = False

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


class EventTracker:
    """
    Tracks people across frames and emits events on state changes.

    Usage:
        from events.bus import event_bus

        tracker = EventTracker(event_bus)

        # Subscribe to events
        event_bus.on("appeared", lambda data: print(f"New person: {data}"))

        # Feed detections every frame
        tracker.update(detections, timestamp)
    """

    def __init__(self, event_bus, loiter_threshold=300, quiet_hours=None,
                 stationary_threshold=5.0, departure_seconds=3.0,
                 companion_distance=200):
        """
        Args:
            event_bus: pyee EventEmitter instance (from events/bus.py)
            loiter_threshold: seconds before flagging as loitering
            quiet_hours: dict {"start": "22:00", "end": "06:00"} or None
            stationary_threshold: pixels/frame below which = standing still
            departure_seconds: seconds without seeing a person before marking departed.
                               Uses actual time, not frame counts — works at any FPS.
            companion_distance: max pixels between two people to trigger COMPANION
        """
        self.bus = event_bus
        self.loiter_threshold = loiter_threshold
        self.quiet_hours = quiet_hours
        self.stationary_threshold = stationary_threshold
        self.departure_seconds = departure_seconds
        self.companion_distance = companion_distance

        # Active tracks: track_id → TrackedPerson
        self.tracks = {}

        # Recently departed (kept briefly for RETURNED detection)
        self.departed_tracks = {}

    def update(self, detections, timestamp):
        """
        Process one frame of detections.
        Emits events via the bus when state changes are detected.

        Args:
            detections: list of dicts with keys: bbox, confidence, label, track_id
                        (track_id comes from ByteTrack via the Detector)
            timestamp: datetime of this frame
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

        # Clean up old departed tracks
        self._cleanup_departed(timestamp)

    def _update_existing(self, track_id, bbox, nearby_objects, timestamp):
        """Update an existing tracked person and check for state changes."""
        person = self.tracks[track_id]
        old_state = person.state
        old_objects = person.nearby_objects.copy()

        person.update(bbox, timestamp)
        movement = person.recent_movement()

        # --- Check LOITERING ---
        if movement < self.stationary_threshold:
            person.state = STATE_STATIONARY

            duration = person.duration_seconds()
            if duration >= self.loiter_threshold and not person.loiter_alerted:
                person.loiter_alerted = True
                self.bus.emit(EVENT_LOITERING, self._event_data(person, timestamp))

        else:
            # --- Check MOVING (was stationary, now moving) ---
            if old_state == STATE_STATIONARY:
                person.state = STATE_ACTIVE
                person.loiter_alerted = False  # reset so it can re-alert if they stop again
                self.bus.emit(EVENT_MOVING, self._event_data(person, timestamp))
            else:
                person.state = STATE_ACTIVE

        # --- Check OBJECTS_CHANGED ---
        if nearby_objects and nearby_objects != old_objects:
            person.nearby_objects = nearby_objects
            event_data = self._event_data(person, timestamp)
            event_data["objects_before"] = list(old_objects)
            event_data["objects_after"] = list(nearby_objects)
            self.bus.emit(EVENT_OBJECTS_CHANGED, event_data)

    def _handle_returned(self, track_id, bbox, timestamp):
        """Handle a person who was departed but came back."""
        person = self.departed_tracks.pop(track_id)
        person.update(bbox, timestamp)
        person.state = STATE_ACTIVE
        person.frames_missing = 0
        self.tracks[track_id] = person
        self.bus.emit(EVENT_RETURNED, self._event_data(person, timestamp))

    def _handle_new(self, track_id, bbox, timestamp):
        """Handle a brand new person."""
        person = TrackedPerson(track_id, bbox, timestamp)
        self.tracks[track_id] = person
        self.bus.emit(EVENT_APPEARED, self._event_data(person, timestamp))

        # Check for COMPANION — is this new person near an existing one?
        # Only consider tracks that are actively being detected right now
        # (frames_missing == 0), not "ghost" tracks that haven't been seen recently
        for other_id, other in self.tracks.items():
            if other_id == track_id:
                continue
            if other.frames_missing > 10:
                continue  # skip ghost tracks — not currently visible
            dist = self._distance(person.positions[-1], other.positions[-1])
            if dist < self.companion_distance:
                event_data = self._event_data(person, timestamp)
                event_data["near_track_id"] = other_id
                self.bus.emit(EVENT_COMPANION, event_data)
                break

    def _check_departures(self, seen_track_ids, timestamp):
        """Mark people as departed if they haven't been seen for departure_seconds."""
        for track_id in list(self.tracks.keys()):
            if track_id not in seen_track_ids:
                person = self.tracks[track_id]
                person.frames_missing += 1

                # Use actual elapsed time, not frame count
                # This works correctly regardless of FPS (10, 15, 30, etc.)
                seconds_missing = (timestamp - person.last_seen).total_seconds()

                if seconds_missing >= self.departure_seconds:
                    person.state = STATE_DEPARTED
                    self.bus.emit(EVENT_DEPARTED, self._event_data(person, timestamp))
                    self.departed_tracks[track_id] = self.tracks.pop(track_id)

    def _cleanup_departed(self, timestamp, max_age=60):
        """Remove departed tracks older than max_age seconds."""
        for track_id in list(self.departed_tracks.keys()):
            person = self.departed_tracks[track_id]
            if (timestamp - person.last_seen).total_seconds() > max_age:
                del self.departed_tracks[track_id]

    def _event_data(self, person, timestamp):
        """Build a structured event dict for emission."""
        return {
            "track_id": person.track_id,
            "timestamp": timestamp.isoformat(),
            "first_seen": person.first_seen.isoformat(),
            "duration_seconds": round(person.duration_seconds(), 1),
            "bbox": person.current_bbox,
            "movement": "stationary" if person.state == STATE_STATIONARY else "moving",
            "is_quiet_hours": self.is_quiet_hours(timestamp),
            "nearby_objects": list(person.nearby_objects),
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
