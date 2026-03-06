"""
Detection pipeline runner for StangWatch.

Runs the Camera → Detector → EventTracker loop in a background thread
so it can coexist with the FastAPI server in the same process.

Usage:
    runner = PipelineRunner(config)
    runner.start(storage)       # starts detection in background thread
    # ... FastAPI serves requests while detection runs ...
    runner.stop()               # clean shutdown
"""

import threading
import time
from datetime import datetime

from capture.camera import Camera
from detection.detector import Detector
from events.bus import event_bus
from events.tracker import EventTracker
from events.storage import EventStorage
from utils import filter_overlapping, save_snapshot, draw_boxes
from agent.memory import SceneMemory
from redis_client import get_redis
from events.redis_bus import RedisBus


class PipelineRunner:
    """
    Runs the detection loop in a background daemon thread.

    Thread-safe because:
    - SQLite WAL mode handles concurrent reads (FastAPI) + writes (pipeline)
    - The event bus handlers run synchronously in the pipeline thread
    - FastAPI only reads the database, pipeline only writes
    """

    def __init__(self, config):
        self.config = config
        self.running = False
        self._thread = None

        # Public status (read by /pipeline/status endpoint)
        self.status = "stopped"     # stopped | starting | running | error
        self.error = None
        self.frame_count = 0
        self.fps = 0.0
        self.active_tracks = 0

    def start(self, storage):
        """Start the detection loop in a background thread."""
        if self.running:
            return

        self.running = True
        self.status = "starting"

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(storage,),
            daemon=True,
            name="detection-pipeline",
        )
        self._thread.start()

    def stop(self):
        """Signal the loop to stop."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run_loop(self, storage):
        """The detection loop. Runs in its own thread."""
        try:
            # Resolve camera source from config
            camera_configs = [c for c in self.config.cameras if c.enabled]
            if camera_configs:
                cam_cfg = camera_configs[0]
                source = cam_cfg.resolved_source(self.config.secrets)
                tracking = cam_cfg.effective_tracking(self.config.tracking)
            else:
                source = 0
                tracking = self.config.tracking

            if isinstance(source, str) and source.isdigit():
                source = int(source)

            # Initialize components
            camera = Camera(source=source)
            if not camera.connect():
                self.status = "error"
                self.error = f"Could not connect to camera: {source}"
                return

            detector = Detector(
                model_name=self.config.detection.model_name,
                confidence_threshold=self.config.detection.confidence_threshold,
                object_association_distance=self.config.detection.association_distance,
            )
            detector.load()

            # Choose event bus: RedisBus if enabled, else pyee
            if self.config.redis.enabled:
                try:
                    r = get_redis(
                        host=self.config.redis.host,
                        port=self.config.redis.port,
                        db=self.config.redis.db,
                        username=self.config.redis.username,
                        password=self.config.redis.password,
                    )
                    r.ping()
                    bus = RedisBus(r, camera_id=cam_cfg.name if camera_configs else "cam1")
                    scene_memory = SceneMemory(r, camera_id=cam_cfg.name if camera_configs else "cam1")
                    print(f"Using RedisBus (stream: {bus.stream})")
                except Exception as e:
                    print(f"Redis unavailable ({e}), falling back to in-process bus")
                    bus = event_bus
                    scene_memory = None
            else:
                bus = event_bus
                scene_memory = None

            tracker = EventTracker(
                event_bus=bus,
                loiter_threshold=tracking.loiter_threshold,
                quiet_hours=tracking.quiet_hours,
                stationary_threshold=tracking.stationary_threshold,
                departure_seconds=tracking.departure_seconds,
                companion_distance=tracking.companion_distance,
            )

            # Subscribe storage to bus (auto-saves events)
            storage.subscribe(bus)

            # Subscribe snapshot saver
            self._subscribe_snapshot_saver(bus)

            # Subscribe event logger (so events are visible in console)
            self._subscribe_event_logger(bus)

            camera.warm_up()
            self.status = "running"
            print(f"Pipeline running: source={source}, FPS={camera.fps}")

            # FPS tracking
            fps_start = time.time()
            fps_frames = 0

            while self.running:
                frame = camera.read_frame()
                if frame is None:
                    # Video file ended or camera disconnected
                    if isinstance(source, str) and not source.startswith("rtsp"):
                        # Video file ended — stop gracefully
                        break
                    time.sleep(0.01)
                    continue

                timestamp = datetime.now()
                self._current_frame = frame  # available to snapshot handler

                # Detect + track
                detections = detector.track_people(frame)
                detections = filter_overlapping(detections)

                # Feed to event tracker (emits events via bus)
                tracker.update(detections, timestamp)

                # Update scene memory for agent context
                if scene_memory is not None:
                    scene_memory.update_scene(detections, tracker)

                # Update stats
                self.frame_count += 1
                fps_frames += 1
                self.active_tracks = len(tracker.tracks)

                # Update FPS every second
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    self.fps = round(fps_frames / elapsed, 1)
                    fps_frames = 0
                    fps_start = time.time()

            camera.release()
            self.status = "stopped"
            print("Pipeline stopped.")

        except Exception as e:
            self.status = "error"
            self.error = str(e)
            print(f"Pipeline error: {e}")

    def _subscribe_snapshot_saver(self, bus):
        """Save a snapshot image for each event."""
        import os
        import cv2

        # We need access to the current frame from the event handler.
        # Since the handler runs synchronously in the pipeline thread
        # (same thread as the detection loop), we use a mutable container.
        self._current_frame = None

        def _save(event_type, event_data):
            if self._current_frame is None:
                return

            os.makedirs("data/events", exist_ok=True)
            bbox = event_data.get("bbox")
            track_id = event_data.get("track_id", 0)
            ts = datetime.fromisoformat(event_data["timestamp"])
            ts_str = ts.strftime("%Y%m%d_%H%M%S")

            filename = f"evt_{ts_str}_track{track_id}_{event_type}.jpg"
            path = f"data/events/{filename}"

            if bbox:
                dets = [{"bbox": bbox, "confidence": 1.0, "label": f"#{track_id}"}]
                frame = draw_boxes(self._current_frame, dets)
            else:
                frame = self._current_frame.copy()

            cv2.putText(
                frame,
                f"{event_type.upper()} | Track #{track_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(path, frame)

        from events.tracker import (
            EVENT_APPEARED, EVENT_LOITERING, EVENT_MOVING,
            EVENT_COMPANION, EVENT_DEPARTED, EVENT_OBJECTS_CHANGED,
            EVENT_RETURNED,
        )

        all_events = [
            EVENT_APPEARED, EVENT_LOITERING, EVENT_MOVING,
            EVENT_COMPANION, EVENT_DEPARTED, EVENT_OBJECTS_CHANGED,
            EVENT_RETURNED,
        ]

        for et in all_events:
            def make_handler(event_type):
                def handler(event_data):
                    _save(event_type, event_data)
                return handler
            bus.on(et, make_handler(et))

    def _subscribe_event_logger(self, bus):
        """Log events to console so they're visible during development."""
        from events.tracker import (
            EVENT_APPEARED, EVENT_LOITERING, EVENT_MOVING,
            EVENT_COMPANION, EVENT_DEPARTED, EVENT_OBJECTS_CHANGED,
            EVENT_RETURNED,
        )

        all_events = [
            EVENT_APPEARED, EVENT_LOITERING, EVENT_MOVING,
            EVENT_COMPANION, EVENT_DEPARTED, EVENT_OBJECTS_CHANGED,
            EVENT_RETURNED,
        ]

        for et in all_events:
            def make_handler(event_type):
                def handler(event_data):
                    track_id = event_data.get("track_id", "?")
                    ts = event_data.get("timestamp", "")
                    extra = ""
                    if event_data.get("duration"):
                        extra = f" | duration={event_data['duration']}s"
                    if event_data.get("nearby_objects"):
                        extra += f" | objects={event_data['nearby_objects']}"
                    print(f"  EVENT: {event_type.upper()} | Track #{track_id} | {ts}{extra}")
                return handler
            bus.on(et, make_handler(et))
