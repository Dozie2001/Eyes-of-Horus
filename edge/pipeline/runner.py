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

import collections
import logging
import os
import threading
import time
from datetime import datetime

import cv2

logger = logging.getLogger(__name__)

from capture.camera import Camera
from detection.detector import Detector
from events.bus import event_bus
from events.tracker import EventTracker
from events.storage import EventStorage
from utils import filter_overlapping, save_snapshot, draw_boxes
from agent.memory import SceneMemory
from agent.evaluator import EvalAgent
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

    def __init__(self, config, escalation=None, camera_config=None,
                 profile_storage=None):
        self.config = config
        self._escalation = escalation
        self._camera_config = camera_config
        self._profile_storage = profile_storage
        self.camera_id = camera_config.name if camera_config else "cam1"
        self.running = False
        self._thread = None
        self._eval_agent = None
        self._tracker = None

        # Public status (read by /pipeline/status endpoint)
        self.status = "stopped"     # stopped | starting | running | error
        self.error = None
        self.frame_count = 0
        self.fps = 0.0
        self.active_tracks = 0

        # Frame ring buffer for video clips (initialized after camera connects)
        self._frame_buffer = None
        self._source_fps = 15
        self._frame_size = (0, 0)

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
            name=f"pipeline-{self.camera_id}",
        )
        self._thread.start()

    def stop(self):
        """Signal the loop to stop."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def get_current_frame(self):
        """Get the latest frame from the pipeline (thread-safe read).

        Returns the raw frame as JPEG bytes, or None if no frame available.
        Used by the MJPEG streaming endpoint.
        """
        frame = getattr(self, '_current_frame', None)
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def _run_loop(self, storage):
        """The detection loop. Runs in its own thread."""
        try:
            # Resolve camera source from config
            cam_cfg = self._camera_config
            if cam_cfg is None:
                # Backward compat: pick first enabled camera
                camera_configs = [c for c in self.config.cameras if c.enabled]
                if camera_configs:
                    cam_cfg = camera_configs[0]
                    self.camera_id = cam_cfg.name

            if cam_cfg:
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
                self.error = f"[{self.camera_id}] Could not connect to camera: {source}"
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
                    bus = RedisBus(r, camera_id=self.camera_id)
                    scene_memory = SceneMemory(r, camera_id=self.camera_id)
                    logger.info(f"[{self.camera_id}] Using RedisBus (stream: {bus.stream})")
                except Exception as e:
                    logger.warning(f"[{self.camera_id}] Redis unavailable ({e}), falling back to in-process bus")
                    bus = event_bus
                    scene_memory = None
            else:
                bus = event_bus
                scene_memory = None

            # Load zones from camera profile
            camera_zones = []
            if self._profile_storage:
                profile = self._profile_storage.get_profile(self.camera_id)
                if profile and profile.get("zones"):
                    camera_zones = profile["zones"]

            self._tracker = EventTracker(
                event_bus=bus,
                quiet_hours=tracking.quiet_hours,
                departure_seconds=tracking.departure_seconds,
                companion_distance=tracking.companion_distance,
                summary_interval=tracking.summary_interval,
                camera_id=self.camera_id,
                zones=camera_zones,
            )

            # Subscribe storage to bus (auto-saves events)
            storage.subscribe(bus)

            # Subscribe snapshot saver
            self._subscribe_snapshot_saver(bus)

            # Subscribe event logger (so events are visible in console)
            self._subscribe_event_logger(bus)

            # Start AI evaluation agent (if enabled and Ollama is reachable)
            # Subscribe SceneMemory to departure events (for cross-camera correlation)
            if scene_memory is not None:
                from events.tracker import EVENT_DEPARTED
                bus.on(EVENT_DEPARTED, lambda data: scene_memory.record_departure(data))

            # Initialize frame ring buffer (15 seconds of frames)
            # Must be before EvalAgent so the agent can snapshot it for alert clips
            self._source_fps = max(camera.fps, 10) or 15
            self._frame_size = (camera.width, camera.height)
            buffer_frames = int(15 * self._source_fps)
            self._frame_buffer = collections.deque(maxlen=buffer_frames)

            # Register scene memory with escalation manager for /ask, /scene, /compare
            if scene_memory is not None and self._escalation is not None:
                self._escalation.register_scene_memory(self.camera_id, scene_memory)

            self._eval_agent = None
            if self.config.agent.enabled:
                redis_client = r if self.config.redis.enabled and scene_memory else None
                agent = EvalAgent(
                    config=self.config,
                    storage=storage,
                    scene_memory=scene_memory,
                    redis_client=redis_client,
                    escalation=self._escalation,
                    camera_id=self.camera_id,
                    profile_storage=self._profile_storage,
                    frame_buffer=self._frame_buffer,
                    source_fps=self._source_fps,
                )
                agent.subscribe(bus)
                agent.start()
                self._eval_agent = agent

            camera.warm_up()
            self.status = "running"
            logger.info(f"[{self.camera_id}] Pipeline running: source={source}, FPS={camera.fps}")

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
                self._frame_buffer.append(frame.copy())  # ring buffer for clips

                # Detect + track
                detections = detector.track_people(frame)
                detections = filter_overlapping(detections)

                # Feed to event tracker (emits events via bus)
                self._tracker.update(detections, timestamp)

                # Update scene memory for agent context
                if scene_memory is not None:
                    scene_memory.update_scene(detections, self._tracker)

                # Update stats
                self.frame_count += 1
                fps_frames += 1
                self.active_tracks = len(self._tracker.tracks)

                # Update FPS every second
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    self.fps = round(fps_frames / elapsed, 1)
                    fps_frames = 0
                    fps_start = time.time()

            camera.release()
            if self._eval_agent is not None:
                self._eval_agent.stop()
            self.status = "stopped"
            logger.info(f"[{self.camera_id}] Pipeline stopped.")

        except Exception as e:
            self.status = "error"
            self.error = str(e)
            logger.error(f"[{self.camera_id}] Pipeline error: {e}")

    def _subscribe_snapshot_saver(self, bus):
        """Save a snapshot image + video clip for each event."""
        import cv2

        self._current_frame = None

        def _save(event_type, event_data):
            if self._current_frame is None:
                return

            cam_id = event_data.get("camera_id", self.camera_id)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            events_dir = os.path.join(project_root, "data", "events", cam_id)
            os.makedirs(events_dir, exist_ok=True)
            bbox = event_data.get("bbox")
            track_id = event_data.get("track_id", 0)
            ts = datetime.fromisoformat(event_data["timestamp"])
            ts_str = ts.strftime("%Y%m%d_%H%M%S")

            # Track summaries overwrite per-track (one snapshot at a time)
            # Lifecycle events get unique filenames
            if event_type == "track_summary":
                snap_filename = f"summary_track{track_id}.jpg"
            else:
                snap_filename = f"evt_{ts_str}_track{track_id}_{event_type}.jpg"

            snap_path = f"{events_dir}/{snap_filename}"

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
            cv2.imwrite(snap_path, frame)

            # Save video clip only for lifecycle events (not every summary)
            if event_type != "track_summary":
                if self._frame_buffer and len(self._frame_buffer) > 10:
                    clip_filename = f"evt_{ts_str}_track{track_id}_{event_type}.mp4"
                    clip_path = f"{events_dir}/{clip_filename}"
                    frames = list(self._frame_buffer)
                    threading.Thread(
                        target=self._write_clip,
                        args=(clip_path, frames),
                        daemon=True,
                    ).start()

        from events.tracker import (
            EVENT_APPEARED, EVENT_DEPARTED, EVENT_RETURNED,
            EVENT_COMPANION, EVENT_OBJECTS_CHANGED,
            EVENT_TRACK_SUMMARY,
        )

        all_events = [
            EVENT_APPEARED, EVENT_DEPARTED, EVENT_RETURNED,
            EVENT_COMPANION, EVENT_OBJECTS_CHANGED,
            EVENT_TRACK_SUMMARY,
        ]

        for et in all_events:
            def make_handler(event_type):
                def handler(event_data):
                    _save(event_type, event_data)
                return handler
            bus.on(et, make_handler(et))

    def _write_clip(self, path, frames):
        """Write buffered frames to an H.264 MP4 via ffmpeg. Runs in background thread."""
        from utils import write_clip_ffmpeg
        write_clip_ffmpeg(path, frames, source_fps=self._source_fps)

    def _subscribe_event_logger(self, bus):
        """Log events to console so they're visible during development."""
        from events.tracker import (
            EVENT_APPEARED, EVENT_DEPARTED, EVENT_RETURNED,
            EVENT_COMPANION, EVENT_OBJECTS_CHANGED,
            EVENT_TRACK_SUMMARY,
        )

        all_events = [
            EVENT_APPEARED, EVENT_DEPARTED, EVENT_RETURNED,
            EVENT_COMPANION, EVENT_OBJECTS_CHANGED,
            EVENT_TRACK_SUMMARY,
        ]

        for et in all_events:
            def make_handler(event_type):
                def handler(event_data):
                    track_id = event_data.get("track_id", "?")
                    ts = event_data.get("timestamp", "")
                    dur = event_data.get("duration_seconds", 0)
                    mvmt = event_data.get("avg_movement_30f", 0)
                    spread = event_data.get("position_spread", 0)
                    objects = event_data.get("nearby_objects", [])
                    extra = f" | dur={dur}s | mvmt={mvmt} | spread={spread}"
                    if objects:
                        extra += f" | objects={objects}"
                    cam = event_data.get("camera_id", "?")
                    logger.debug(f"  EVENT: [{cam}] {event_type.upper()} | Track #{track_id} | {ts}{extra}")
                return handler
            bus.on(et, make_handler(et))
