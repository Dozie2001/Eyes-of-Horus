"""
Agent short-term memory backed by Redis.

Stores the current scene state (who's on camera, what objects are visible)
in Redis hashes with TTL. The AI agent queries this for real-time context
without needing to subscribe to every event.

Data auto-expires — if the pipeline crashes, stale entries clean themselves.

Usage:
    from agent.memory import SceneMemory
    from redis_client import get_redis

    memory = SceneMemory(get_redis(), camera_id="front_gate")

    # Called every frame in the pipeline loop
    memory.update_scene(detections, tracker)

    # Called by the agent when it needs context
    context = memory.get_scene_summary()
    # → {"people_count": 2, "people": [...], "objects": [...], "stats": {...}}
"""

import json
from datetime import datetime


class SceneMemory:
    """
    Writes current scene state to Redis after each detection frame.
    Provides scene summaries for the AI agent.
    """

    def __init__(self, redis_client, camera_id="cam1", ttl_seconds=30):
        """
        Args:
            redis_client: Redis connection
            camera_id: which camera this memory belongs to
            ttl_seconds: how long entries live without being refreshed
        """
        self.redis = redis_client
        self.camera_id = camera_id
        self.ttl = ttl_seconds

        # Redis key prefixes
        self._people_key = f"stang:scene:{camera_id}:people"
        self._stats_key = f"stang:scene:{camera_id}:stats"
        self._objects_key = f"stang:scene:{camera_id}:objects"
        self._departures_key = f"stang:scene:{camera_id}:departures"

    def update_scene(self, detections, tracker):
        """
        Called after each detection frame. Updates Redis with current scene.

        Args:
            detections: list of detection dicts from detector.track_people()
            tracker: EventTracker instance (has .tracks dict)
        """
        try:
            pipe = self.redis.pipeline()

            # Clear old people data and write fresh
            pipe.delete(self._people_key)

            for det in detections:
                track_id = det.get("track_id")
                if track_id is None:
                    continue

                person_data = {
                    "bbox": json.dumps(det["bbox"]),
                    "confidence": det["confidence"],
                    "objects": json.dumps(det.get("nearby_objects", [])),
                }

                # Add measurement info from tracker if available
                tracked = tracker.tracks.get(track_id)
                if tracked:
                    person_data["duration"] = round(tracked.duration_seconds(), 1)
                    person_data["movement_30f"] = round(tracked.recent_movement(30), 1)
                    person_data["position_spread"] = round(tracked.position_spread(), 1)

                pipe.hset(self._people_key, str(track_id), json.dumps(person_data))

            pipe.expire(self._people_key, self.ttl)

            # Update stats
            pipe.hset(self._stats_key, mapping={
                "people_count": str(len(detections)),
                "active_tracks": str(len(tracker.tracks)),
                "departed_tracks": str(len(tracker.departed_tracks)),
                "last_updated": datetime.now().isoformat(),
            })
            pipe.expire(self._stats_key, self.ttl)

            # Collect all visible objects
            all_objects = []
            for det in detections:
                for obj in det.get("nearby_objects", []):
                    all_objects.append(obj)

            pipe.delete(self._objects_key)
            if all_objects:
                for i, obj in enumerate(all_objects):
                    pipe.hset(self._objects_key, str(i), obj)
                pipe.expire(self._objects_key, self.ttl)

            pipe.execute()

        except Exception as e:
            # Redis down? Don't crash the pipeline
            print(f"Scene memory update failed: {e}")

    def record_departure(self, event_data):
        """
        Record a departure event for cross-camera temporal correlation.

        Called when a person leaves this camera's view. Stored in Redis
        with 60s TTL so other cameras can see "someone just left camera X".

        Args:
            event_data: dict from EventTracker departed event
        """
        try:
            track_id = event_data.get("track_id", 0)
            departure_info = {
                "track_id": track_id,
                "timestamp": event_data.get("timestamp", ""),
                "duration": event_data.get("duration_seconds", 0),
                "objects": event_data.get("nearby_objects", []),
            }
            self.redis.hset(
                self._departures_key,
                str(track_id),
                json.dumps(departure_info),
            )
            self.redis.expire(self._departures_key, 60)
        except Exception as e:
            # Redis down? Don't crash the pipeline
            print(f"Scene memory departure record failed: {e}")

    def get_scene_summary(self):
        """
        Get the current scene state. Called by the AI agent.

        Returns:
            dict with people_count, people details, objects, stats
        """
        try:
            # Get all people
            people_raw = self.redis.hgetall(self._people_key)
            people = []
            for track_id, data_str in people_raw.items():
                person = json.loads(data_str)
                person["track_id"] = int(track_id)
                people.append(person)

            # Get stats
            stats = self.redis.hgetall(self._stats_key)

            # Get objects
            objects_raw = self.redis.hgetall(self._objects_key)
            objects = list(objects_raw.values())

            return {
                "camera_id": self.camera_id,
                "people_count": len(people),
                "people": people,
                "objects": objects,
                "stats": stats,
            }

        except Exception as e:
            print(f"Scene memory read failed: {e}")
            return {
                "camera_id": self.camera_id,
                "people_count": 0,
                "people": [],
                "objects": [],
                "stats": {},
                "error": str(e),
            }

    def get_all_cameras_summary(self):
        """
        Get scene summaries from ALL cameras (for cross-camera context).

        Scans Redis for all camera stats keys, reads each camera's summary,
        and returns a list (excluding this camera's own data).

        Returns:
            list of dicts: [{camera_id, people_count, objects, last_update}, ...]
        """
        try:
            # Find all camera stats keys
            keys = self.redis.keys("stang:scene:*:stats")
            summaries = []

            for key in keys:
                # Extract camera_id from key: stang:scene:{camera_id}:stats
                parts = key.split(":")
                if len(parts) < 4:
                    continue
                cam_id = parts[2]

                # Skip our own camera
                if cam_id == self.camera_id:
                    continue

                stats = self.redis.hgetall(f"stang:scene:{cam_id}:stats")
                if not stats:
                    continue

                # Read objects for this camera
                objects_raw = self.redis.hgetall(f"stang:scene:{cam_id}:objects")
                objects = list(objects_raw.values()) if objects_raw else []

                # Read recent departures for this camera
                departures_raw = self.redis.hgetall(f"stang:scene:{cam_id}:departures")
                recent_departures = []
                if departures_raw:
                    now = datetime.now()
                    for _tid, dep_str in departures_raw.items():
                        try:
                            dep = json.loads(dep_str)
                            dep_ts = dep.get("timestamp", "")
                            if dep_ts:
                                dep_time = datetime.fromisoformat(dep_ts)
                                seconds_ago = int((now - dep_time).total_seconds())
                            else:
                                seconds_ago = -1
                            recent_departures.append({
                                "track_id": dep.get("track_id", 0),
                                "seconds_ago": seconds_ago,
                                "duration": dep.get("duration", 0),
                                "objects": dep.get("objects", []),
                            })
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue

                summaries.append({
                    "camera_id": cam_id,
                    "people_count": int(stats.get("people_count", 0)),
                    "active_tracks": int(stats.get("active_tracks", 0)),
                    "objects": objects,
                    "last_update": stats.get("last_updated", ""),
                    "recent_departures": recent_departures,
                })

            return summaries

        except Exception as e:
            print(f"Cross-camera summary failed: {e}")
            return []
