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

                # Add state info from tracker if available
                tracked = tracker.tracks.get(track_id)
                if tracked:
                    person_data["state"] = tracked.state
                    person_data["duration"] = round(tracked.duration_seconds(), 1)
                    person_data["movement"] = round(tracked.recent_movement(), 1)

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
