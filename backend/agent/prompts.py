


SYSTEM_PROMPT = """You are a security monitoring AI for a CCTV surveillance system in Nigeria.

Your job: evaluate detection events and decide if they warrant a human alert.

You receive structured data about what the cameras see — including raw movement measurements. You must respond with a JSON object.

## Understanding the movement data

Each tracked person has these measurements:

- **avg_movement_30f** (pixels/frame, ~1 second window): How much the person moved recently.
  - 0–5 = standing still (normal tracking jitter)
  - 5–20 = slow movement (shifting weight, looking around)
  - 20+ = walking or running

- **avg_movement_150f** (pixels/frame, ~5 second window): Longer-term movement trend.
  - Consistently low = standing/sitting in one spot
  - Moderate = pacing, wandering
  - High = purposeful walking or running

- **total_distance** (cumulative pixels traveled): How much the person has moved in total since first seen.
  - Low total + long duration = staying in one spot (could be loitering)
  - High total + long duration = moving around the area (patrol, exploring)

- **position_spread** (max distance from centroid): How far they've roamed from their average position.
  - Low spread = staying in a tight area
  - High spread = covering a wide area

- **duration_seconds**: How long this person has been on camera.

- **frames_tracked**: Total number of frames this person has been detected in.

## What makes something worth alerting:

1. **Quiet hours activity**: Any person detected during configured quiet hours (nighttime) deserves more scrutiny. This is the strongest signal.

2. **Standing in one spot for a long time**: Low movement (avg_movement < 5) with long duration (> 120s) and low position_spread — especially near entry points during quiet hours.

3. **Object interactions**: A person with unusual objects (backpack, bag, box, suitcase) — especially during quiet hours.

4. **Repeated appearances**: A person who departed and returned — possible casing or surveillance of the location.

5. **Companions arriving**: Two or more people meeting at the location, especially during quiet hours.

6. **Unusual movement patterns**: Very high total_distance with low position_spread (pacing), or sudden changes in movement levels.

## Context matters:

- During normal hours, most activity is routine. A person standing around is usually just a person standing around.
- During quiet hours, even routine-looking activity deserves attention.
- Objects (backpacks, bags) near a person during quiet hours are more concerning than during the day.
- Brief appearances during normal hours (< 30 seconds) are almost always routine.

## Severity levels:

- **ignore**: Normal activity, no alert needed.
- **low**: Slightly unusual but probably harmless. Log but don't alert.
- **medium**: Warrants attention. Alert the operator.
- **high**: Urgent. Multiple suspicious signals combined.

## Rules:

- Be factual. Describe WHAT you see, not what you think someone intends.
- Never say "thief", "criminal", "suspicious person". Say "person detected at [time] with [behavior]".
- Keep your reason to 1-2 sentences. Reference specific numbers from the data.
- Keep your recommendation to 1 sentence (what the operator should do).

## Response format (JSON only):

{"alert": true/false, "severity": "ignore|low|medium|high", "reason": "...", "recommendation": "..."}"""


def build_user_prompt(event_type, event_data, scene_summary=None,
                      track_history=None, visual_description=None,
                      cross_camera_context=None, feedback_context=None,
                      camera_profile=None):
    """
    Build the user prompt with context layers.

    Args:
        event_type: e.g. "appeared", "track_summary", "departed"
        event_data: dict from EventTracker._build_track_summary()
        scene_summary: dict from SceneMemory.get_scene_summary() or None
        track_history: list of past events for this track from EventStorage or None
        visual_description: text from VisionDescriber or None
        feedback_context: dict from EvalAgent._get_feedback_context() or None
        camera_profile: dict from CameraProfileStorage.get_profile() or None

    Returns:
        str: formatted user prompt
    """
    parts = []

    # Layer 0: Camera context (sets the scene before event data)
    if camera_profile is not None:
        parts.append("## Camera context")
        parts.append(f"Camera: {camera_profile['camera_id']}")
        if camera_profile.get("description"):
            parts.append(f"Description: {camera_profile['description']}")

        schedule = camera_profile.get("schedule")
        if schedule:
            from datetime import datetime
            now = datetime.now()
            day_of_week = now.weekday()
            is_weekend = day_of_week >= 5
            day_name = now.strftime("%A")

            if is_weekend:
                hours = schedule.get("weekend")
            else:
                hours = schedule.get("weekday")

            if hours and hours.get("start") and hours.get("end"):
                parts.append(f"Normal hours today ({day_name}): {hours['start']} - {hours['end']}")
                # Check if current time is within normal hours
                from datetime import time as dt_time
                start = dt_time.fromisoformat(hours["start"])
                end = dt_time.fromisoformat(hours["end"])
                current = now.time()
                if start <= end:
                    within = start <= current <= end
                else:
                    within = current >= start or current <= end
                if within:
                    parts.append("Current time is within normal hours.")
                else:
                    parts.append("Current time is OUTSIDE normal hours.")
            else:
                day_type = "weekends" if is_weekend else "weekdays"
                parts.append(f"No defined schedule for {day_type}.")
        else:
            parts.append("No defined schedule for this camera.")
        parts.append("")

    # Layer 1: The triggering event
    parts.append("## Event to evaluate")
    parts.append(f"Type: {event_type}")
    parts.append(f"Track ID: #{event_data.get('track_id', '?')}")
    parts.append(f"Timestamp: {event_data.get('timestamp', 'unknown')}")
    parts.append(f"During quiet hours: {'YES' if event_data.get('is_quiet_hours') else 'no'}")

    # Track measurements
    parts.append("")
    parts.append("## Track measurements")
    parts.append(f"Duration on camera: {event_data.get('duration_seconds', 0)} seconds")
    parts.append(f"Frames tracked: {event_data.get('frames_tracked', 0)}")
    parts.append(f"Recent movement (1s): {event_data.get('avg_movement_30f', 0)} px/frame")
    parts.append(f"Recent movement (5s): {event_data.get('avg_movement_150f', 0)} px/frame")
    parts.append(f"Total distance traveled: {event_data.get('total_distance', 0)} px")
    parts.append(f"Position spread (roaming radius): {event_data.get('position_spread', 0)} px")

    objects = event_data.get("nearby_objects", [])
    if objects:
        parts.append(f"Objects near person: {', '.join(objects)}")
    else:
        parts.append("Objects near person: none")

    companions = event_data.get("companion_track_ids", [])
    if companions:
        parts.append(f"Companion tracks nearby: {companions}")

    # Extra fields for specific event types
    if event_type == "companion":
        near_id = event_data.get("near_track_id")
        if near_id is not None:
            parts.append(f"Near existing person: Track #{near_id}")

    if event_type == "objects_changed":
        before = event_data.get("objects_before", [])
        after = event_data.get("objects_after", [])
        parts.append(f"Objects before: {before if before else 'none'}")
        parts.append(f"Objects after: {after if after else 'none'}")

    # Layer 2: Visual description (from vision LLM)
    parts.append("")
    parts.append("## What the camera shows")
    if visual_description:
        parts.append(visual_description)
    else:
        parts.append("No visual description available.")

    # Layer 3: Current scene context
    parts.append("")
    parts.append("## Current scene")
    if scene_summary and scene_summary.get("people_count", 0) > 0:
        parts.append(f"People on camera: {scene_summary['people_count']}")
        for person in scene_summary.get("people", []):
            tid = person.get("track_id", "?")
            dur = person.get("duration", 0)
            mvmt = person.get("movement_30f", 0)
            parts.append(f"  - Track #{tid}: on camera {dur}s, movement={mvmt}")
        scene_objects = scene_summary.get("objects", [])
        if scene_objects:
            parts.append(f"Visible objects: {', '.join(scene_objects)}")
    else:
        parts.append("No scene data available (Redis may be down)")

    # Layer 4: Cross-camera context
    if cross_camera_context:
        parts.append("")
        parts.append("## Other cameras on site")
        for cam in cross_camera_context:
            cam_id = cam.get("camera_id", "?")
            pcount = cam.get("people_count", 0)
            objects = cam.get("objects", [])
            last_update = cam.get("last_update", "")
            if pcount > 0:
                obj_str = f", objects: [{', '.join(objects)}]" if objects else ""
                parts.append(f"- {cam_id}: {pcount} people{obj_str} (updated: {last_update})")
            else:
                parts.append(f"- {cam_id}: empty (updated: {last_update})")

            # Recent departures from this camera
            departures = cam.get("recent_departures", [])
            for dep in departures:
                secs_ago = dep.get("seconds_ago", "?")
                dur = dep.get("duration", 0)
                dep_objects = dep.get("objects", [])
                obj_part = f", had {', '.join(dep_objects)}" if dep_objects else ""
                parts.append(f"  Recent departure: Track #{dep.get('track_id', '?')} left {secs_ago}s ago (was on camera {dur}s{obj_part})")

    # Layer 5: Track history
    parts.append("")
    parts.append("## This person's history")
    if track_history:
        parts.append(f"Previous events for Track #{event_data.get('track_id', '?')}:")
        for evt in track_history[:10]:  # last 10 events max
            evt_type = evt.get("event_type", "?")
            evt_ts = evt.get("timestamp", "?")
            evt_dur = evt.get("duration_seconds", 0)
            evt_quiet = evt.get("is_quiet_hours", False)
            parts.append(f"  - {evt_type} at {evt_ts} (duration: {evt_dur}s, quiet hours: {evt_quiet})")
    else:
        parts.append("No previous events for this person.")

    # Layer 6: Feedback from past alert outcomes
    if feedback_context is not None:
        parts.append("")
        parts.append("## Feedback from previous alerts (learn from this)")
        fp_pct = int(feedback_context["overall_fp_rate"] * 100)
        resolved = feedback_context["resolved_total"]
        parts.append(f"Overall false alarm rate: {fp_pct}% (from {resolved} resolved alerts).")

        et_guidance = feedback_context.get("event_type_guidance")
        if et_guidance:
            parts.append(f"Event-type guidance: {et_guidance}")

        for sev_line in feedback_context.get("severity_guidance", []):
            parts.append(f"Severity guidance: {sev_line}")

        parts.append("Use this feedback to calibrate your judgment. If false alarm rates are high, require stronger evidence before alerting.")

    return "\n".join(parts)
