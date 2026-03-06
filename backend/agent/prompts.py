"""
Prompt builder for StangWatch AI evaluation agent.

Builds system + user prompts that give the LLM three layers of context:
1. The triggering event (type, track_id, duration, movement, objects, quiet hours)
2. Current scene from SceneMemory (who else is on camera)
3. Track history from EventStorage (this person's past events)

The LLM returns a JSON decision:
    {"alert": bool, "severity": "ignore|low|medium|high",
     "reason": "...", "recommendation": "..."}
"""


SYSTEM_PROMPT = """You are a security monitoring AI for a CCTV surveillance system in Nigeria.

Your job: evaluate detection events and decide if they warrant a human alert.

You receive structured data about what the cameras see. You must respond with a JSON object.

## What makes something suspicious:

1. **Quiet hours activity**: Any person detected during configured quiet hours (nighttime) is inherently more suspicious. This is the strongest signal.

2. **Loitering**: A person standing still in one area for an extended time, especially near entry points or valuable areas.

3. **Object interactions**: A person picking up or dropping objects (backpack, bag, suitcase) — especially during quiet hours.

4. **Repeated appearances**: A person who departed and returned — possible casing or surveillance of the location.

5. **Companions arriving**: Two or more people meeting at the location, especially during quiet hours — possible coordination.

## What is NOT suspicious:

- People appearing briefly during normal hours
- Normal foot traffic (appearing then departing quickly)
- A single person during daytime with no unusual behavior

## Severity levels:

- **ignore**: Normal activity, no alert needed
- **low**: Slightly unusual but probably harmless. Log but don't alert.
- **medium**: Warrants attention. Alert the operator.
- **high**: Urgent. Multiple suspicious signals combined (quiet hours + loitering + objects, or repeated returns at night).

## Rules:

- Be factual. Describe WHAT you see, not what you think someone intends.
- Never say "thief", "criminal", "suspicious person". Say "person detected at [time] with [behavior]".
- If quiet hours AND any other signal → at minimum severity "medium".
- If NOT quiet hours and behavior is mundane → severity "ignore" or "low".
- Keep your reason to 1-2 sentences.
- Keep your recommendation to 1 sentence (what the operator should do).

## Response format (JSON only):

{"alert": true/false, "severity": "ignore|low|medium|high", "reason": "...", "recommendation": "..."}"""


def build_user_prompt(event_type, event_data, scene_summary=None,
                      track_history=None):
    """
    Build the user prompt with three layers of context.

    Args:
        event_type: e.g. "appeared", "loitering"
        event_data: dict from EventTracker._event_data()
        scene_summary: dict from SceneMemory.get_scene_summary() or None
        track_history: list of past events for this track from EventStorage or None

    Returns:
        str: formatted user prompt
    """
    parts = []

    # Layer 1: The triggering event
    parts.append("## Event to evaluate")
    parts.append(f"Type: {event_type}")
    parts.append(f"Track ID: #{event_data.get('track_id', '?')}")
    parts.append(f"Timestamp: {event_data.get('timestamp', 'unknown')}")
    parts.append(f"Duration on camera: {event_data.get('duration_seconds', 0)} seconds")
    parts.append(f"Movement: {event_data.get('movement', 'unknown')}")
    parts.append(f"During quiet hours: {'YES' if event_data.get('is_quiet_hours') else 'no'}")

    objects = event_data.get("nearby_objects", [])
    if objects:
        parts.append(f"Objects near person: {', '.join(objects)}")

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

    # Layer 2: Current scene context
    parts.append("")
    parts.append("## Current scene")
    if scene_summary and scene_summary.get("people_count", 0) > 0:
        parts.append(f"People on camera: {scene_summary['people_count']}")
        for person in scene_summary.get("people", []):
            tid = person.get("track_id", "?")
            state = person.get("state", "unknown")
            dur = person.get("duration", 0)
            parts.append(f"  - Track #{tid}: {state}, on camera {dur}s")
        scene_objects = scene_summary.get("objects", [])
        if scene_objects:
            parts.append(f"Visible objects: {', '.join(scene_objects)}")
    else:
        parts.append("No scene data available (Redis may be down)")

    # Layer 3: Track history
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

    return "\n".join(parts)
