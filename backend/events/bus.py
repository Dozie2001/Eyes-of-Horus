"""
Event bus for StangWatch.
Uses pyee to provide a publish/subscribe event system.

Components emit events, other components subscribe and react.
The event bus decouples everything — the tracker doesn't know about alerts,
the alerts don't know about the database, etc.

Usage:
    from events.bus import event_bus

    # Subscribe: "when 'loitering' happens, call this function"
    @event_bus.on("loitering")
    def handle_loitering(event_data):
        print(f"Loitering detected: {event_data}")

    # Emit: "loitering just happened"
    event_bus.emit("loitering", {"track_id": 1, "duration": 300})

Event names (defined in events/tracker.py):
    "appeared"        — new person entered the frame
    "loitering"       — person stayed too long in same spot
    "moving"          — stationary person started moving
    "companion"       — additional person appeared near existing one
    "departed"        — person left the frame
    "objects_changed" — objects near person changed
    "returned"        — person reappeared after brief absence
"""

from pyee.base import EventEmitter

# Single shared event bus for the entire application.
# All components import this same instance.
event_bus = EventEmitter()
