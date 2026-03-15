"use client";

import type { EventRecord } from "@/lib/types";
import { EventTypeBadge } from "@/components/shared/event-type-badge";
import { RelativeTime } from "@/components/shared/relative-time";
import { BlurFade } from "@/components/ui/blur-fade";

interface EventTableProps {
  events: EventRecord[];
  onSelect: (event: EventRecord) => void;
}

export function EventTable({ events, onSelect }: EventTableProps) {
  if (events.length === 0) {
    return (
      <div className="rounded-lg border border-border py-16 text-center">
        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/40">
          No events found
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {/* Header — hidden on mobile */}
      <div className="hidden border-b border-border px-4 py-2 md:grid md:grid-cols-12 md:gap-4">
        {["Time", "Camera", "Type", "Track", "Duration", "Objects", ""].map(
          (h) => (
            <span
              key={h}
              className={`font-mono text-[9px] uppercase tracking-wider text-muted-foreground/40 ${
                h === "Time" ? "col-span-2" :
                h === "Camera" ? "col-span-2" :
                h === "Type" ? "col-span-2" :
                h === "Track" ? "col-span-1" :
                h === "Duration" ? "col-span-2" :
                h === "Objects" ? "col-span-2" :
                "col-span-1"
              }`}
            >
              {h}
            </span>
          )
        )}
      </div>

      {/* Rows */}
      {events.map((event, i) => (
        <BlurFade key={event.id} delay={Math.min(i * 0.03, 0.3)} direction="up" offset={3}>
          <button
            onClick={() => onSelect(event)}
            className="group w-full rounded-lg px-4 py-3 text-left transition-colors hover:bg-card/60"
          >
            {/* Desktop row */}
            <div className="hidden md:grid md:grid-cols-12 md:items-center md:gap-4">
              <RelativeTime
                timestamp={event.timestamp}
                className="col-span-2 font-mono text-[10px] text-muted-foreground/50"
              />
              <span className="col-span-2 font-mono text-[10px] text-muted-foreground/60">
                {event.camera_id}
              </span>
              <span className="col-span-2">
                <EventTypeBadge type={event.event_type} />
              </span>
              <span className="col-span-1 font-mono text-[10px] text-muted-foreground/50">
                #{event.track_id}
              </span>
              <span className="col-span-2 font-mono text-[10px] tabular-nums text-muted-foreground/50">
                {event.duration_seconds > 0
                  ? `${Math.round(event.duration_seconds)}s`
                  : "—"}
              </span>
              <span className="col-span-2 font-mono text-[10px] text-muted-foreground/40">
                {event.nearby_objects.length > 0
                  ? event.nearby_objects.join(", ")
                  : "—"}
              </span>
              <span className="col-span-1 text-right font-mono text-[10px] text-muted-foreground/20 opacity-0 transition-opacity group-hover:opacity-100">
                {event.is_quiet_hours ? "QH" : ""}
              </span>
            </div>

            {/* Mobile row */}
            <div className="flex items-center gap-3 md:hidden">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <EventTypeBadge type={event.event_type} />
                  <span className="font-mono text-[10px] text-muted-foreground/40">
                    #{event.track_id}
                  </span>
                  {event.is_quiet_hours && (
                    <span className="font-mono text-[9px] text-warning/50">QH</span>
                  )}
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <span className="font-mono text-[10px] text-muted-foreground/40">
                    {event.camera_id}
                  </span>
                  {event.duration_seconds > 0 && (
                    <span className="font-mono text-[10px] text-muted-foreground/30">
                      {Math.round(event.duration_seconds)}s
                    </span>
                  )}
                </div>
              </div>
              <RelativeTime
                timestamp={event.timestamp}
                className="font-mono text-[10px] text-muted-foreground/30"
              />
            </div>
          </button>
        </BlurFade>
      ))}
    </div>
  );
}
