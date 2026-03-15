"use client";

import { useQuery } from "@tanstack/react-query";
import type { EventRecord } from "@/lib/types";
import { getEventsByTrack } from "@/lib/api";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { EventTypeBadge } from "@/components/shared/event-type-badge";
import { RelativeTime } from "@/components/shared/relative-time";
import { SnapshotImage } from "@/components/shared/snapshot-image";
import { Separator } from "@/components/ui/separator";

interface EventDetailSheetProps {
  event: EventRecord | null;
  open: boolean;
  onClose: () => void;
}

export function EventDetailSheet({ event, open, onClose }: EventDetailSheetProps) {
  const { data: trackHistory } = useQuery({
    queryKey: ["track-history", event?.track_id, event?.camera_id],
    queryFn: () => getEventsByTrack(event!.track_id, 20, event!.camera_id),
    enabled: event !== null,
  });

  if (!event) return null;

  const extra = event.extra as Record<string, unknown>;

  return (
    <Sheet open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-md">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <EventTypeBadge type={event.event_type} className="text-[11px]" />
            <span className="font-mono text-sm font-light">
              Track #{event.track_id}
            </span>
          </SheetTitle>
          <SheetDescription className="sr-only">
            Event details for track {event.track_id}
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-col gap-6 px-1 pt-4">
          {/* Snapshot */}
          <SnapshotImage
            path={null}
            alt={`Event ${event.id}`}
            className="aspect-video w-full rounded-md"
          />

          {/* Metadata grid */}
          <div className="grid grid-cols-2 gap-4">
            <MetaItem label="Camera" value={event.camera_id} />
            <MetaItem label="Time">
              <RelativeTime
                timestamp={event.timestamp}
                className="font-mono text-xs text-foreground/80"
              />
            </MetaItem>
            <MetaItem
              label="Duration"
              value={
                event.duration_seconds > 0
                  ? `${Math.round(event.duration_seconds)}s`
                  : "—"
              }
            />
            <MetaItem
              label="Quiet hours"
              value={event.is_quiet_hours ? "Yes" : "No"}
            />
            <MetaItem
              label="Objects"
              value={
                event.nearby_objects.length > 0
                  ? event.nearby_objects.join(", ")
                  : "None"
              }
            />
          </div>

          {/* Extra data */}
          {Object.keys(extra).length > 0 && (
            <>
              <Separator className="opacity-30" />
              <div>
                <p className="mb-3 font-mono text-[9px] uppercase tracking-widest text-muted-foreground/40">
                  Measurements
                </p>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(extra).map(([key, val]) => (
                    <MetaItem
                      key={key}
                      label={key.replace(/_/g, " ")}
                      value={typeof val === "number" ? String(Math.round(val * 100) / 100) : String(val ?? "—")}
                    />
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Track history */}
          {trackHistory && trackHistory.length > 1 && (
            <>
              <Separator className="opacity-30" />
              <div>
                <p className="mb-3 font-mono text-[9px] uppercase tracking-widest text-muted-foreground/40">
                  Track history
                </p>
                <div className="flex flex-col gap-2">
                  {trackHistory.map((evt) => (
                    <div
                      key={evt.id}
                      className={`flex items-center justify-between rounded px-2 py-1.5 ${
                        evt.id === event.id ? "bg-horus-muted" : ""
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <EventTypeBadge type={evt.event_type} />
                        {evt.duration_seconds > 0 && (
                          <span className="font-mono text-[10px] text-muted-foreground/40">
                            {Math.round(evt.duration_seconds)}s
                          </span>
                        )}
                      </div>
                      <RelativeTime
                        timestamp={evt.timestamp}
                        className="font-mono text-[10px] text-muted-foreground/30"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function MetaItem({
  label,
  value,
  children,
}: {
  label: string;
  value?: string;
  children?: React.ReactNode;
}) {
  return (
    <div>
      <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/40">
        {label}
      </p>
      {children ?? (
        <p className="mt-0.5 font-mono text-xs text-foreground/80">{value}</p>
      )}
    </div>
  );
}
