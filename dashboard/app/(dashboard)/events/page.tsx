"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getEvents, getEventsByType } from "@/lib/api";
import type { EventRecord } from "@/lib/types";
import { EventFilters } from "@/components/events/event-filters";
import { EventTable } from "@/components/events/event-table";
import { EventDetailSheet } from "@/components/events/event-detail-sheet";

export default function EventsPage() {
  const [cameraId, setCameraId] = useState("");
  const [eventType, setEventType] = useState("");
  const [selected, setSelected] = useState<EventRecord | null>(null);

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["events", cameraId, eventType],
    queryFn: () =>
      eventType
        ? getEventsByType(eventType, 100, cameraId || undefined)
        : getEvents(100, cameraId || undefined),
    refetchInterval: 15_000,
  });

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-xl font-light tracking-tight">Events</h1>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          Detection event log
        </p>
      </div>

      <EventFilters
        cameraId={cameraId}
        eventType={eventType}
        onCameraChange={setCameraId}
        onEventTypeChange={setEventType}
        onRefresh={() => refetch()}
        isFetching={isFetching}
      />

      {isLoading ? (
        <div className="flex flex-col gap-2">
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              className="h-12 animate-pulse rounded-lg bg-card/30"
            />
          ))}
        </div>
      ) : (
        <EventTable
          events={data ?? []}
          onSelect={setSelected}
        />
      )}

      <EventDetailSheet
        event={selected}
        open={selected !== null}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
