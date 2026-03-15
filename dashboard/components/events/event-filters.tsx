"use client";

import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CameraSelect } from "@/components/shared/camera-select";

const EVENT_TYPES = [
  "",
  "appeared",
  "departed",
  "returned",
  "companion",
  "objects_changed",
  "track_summary",
];

interface EventFiltersProps {
  cameraId: string;
  eventType: string;
  onCameraChange: (value: string) => void;
  onEventTypeChange: (value: string) => void;
  onRefresh: () => void;
  isFetching: boolean;
}

export function EventFilters({
  cameraId,
  eventType,
  onCameraChange,
  onEventTypeChange,
  onRefresh,
  isFetching,
}: EventFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <CameraSelect value={cameraId} onChange={onCameraChange} />

      <select
        value={eventType}
        onChange={(e) => onEventTypeChange(e.target.value)}
        className="h-8 rounded-md border border-border bg-background px-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground outline-none transition-colors hover:border-border focus:border-horus/50"
      >
        <option value="">All types</option>
        {EVENT_TYPES.filter(Boolean).map((type) => (
          <option key={type} value={type}>
            {type.replace("_", " ")}
          </option>
        ))}
      </select>

      <Button
        variant="ghost"
        size="sm"
        onClick={onRefresh}
        className="ml-auto h-8 text-muted-foreground hover:text-foreground"
      >
        <RefreshCw
          data-icon="inline-start"
          className={isFetching ? "animate-spin" : ""}
        />
        <span className="font-mono text-[10px] uppercase tracking-wider">
          Refresh
        </span>
      </Button>
    </div>
  );
}
