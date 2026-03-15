"use client";

import { useQuery } from "@tanstack/react-query";
import { getPipelineStatus } from "@/lib/api";
import { MagicCard } from "@/components/ui/magic-card";
import { NumberTicker } from "@/components/ui/number-ticker";
import { Skeleton } from "@/components/ui/skeleton";

function statusColor(status: string): string {
  switch (status) {
    case "running":  return "bg-green-500";
    case "starting": return "bg-yellow-500";
    case "error":    return "bg-destructive";
    default:         return "bg-muted-foreground/40";
  }
}

export function CameraStatusCards() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: getPipelineStatus,
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border/50 p-6">
          <Skeleton className="mb-4 h-3 w-16" />
          <Skeleton className="h-10 w-full" />
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-lg border border-border/30 p-6">
        <p className="text-sm text-muted-foreground">
          {error?.message ?? "Cannot load cameras"}
        </p>
      </div>
    );
  }

  const cameras = Object.entries(data.cameras);

  if (cameras.length === 0) {
    return (
      <div className="rounded-lg border border-border/30 p-6">
        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          No cameras
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {cameras.map(([camId, cam]) => (
        <MagicCard
          key={camId}
          className="rounded-lg"
          gradientColor="oklch(0.82 0.12 70 / 3%)"
          gradientFrom="oklch(1 0 0 / 8%)"
          gradientTo="transparent"
        >
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`size-1.5 rounded-full ${statusColor(cam.status)}`} />
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  {camId}
                </span>
              </div>
              <span className="font-mono text-[10px] text-muted-foreground/40">
                {cam.status}
              </span>
            </div>

            <div className="mt-6 grid grid-cols-3 gap-4">
              <div>
                <NumberTicker
                  value={cam.fps}
                  decimalPlaces={1}
                  className="text-xl font-light tabular-nums"
                />
                <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/50">
                  fps
                </p>
              </div>
              <div>
                <NumberTicker
                  value={cam.active_tracks}
                  className="text-xl font-light tabular-nums"
                />
                <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/50">
                  people
                </p>
              </div>
              <div>
                <span className="text-xl font-light tabular-nums">
                  {cam.frame_count > 999
                    ? `${(cam.frame_count / 1000).toFixed(0)}k`
                    : cam.frame_count}
                </span>
                <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/50">
                  frames
                </p>
              </div>
            </div>

            {cam.status === "error" && cam.error && (
              <p className="mt-4 font-mono text-[10px] text-destructive/80">
                {cam.error}
              </p>
            )}
          </div>
        </MagicCard>
      ))}
    </div>
  );
}
