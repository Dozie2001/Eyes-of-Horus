/**
 * Camera status cards — one card per camera showing live stats.
 *
 * Data source: GET /pipeline/status (polled every 10s)
 *
 * Each camera card shows:
 *   - Camera name
 *   - Status dot (green=running, yellow=starting, red=error/stopped)
 *   - FPS (frames per second the detector processes)
 *   - Active tracks (how many people are currently on camera)
 *   - Error message (if any)
 *
 * This renders a grid of cards — one per camera in the pipeline.
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import { getPipelineStatus } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Map camera status to a colored dot.
 *   running  → green  (everything is working)
 *   starting → yellow (still initializing)
 *   stopped  → gray   (not running)
 *   error    → red    (something broke)
 */
function statusColor(status: string): string {
  switch (status) {
    case "running":  return "bg-green-500";
    case "starting": return "bg-yellow-500";
    case "error":    return "bg-destructive";
    default:         return "bg-muted-foreground";
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
        <Card>
          <CardHeader>
            <Skeleton className="h-5 w-24" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-12 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Cameras</CardTitle>
          <CardDescription className="text-destructive">
            {error?.message ?? "Cannot load camera status"}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const cameras = Object.entries(data.cameras);

  if (cameras.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Cameras</CardTitle>
          <CardDescription>No cameras configured</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {cameras.map(([camId, cam]) => (
        <Card key={camId}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <span className={`size-2 rounded-full ${statusColor(cam.status)}`} />
              {camId}
            </CardTitle>
            <CardDescription>
              {cam.status === "error" ? cam.error : cam.status}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-6 text-sm">
              <div>
                <span className="text-lg font-bold">{cam.fps}</span>
                <p className="text-xs text-muted-foreground">FPS</p>
              </div>
              <div>
                <span className="text-lg font-bold">{cam.active_tracks}</span>
                <p className="text-xs text-muted-foreground">People</p>
              </div>
              <div>
                <span className="text-lg font-bold">{cam.frame_count.toLocaleString()}</span>
                <p className="text-xs text-muted-foreground">Frames</p>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
