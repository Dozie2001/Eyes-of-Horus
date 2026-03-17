"use client";

import { useQuery } from "@tanstack/react-query";
import { getPipelineStatus } from "@/lib/api";
import { CameraStream } from "@/components/live/camera-stream";
import { NumberTicker } from "@/components/ui/number-ticker";
import { Skeleton } from "@/components/ui/skeleton";

export default function LivePage() {
  const { data, isLoading } = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: getPipelineStatus,
    refetchInterval: 5_000,
  });

  const cameras = data ? Object.entries(data.cameras) : [];

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-xl font-light tracking-tight">Live View</h1>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          Real-time camera feeds
        </p>
      </div>

      {isLoading ? (
        <Skeleton className="aspect-video w-full rounded-lg" />
      ) : cameras.length === 0 ? (
        <div className="rounded-lg border border-border py-16 text-center">
          <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/40">
            No cameras configured
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          {cameras.map(([camId, cam]) => (
            <div key={camId}>
              {/* Camera header */}
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div
                    className={`size-1.5 rounded-full ${
                      cam.status === "running" ? "bg-green-500" :
                      cam.status === "error" ? "bg-destructive" :
                      "bg-muted-foreground/40"
                    }`}
                  />
                  <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    {camId}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <NumberTicker
                      value={cam.fps}
                      decimalPlaces={1}
                      className="font-mono text-xs tabular-nums text-muted-foreground"
                    />
                    <span className="font-mono text-[9px] text-muted-foreground/40">fps</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <NumberTicker
                      value={cam.active_tracks}
                      className="font-mono text-xs tabular-nums text-muted-foreground"
                    />
                    <span className="font-mono text-[9px] text-muted-foreground/40">people</span>
                  </div>
                </div>
              </div>

              {/* Stream */}
              <CameraStream
                cameraId={camId}
                status={cam.status}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
