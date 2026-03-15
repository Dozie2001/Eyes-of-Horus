"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPipelineStatus, getCameraProfiles } from "@/lib/api";
import type { CameraProfile } from "@/lib/types";
import { MagicCard } from "@/components/ui/magic-card";
import { BlurFade } from "@/components/ui/blur-fade";
import { NumberTicker } from "@/components/ui/number-ticker";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { CameraProfileDialog } from "./camera-profile-dialog";
import { Pencil, Clock } from "lucide-react";

function statusColor(status: string): string {
  switch (status) {
    case "running":  return "bg-green-500";
    case "starting": return "bg-yellow-500";
    case "error":    return "bg-destructive";
    default:         return "bg-muted-foreground/40";
  }
}

export function CameraList() {
  const pipeline = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: getPipelineStatus,
    refetchInterval: 10_000,
  });

  const profiles = useQuery({
    queryKey: ["camera-profiles"],
    queryFn: getCameraProfiles,
  });

  const [editCamera, setEditCamera] = useState<string | null>(null);

  if (pipeline.isLoading) {
    return (
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Skeleton className="h-48 rounded-lg" />
      </div>
    );
  }

  const cameras = pipeline.data ? Object.entries(pipeline.data.cameras) : [];
  const profileMap = new Map<string, CameraProfile>();
  for (const p of profiles.data ?? []) {
    profileMap.set(p.camera_id, p);
  }

  if (cameras.length === 0) {
    return (
      <div className="rounded-lg border border-border py-16 text-center">
        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/40">
          No cameras configured
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {cameras.map(([camId, cam], i) => {
          const profile = profileMap.get(camId);
          return (
            <BlurFade key={camId} delay={i * 0.1} direction="up" offset={6}>
              <MagicCard
                className="rounded-lg"
                gradientColor="oklch(0.82 0.12 70 / 3%)"
                gradientFrom="oklch(1 0 0 / 6%)"
                gradientTo="transparent"
              >
                <div className="p-6">
                  {/* Header */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`size-1.5 rounded-full ${statusColor(cam.status)}`} />
                      <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                        {camId}
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setEditCamera(camId)}
                      className="h-7 text-muted-foreground/40 hover:text-foreground"
                    >
                      <Pencil data-icon="inline-start" />
                      <span className="font-mono text-[9px] uppercase tracking-wider">
                        {profile ? "Edit" : "Add profile"}
                      </span>
                    </Button>
                  </div>

                  {/* Stats */}
                  <div className="mt-5 grid grid-cols-3 gap-4">
                    <div>
                      <NumberTicker
                        value={cam.fps}
                        decimalPlaces={1}
                        className="text-xl font-light tabular-nums"
                      />
                      <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/40">
                        fps
                      </p>
                    </div>
                    <div>
                      <NumberTicker
                        value={cam.active_tracks}
                        className="text-xl font-light tabular-nums"
                      />
                      <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/40">
                        people
                      </p>
                    </div>
                    <div>
                      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/50">
                        {cam.status}
                      </span>
                    </div>
                  </div>

                  {/* Profile info */}
                  {profile && (
                    <div className="mt-5 border-t border-border/20 pt-4">
                      {profile.description && (
                        <p className="text-sm font-light leading-relaxed text-foreground/60 line-clamp-2">
                          {profile.description}
                        </p>
                      )}
                      {profile.schedule && (
                        <div className="mt-3 flex items-center gap-1.5 text-muted-foreground/40">
                          <Clock className="size-3" />
                          <span className="font-mono text-[9px]">
                            {profile.schedule.weekday
                              ? `${profile.schedule.weekday.start}–${profile.schedule.weekday.end}`
                              : "No weekday schedule"}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {cam.status === "error" && cam.error && (
                    <p className="mt-4 font-mono text-[10px] text-destructive/70">
                      {cam.error}
                    </p>
                  )}
                </div>
              </MagicCard>
            </BlurFade>
          );
        })}
      </div>

      <CameraProfileDialog
        cameraId={editCamera ?? ""}
        profile={editCamera ? (profileMap.get(editCamera) ?? null) : null}
        open={editCamera !== null}
        onClose={() => setEditCamera(null)}
      />
    </>
  );
}
