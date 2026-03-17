"use client";

import type { EscalationAlert } from "@/lib/types";
import { mediaUrl, snapshotUrl } from "@/lib/api";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { SeverityBadge } from "@/components/shared/severity-badge";
import { RelativeTime } from "@/components/shared/relative-time";
import { Separator } from "@/components/ui/separator";
import { Camera } from "lucide-react";

interface AlertDetailSheetProps {
  alert: EscalationAlert | null;
  open: boolean;
  onClose: () => void;
}

export function AlertDetailSheet({ alert, open, onClose }: AlertDetailSheetProps) {
  if (!alert) return null;

  const hasVideo = alert.video_path && alert.video_path.length > 0;
  const hasSnapshot = alert.snapshot_path && alert.snapshot_path.length > 0;

  return (
    <Sheet open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-lg">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <SeverityBadge severity={alert.severity} />
            <span className="font-mono text-sm font-light">
              Alert #{alert.id}
            </span>
          </SheetTitle>
          <SheetDescription className="sr-only">
            Alert details for alert {alert.id}
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-col gap-6 px-1 pt-4">
          {/* Video player — primary media */}
          {hasVideo ? (
            <div>
              <p className="mb-2 font-mono text-[9px] uppercase tracking-widest text-muted-foreground/50">
                Video clip
              </p>
              <video
                src={mediaUrl(alert.video_path)}
                controls
                autoPlay
                loop
                className="w-full rounded-lg bg-black"
              />
            </div>
          ) : hasSnapshot ? (
            <div>
              <p className="mb-2 font-mono text-[9px] uppercase tracking-widest text-muted-foreground/50">
                Snapshot
              </p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={snapshotUrl(alert.snapshot_path)}
                alt={`Alert #${alert.id}`}
                className="w-full rounded-lg object-cover"
              />
            </div>
          ) : (
            <div className="flex aspect-video items-center justify-center rounded-lg bg-muted/50">
              <Camera className="size-8 text-muted-foreground/30" />
            </div>
          )}

          {/* If we have both video and snapshot, show snapshot as secondary */}
          {hasVideo && hasSnapshot && (
            <div>
              <p className="mb-2 font-mono text-[9px] uppercase tracking-widest text-muted-foreground/50">
                Snapshot
              </p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={snapshotUrl(alert.snapshot_path)}
                alt={`Alert #${alert.id} snapshot`}
                className="w-full rounded-lg object-cover"
              />
            </div>
          )}

          <Separator className="opacity-30" />

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4">
            <MetaItem label="Event" value={alert.event_type.replace("_", " ")} />
            <MetaItem label="Track" value={`#${alert.track_id}`} />
            <MetaItem label="Camera" value={alert.camera_id} />
            <MetaItem label="Severity" value={alert.severity} />
            <MetaItem label="Status" value={alert.status} />
            <MetaItem label="Time">
              <RelativeTime
                timestamp={alert.created_at}
                className="font-mono text-xs text-foreground/80"
              />
            </MetaItem>
          </div>

          {/* Escalation info */}
          {alert.current_level > 0 && (
            <MetaItem
              label="Escalation"
              value={`Level ${alert.current_level + 1} of ${alert.max_level + 1}`}
            />
          )}

          {alert.acknowledged_by && (
            <>
              <Separator className="opacity-30" />
              <div className="grid grid-cols-2 gap-4">
                <MetaItem
                  label={alert.outcome === "false_alarm" ? "Dismissed by" : "Acknowledged by"}
                  value={alert.acknowledged_by || alert.dismissed_by || "—"}
                />
                {alert.acknowledged_at && (
                  <MetaItem label="At">
                    <RelativeTime
                      timestamp={alert.acknowledged_at}
                      className="font-mono text-xs text-foreground/80"
                    />
                  </MetaItem>
                )}
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
