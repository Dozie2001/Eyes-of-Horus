"use client";

import { Check, X } from "lucide-react";
import type { EscalationAlert } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { SeverityBadge } from "@/components/shared/severity-badge";
import { RelativeTime } from "@/components/shared/relative-time";
import { SnapshotImage } from "@/components/shared/snapshot-image";

function borderAccent(severity: string): string {
  switch (severity) {
    case "high":   return "border-l-destructive/60";
    case "medium": return "border-l-warning/60";
    default:       return "border-l-border";
  }
}

interface AlertCardProps {
  alert: EscalationAlert;
  onAcknowledge?: (id: number) => void;
  onDismiss?: (id: number) => void;
  onClick?: () => void;
  readOnly?: boolean;
}

export function AlertCard({ alert, onAcknowledge, onDismiss, onClick, readOnly = false }: AlertCardProps) {
  const isPending = alert.status === "pending";

  return (
    <div
      onClick={onClick}
      className={`border-l-2 rounded-lg border border-border cursor-pointer transition-all hover:border-border hover:bg-card/40 ${borderAccent(alert.severity)} ${
        isPending && alert.severity === "high" ? "animate-pulse-subtle" : ""
      }`}
    >
      <div className="flex flex-col gap-4 p-5 md:flex-row">
        {/* Snapshot */}
        <div className="w-full shrink-0 md:w-44">
          <SnapshotImage
            path={alert.snapshot_path || null}
            videoPath={alert.video_path || null}
            alt={`Alert #${alert.id}`}
            className="aspect-video w-full rounded-md"
          />
        </div>

        {/* Content */}
        <div className="flex flex-1 flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <SeverityBadge severity={alert.severity} />
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/60">
              {alert.event_type}
            </span>
            <span className="font-mono text-[10px] text-muted-foreground/40">
              #{alert.track_id}
            </span>
            <span className="font-mono text-[10px] text-muted-foreground/30">
              {alert.camera_id}
            </span>
            <RelativeTime
              timestamp={alert.created_at}
              className="ml-auto font-mono text-[10px] text-muted-foreground/40"
            />
          </div>

          {!isPending && (
            <p className="font-mono text-[10px] text-muted-foreground/50">
              {alert.status === "acknowledged" && `Acknowledged by ${alert.acknowledged_by}`}
              {alert.status === "dismissed" && `Dismissed by ${alert.dismissed_by}`}
            </p>
          )}

          {alert.current_level > 0 && (
            <p className="font-mono text-[10px] text-warning/70">
              Escalated to level {alert.current_level + 1}
            </p>
          )}

          {/* Actions — generous touch targets */}
          {isPending && !readOnly && (
            <div className="mt-auto flex gap-2 pt-1" onClick={(e) => e.stopPropagation()}>
              <Button
                size="sm"
                onClick={() => onAcknowledge?.(alert.id)}
                className="h-8 bg-foreground/10 text-foreground hover:bg-foreground/20"
              >
                <Check data-icon="inline-start" />
                <span className="font-mono text-[10px] uppercase tracking-wider">
                  Acknowledge
                </span>
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onDismiss?.(alert.id)}
                className="h-8 text-muted-foreground hover:text-foreground"
              >
                <X data-icon="inline-start" />
                <span className="font-mono text-[10px] uppercase tracking-wider">
                  False alarm
                </span>
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
