/**
 * Alert card — a single escalation alert with visual severity and actions.
 *
 * Design choices for "lively" feel:
 *   - Colored left border based on severity (red=high, amber=medium)
 *   - Snapshot image is prominent (not a tiny thumbnail)
 *   - Hover effect on the card for interactivity feel
 *   - Large touch-friendly action buttons (guards use phones)
 *   - Pulse animation on high-severity pending alerts
 *
 * Props:
 *   alert: EscalationAlert from the backend
 *   onAcknowledge: called when guard presses Acknowledge
 *   onDismiss: called when guard presses False Alarm
 *   readOnly: true for resolved alerts (no action buttons)
 */

"use client";

import { Check, X } from "lucide-react";
import type { EscalationAlert } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SeverityBadge } from "@/components/shared/severity-badge";
import { RelativeTime } from "@/components/shared/relative-time";
import { SnapshotImage } from "@/components/shared/snapshot-image";

/**
 * Map severity to a left-border color class.
 * This gives each card an immediate visual weight — you can scan
 * a list of cards and instantly spot the red (high) ones.
 */
function borderColor(severity: string): string {
  switch (severity) {
    case "high":   return "border-l-destructive";
    case "medium": return "border-l-warning";
    default:       return "border-l-muted-foreground";
  }
}

interface AlertCardProps {
  alert: EscalationAlert;
  onAcknowledge?: (id: number) => void;
  onDismiss?: (id: number) => void;
  readOnly?: boolean;
}

export function AlertCard({ alert, onAcknowledge, onDismiss, readOnly = false }: AlertCardProps) {
  const isPending = alert.status === "pending";

  return (
    <Card
      className={`border-l-4 transition-colors hover:bg-accent/50 ${borderColor(alert.severity)} ${
        isPending && alert.severity === "high" ? "animate-pulse-subtle" : ""
      }`}
    >
      <CardContent className="flex flex-col gap-4 p-4 md:flex-row">
        {/* Snapshot — prominent on mobile, side panel on desktop */}
        <div className="w-full shrink-0 md:w-48">
          <SnapshotImage
            path={alert.snapshot_path || null}
            alt={`Alert #${alert.id} snapshot`}
          />
        </div>

        {/* Alert details */}
        <div className="flex flex-1 flex-col gap-2">
          {/* Header row: severity + event type + time */}
          <div className="flex flex-wrap items-center gap-2">
            <SeverityBadge severity={alert.severity} />
            <span className="text-sm font-medium">
              {alert.event_type.toUpperCase()}
            </span>
            <span className="text-sm text-muted-foreground">
              Track #{alert.track_id}
            </span>
            <span className="text-xs text-muted-foreground">
              {alert.camera_id}
            </span>
            <RelativeTime
              timestamp={alert.created_at}
              className="ml-auto text-xs text-muted-foreground"
            />
          </div>

          {/* Status for resolved alerts */}
          {!isPending && (
            <div className="text-sm">
              {alert.status === "acknowledged" && (
                <span className="text-green-500">
                  Acknowledged by {alert.acknowledged_by}
                </span>
              )}
              {alert.status === "dismissed" && (
                <span className="text-muted-foreground">
                  Dismissed by {alert.dismissed_by}
                </span>
              )}
            </div>
          )}

          {/* Escalation level indicator */}
          {alert.current_level > 0 && (
            <p className="text-xs text-warning">
              Escalated to level {alert.current_level + 1}
            </p>
          )}

          {/* Action buttons — large for phone touch targets */}
          {isPending && !readOnly && (
            <div className="mt-auto flex gap-2 pt-2">
              <Button
                size="sm"
                onClick={() => onAcknowledge?.(alert.id)}
              >
                <Check data-icon="inline-start" />
                Acknowledge
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onDismiss?.(alert.id)}
              >
                <X data-icon="inline-start" />
                False Alarm
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
