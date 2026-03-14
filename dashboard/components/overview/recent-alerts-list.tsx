/**
 * Recent alerts list — last 5 AI-flagged alerts.
 *
 * Data source: GET /agent/alerts?limit=5 (polled every 10s)
 *
 * Shows a compact list of the most recent alerts with:
 *   - Severity badge
 *   - Event type + track ID
 *   - AI's reason (truncated)
 *   - Relative time ("2 min ago")
 *
 * This answers: "What has the AI flagged recently?"
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import { getAlerts } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { SeverityBadge } from "@/components/shared/severity-badge";
import { RelativeTime } from "@/components/shared/relative-time";

export function RecentAlertsList() {
  const { data, isLoading } = useQuery({
    queryKey: ["agent-alerts", 5],
    queryFn: () => getAlerts(5),
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-28" />
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </CardContent>
      </Card>
    );
  }

  const alerts = data ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Alerts</CardTitle>
        <CardDescription>Last 5 AI-flagged events</CardDescription>
      </CardHeader>
      <CardContent>
        {alerts.length === 0 ? (
          <p className="text-sm text-muted-foreground">No alerts yet</p>
        ) : (
          <div className="flex flex-col gap-3">
            {alerts.map((alert, i) => (
              <div key={alert.id}>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <SeverityBadge severity={alert.severity} />
                      <span className="text-sm font-medium">
                        {alert.event_type} #{alert.track_id}
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                      {alert.reason}
                    </p>
                  </div>
                  <RelativeTime
                    timestamp={alert.created_at}
                    className="shrink-0 text-xs text-muted-foreground"
                  />
                </div>
                {i < alerts.length - 1 && <Separator className="mt-3" />}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
