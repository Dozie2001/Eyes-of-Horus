/**
 * Pending alerts card — count of unacknowledged alerts.
 *
 * Data source: GET /escalation/pending (polled every 10s)
 *
 * Shows:
 *   - Total pending count (big number)
 *   - Breakdown by severity (high: 2, medium: 3)
 *   - Visual urgency: card border turns red if high-severity alerts exist
 *
 * Guards look at this to answer: "Do I need to act right now?"
 */

"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { getPendingAlerts } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SeverityBadge } from "@/components/shared/severity-badge";
import { Skeleton } from "@/components/ui/skeleton";

export function PendingAlertsCard() {
  const { data, isLoading } = useQuery({
    queryKey: ["escalation-pending"],
    queryFn: () => getPendingAlerts(),
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-28" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-10 w-16" />
        </CardContent>
      </Card>
    );
  }

  const alerts = data ?? [];
  const count = alerts.length;

  // Count by severity for the breakdown
  const bySeverity: Record<string, number> = {};
  for (const alert of alerts) {
    bySeverity[alert.severity] = (bySeverity[alert.severity] ?? 0) + 1;
  }

  const hasHigh = (bySeverity["high"] ?? 0) > 0;

  return (
    <Card className={hasHigh ? "border-destructive" : undefined}>
      <CardHeader>
        <CardTitle>Pending Alerts</CardTitle>
        <CardDescription>
          {count === 0 ? "All clear" : "Awaiting acknowledgment"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold">{count}</div>
        {count > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(bySeverity).map(([severity, n]) => (
              <span key={severity} className="flex items-center gap-1 text-sm">
                <SeverityBadge severity={severity} />
                <span className="text-muted-foreground">{n}</span>
              </span>
            ))}
          </div>
        )}
      </CardContent>
      {count > 0 && (
        <CardFooter>
          <Button
            variant="outline"
            size="sm"
            render={<Link href="/alerts" />}
            nativeButton={false}
          >
            View alerts
          </Button>
        </CardFooter>
      )}
    </Card>
  );
}
