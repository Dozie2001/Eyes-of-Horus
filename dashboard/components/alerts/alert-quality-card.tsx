/**
 * Alert quality card — false positive rate and outcome breakdown.
 *
 * Data source: GET /metrics/alert-quality (loaded on mount, no polling)
 *
 * Shows:
 *   - Overall false positive rate as a big percentage
 *   - Total resolved alerts
 *   - Breakdown: true alerts vs false alarms
 *   - Per-severity breakdown
 *
 * This answers: "Is the AI accurate, or is it crying wolf?"
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import { getAlertQuality } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { SeverityBadge } from "@/components/shared/severity-badge";

export function AlertQualityCard() {
  // Refetch every 60s — metrics don't change fast
  const { data, isLoading, error } = useQuery({
    queryKey: ["alert-quality"],
    queryFn: () => getAlertQuality(),
    refetchInterval: 60_000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Alert Quality</CardTitle>
          <CardDescription className="text-muted-foreground">
            Escalation not enabled or no data yet
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (!data) return null;

  const fpRate = Math.round(data.false_positive_rate * 100);
  const resolved = data.true_alerts + data.false_alarms;

  // Color the FP rate: green (<25%), amber (25-50%), red (>50%)
  let fpColor = "text-green-500";
  if (fpRate >= 50) fpColor = "text-destructive";
  else if (fpRate >= 25) fpColor = "text-warning";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Alert Quality</CardTitle>
        <CardDescription>Last {data.period_days} days</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {/* Big FP rate number */}
        <div className="flex items-baseline gap-2">
          <span className={`text-4xl font-bold ${fpColor}`}>{fpRate}%</span>
          <span className="text-sm text-muted-foreground">false alarm rate</span>
        </div>

        {/* Outcome breakdown */}
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-lg font-bold text-green-500">{data.true_alerts}</span>
            <p className="text-xs text-muted-foreground">True alerts</p>
          </div>
          <div>
            <span className="text-lg font-bold text-destructive">{data.false_alarms}</span>
            <p className="text-xs text-muted-foreground">False alarms</p>
          </div>
          <div>
            <span className="text-lg font-bold text-muted-foreground">{data.unresolved}</span>
            <p className="text-xs text-muted-foreground">Unresolved</p>
          </div>
        </div>

        {/* Per-severity breakdown */}
        {Object.keys(data.by_severity).length > 0 && (
          <div className="flex flex-col gap-2">
            <p className="text-xs font-medium text-muted-foreground">By severity</p>
            {Object.entries(data.by_severity).map(([sev, stats]) => {
              const sevFp = Math.round(stats.fp_rate * 100);
              return (
                <div key={sev} className="flex items-center gap-2 text-sm">
                  <SeverityBadge severity={sev} />
                  <span className="text-muted-foreground">
                    {sevFp}% FP ({stats.total} total)
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {resolved === 0 && (
          <p className="text-sm text-muted-foreground">
            No resolved alerts yet. Acknowledge or dismiss alerts to build quality data.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
