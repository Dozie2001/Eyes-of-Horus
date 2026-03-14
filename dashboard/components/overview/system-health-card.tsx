/**
 * System health card — shows if the system is up and running.
 *
 * Data source: GET /health (polled every 10s)
 *
 * Displays:
 *   - Site name (from config.yaml)
 *   - Overall status (ok / error)
 *   - Total event count across all cameras
 *   - Green/red dot for quick visual status
 *
 * This is a client component because useQuery needs React context.
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function SystemHealthCard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-32" />
          <Skeleton className="h-4 w-24" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-8 w-20" />
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription className="text-destructive">
            {error?.message ?? "Cannot reach backend"}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const isOk = data.status === "ok";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span
            className={`size-2 rounded-full ${isOk ? "bg-green-500" : "bg-destructive"}`}
          />
          {data.site_name}
        </CardTitle>
        <CardDescription>
          {isOk ? "System operational" : "System error"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{data.event_count.toLocaleString()}</div>
        <p className="text-xs text-muted-foreground">Total events recorded</p>
      </CardContent>
    </Card>
  );
}
