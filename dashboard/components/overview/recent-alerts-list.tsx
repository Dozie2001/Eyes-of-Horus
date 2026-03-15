"use client";

import { useQuery } from "@tanstack/react-query";
import { getAlerts } from "@/lib/api";
import { BlurFade } from "@/components/ui/blur-fade";
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
      <div className="flex flex-col gap-3">
        {[...Array(3)].map((_, i) => (
          <Skeleton key={i} className="h-16 w-full rounded-lg" />
        ))}
      </div>
    );
  }

  const alerts = data ?? [];

  if (alerts.length === 0) {
    return (
      <div className="rounded-lg border border-border/30 py-12 text-center">
        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/40">
          No alerts recorded
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {alerts.map((alert, i) => (
        <BlurFade key={alert.id} delay={i * 0.08} direction="up" offset={4}>
          <div className="group flex items-start gap-4 rounded-lg border border-border/30 p-4 transition-colors hover:border-border/60 hover:bg-card/50">
            {/* Severity indicator — a thin vertical line */}
            <div
              className={`mt-1 h-8 w-0.5 shrink-0 rounded-full ${
                alert.severity === "high"
                  ? "bg-destructive"
                  : alert.severity === "medium"
                    ? "bg-warning"
                    : "bg-muted-foreground/30"
              }`}
            />

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <SeverityBadge severity={alert.severity} />
                <span className="font-mono text-[10px] text-muted-foreground/60">
                  {alert.event_type}
                </span>
                <span className="font-mono text-[10px] text-muted-foreground/40">
                  #{alert.track_id}
                </span>
              </div>
              <p className="mt-1.5 text-sm font-light leading-relaxed text-foreground/70 line-clamp-1">
                {alert.reason}
              </p>
            </div>

            <RelativeTime
              timestamp={alert.created_at}
              className="shrink-0 font-mono text-[10px] text-muted-foreground/40"
            />
          </div>
        </BlurFade>
      ))}
    </div>
  );
}
