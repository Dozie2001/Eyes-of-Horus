"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { getPendingAlerts } from "@/lib/api";
import { MagicCard } from "@/components/ui/magic-card";
import { BorderBeam } from "@/components/ui/border-beam";
import { NumberTicker } from "@/components/ui/number-ticker";
import { Button } from "@/components/ui/button";
import { SeverityBadge } from "@/components/shared/severity-badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowRight } from "lucide-react";

export function PendingAlertsCard() {
  const { data, isLoading } = useQuery({
    queryKey: ["escalation-pending"],
    queryFn: () => getPendingAlerts(),
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border p-6">
        <Skeleton className="mb-4 h-3 w-24" />
        <Skeleton className="h-10 w-16" />
      </div>
    );
  }

  const alerts = data ?? [];
  const count = alerts.length;

  const bySeverity: Record<string, number> = {};
  for (const alert of alerts) {
    bySeverity[alert.severity] = (bySeverity[alert.severity] ?? 0) + 1;
  }

  const hasHigh = (bySeverity["high"] ?? 0) > 0;
  const hasPending = count > 0;

  return (
    <MagicCard
      className={`relative rounded-lg ${hasPending ? "animate-breathe" : ""}`}
      gradientColor={hasHigh ? "oklch(0.65 0.20 22 / 8%)" : "oklch(0.82 0.12 70 / 5%)"}
      gradientFrom={hasHigh ? "oklch(0.65 0.20 22 / 40%)" : "oklch(1 0 0 / 8%)"}
      gradientTo="transparent"
    >
      {/* Border beam only when alerts are pending — draws the eye */}
      {hasPending && (
        <BorderBeam
          size={60}
          duration={4}
          colorFrom={hasHigh ? "oklch(0.65 0.20 22)" : "oklch(0.82 0.12 70)"}
          colorTo="transparent"
        />
      )}

      <div className="p-6">
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Pending
        </span>

        <div className="mt-4 flex items-baseline gap-2">
          {count > 0 ? (
            <NumberTicker
              value={count}
              className={`text-4xl font-light tabular-nums ${hasHigh ? "text-destructive" : "text-foreground"}`}
            />
          ) : (
            <span className="text-4xl font-light text-muted-foreground/30">0</span>
          )}
          <span className="font-mono text-[10px] text-muted-foreground/60">
            {count === 1 ? "alert" : "alerts"}
          </span>
        </div>

        {count > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {Object.entries(bySeverity).map(([severity, n]) => (
              <span key={severity} className="flex items-center gap-1.5">
                <SeverityBadge severity={severity} />
                <span className="font-mono text-[10px] text-muted-foreground">{n}</span>
              </span>
            ))}
          </div>
        )}

        {count > 0 && (
          <div className="mt-6">
            <Button
              variant="ghost"
              size="sm"
              render={<Link href="/alerts" />}
              nativeButton={false}
              className="h-auto p-0 text-horus hover:text-horus/80"
            >
              <span className="font-mono text-[10px] uppercase tracking-wider">
                View alerts
              </span>
              <ArrowRight data-icon="inline-end" />
            </Button>
          </div>
        )}

        {count === 0 && (
          <p className="mt-4 text-sm font-light text-muted-foreground/50">
            All clear
          </p>
        )}
      </div>
    </MagicCard>
  );
}
