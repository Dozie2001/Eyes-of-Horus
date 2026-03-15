"use client";

import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/lib/api";
import { MagicCard } from "@/components/ui/magic-card";
import { NumberTicker } from "@/components/ui/number-ticker";
import { Skeleton } from "@/components/ui/skeleton";

export function SystemHealthCard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border p-6">
        <Skeleton className="mb-4 h-3 w-20" />
        <Skeleton className="h-8 w-32" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-6">
        <p className="font-mono text-[10px] uppercase tracking-widest text-destructive">
          Offline
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          {error?.message ?? "Cannot reach backend"}
        </p>
      </div>
    );
  }

  const isOk = data.status === "ok";

  return (
    <MagicCard
      className="rounded-lg"
      gradientColor="oklch(0.82 0.12 70 / 5%)"
      gradientFrom="oklch(0.82 0.12 70 / 40%)"
      gradientTo="oklch(0.82 0.12 70 / 0%)"
    >
      <div className="p-6">
        <div className="flex items-center gap-2">
          <div className={`size-1.5 rounded-full ${isOk ? "bg-green-500" : "bg-destructive"}`} />
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            System
          </span>
        </div>
        <p className="mt-4 text-sm font-light text-foreground/80">
          {data.site_name}
        </p>
        <div className="mt-6 flex items-baseline gap-1">
          <NumberTicker
            value={data.event_count}
            className="text-3xl font-light tabular-nums tracking-tight"
          />
          <span className="font-mono text-[10px] text-muted-foreground/60">
            events
          </span>
        </div>
      </div>
    </MagicCard>
  );
}
