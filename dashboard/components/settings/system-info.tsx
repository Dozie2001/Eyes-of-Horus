"use client";

import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/lib/api";
import { MagicCard } from "@/components/ui/magic-card";
import { Skeleton } from "@/components/ui/skeleton";

export function SystemInfo() {
  const { data, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return <Skeleton className="h-40 w-full rounded-lg" />;
  }

  if (!data) return null;

  const items = [
    { label: "Status", value: data.status },
    { label: "Site ID", value: data.site_id },
    { label: "Site name", value: data.site_name },
    { label: "Total events", value: data.event_count.toLocaleString() },
    { label: "Cameras", value: Object.keys(data.cameras).length.toString() },
  ];

  return (
    <MagicCard
      className="rounded-lg"
      gradientColor="oklch(0.82 0.12 70 / 3%)"
      gradientFrom="oklch(1 0 0 / 6%)"
      gradientTo="transparent"
    >
      <div className="p-6">
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          System
        </span>

        <div className="mt-5 grid grid-cols-2 gap-4">
          {items.map((item) => (
            <div key={item.label}>
              <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/40">
                {item.label}
              </p>
              <p className="mt-0.5 font-mono text-xs text-foreground/80">
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </MagicCard>
  );
}
