"use client";

import { useState, useEffect } from "react";
import { toast } from "sonner";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getQuietHours, setQuietHours } from "@/lib/api";
import { MagicCard } from "@/components/ui/magic-card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export function QuietHoursForm() {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["quiet-hours"],
    queryFn: getQuietHours,
  });

  const [enabled, setEnabled] = useState(false);
  const [start, setStart] = useState("22:00");
  const [end, setEnd] = useState("06:00");

  useEffect(() => {
    if (data) {
      setEnabled(data.active);
      if (data.start) setStart(data.start);
      if (data.end) setEnd(data.end);
    }
  }, [data]);

  const mutation = useMutation({
    mutationFn: () =>
      enabled ? setQuietHours(start, end) : setQuietHours(null, null),
    onSuccess: () => {
      toast.success("Quiet hours updated");
      queryClient.invalidateQueries({ queryKey: ["quiet-hours"] });
    },
    onError: () => toast.error("Failed to update quiet hours"),
  });

  if (isLoading) {
    return <Skeleton className="h-48 w-full rounded-lg" />;
  }

  return (
    <MagicCard
      className="rounded-lg"
      gradientColor="oklch(0.82 0.12 70 / 3%)"
      gradientFrom="oklch(1 0 0 / 6%)"
      gradientTo="transparent"
    >
      <div className="p-6">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Quiet hours
          </span>
          <button
            onClick={() => setEnabled(!enabled)}
            className={`relative h-5 w-9 rounded-full transition-colors ${
              enabled ? "bg-horus/40" : "bg-muted"
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 size-4 rounded-full transition-transform ${
                enabled ? "translate-x-4 bg-horus" : "bg-muted-foreground/40"
              }`}
            />
          </button>
        </div>

        <p className="mt-2 text-sm font-light text-muted-foreground/60">
          During quiet hours, the AI applies stricter scrutiny to all activity.
        </p>

        {enabled && (
          <div className="mt-6 flex items-center gap-3">
            <div>
              <label className="mb-1.5 block font-mono text-[9px] uppercase tracking-wider text-muted-foreground/40">
                Start
              </label>
              <Input
                type="time"
                value={start}
                onChange={(e) => setStart(e.target.value)}
                className="w-28 font-mono text-xs"
              />
            </div>
            <span className="mt-5 text-muted-foreground/20">—</span>
            <div>
              <label className="mb-1.5 block font-mono text-[9px] uppercase tracking-wider text-muted-foreground/40">
                End
              </label>
              <Input
                type="time"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
                className="w-28 font-mono text-xs"
              />
            </div>
          </div>
        )}

        <div className="mt-6">
          <Button
            size="sm"
            onClick={() => mutation.mutate()}
            className="bg-horus/20 text-horus hover:bg-horus/30"
          >
            <span className="font-mono text-[10px] uppercase tracking-wider">
              Save
            </span>
          </Button>
        </div>
      </div>
    </MagicCard>
  );
}
