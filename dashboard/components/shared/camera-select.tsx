"use client";

import { useQuery } from "@tanstack/react-query";
import { getPipelineStatus } from "@/lib/api";

interface CameraSelectProps {
  value: string;
  onChange: (value: string) => void;
}

export function CameraSelect({ value, onChange }: CameraSelectProps) {
  const { data } = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: getPipelineStatus,
    refetchInterval: 30_000,
  });

  const cameras = data ? Object.keys(data.cameras) : [];

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-8 rounded-md border border-border bg-background px-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground outline-none transition-colors hover:border-border focus:border-horus/50"
    >
      <option value="">All cameras</option>
      {cameras.map((cam) => (
        <option key={cam} value={cam}>
          {cam}
        </option>
      ))}
    </select>
  );
}
