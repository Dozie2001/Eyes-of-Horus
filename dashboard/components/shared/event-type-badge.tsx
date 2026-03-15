import { cn } from "@/lib/utils";

const typeConfig: Record<string, string> = {
  appeared: "text-green-400/80",
  departed: "text-muted-foreground/60",
  returned: "text-horus/80",
  companion: "text-blue-400/80",
  objects_changed: "text-purple-400/80",
  track_summary: "text-muted-foreground/40",
};

interface EventTypeBadgeProps {
  type: string;
  className?: string;
}

export function EventTypeBadge({ type, className }: EventTypeBadgeProps) {
  const color = typeConfig[type] ?? "text-muted-foreground/50";

  return (
    <span
      className={cn(
        "font-mono text-[9px] uppercase tracking-wider",
        color,
        className,
      )}
    >
      {type.replace("_", " ")}
    </span>
  );
}
