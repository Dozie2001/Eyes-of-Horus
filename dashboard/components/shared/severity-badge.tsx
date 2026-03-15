import { cn } from "@/lib/utils";

const severityConfig: Record<string, { bg: string; text: string; label: string }> = {
  high:   { bg: "bg-destructive/15", text: "text-destructive", label: "High" },
  medium: { bg: "bg-warning/15", text: "text-warning", label: "Medium" },
  low:    { bg: "bg-muted", text: "text-muted-foreground", label: "Low" },
  ignore: { bg: "bg-transparent", text: "text-muted-foreground/50", label: "Ignore" },
};

interface SeverityBadgeProps {
  severity: string;
  className?: string;
}

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  const config = severityConfig[severity] ?? severityConfig.ignore;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider",
        config.bg,
        config.text,
        className,
      )}
    >
      {config.label}
    </span>
  );
}
