/**
 * Severity badge — color-coded indicator for alert severity levels.
 *
 * Maps severity strings from the backend to shadcn Badge variants:
 *   high   → destructive (red)  — urgent, needs immediate attention
 *   medium → warning (amber)    — warrants attention
 *   low    → secondary (muted)  — probably harmless
 *   ignore → outline (border)   — normal activity
 *
 * Used on: Overview (recent alerts), Alerts page, Events page.
 */

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const severityConfig: Record<string, { variant: "destructive" | "secondary" | "outline"; className?: string; label: string }> = {
  high:   { variant: "destructive", label: "High" },
  medium: { variant: "secondary", className: "bg-warning/15 text-warning", label: "Medium" },
  low:    { variant: "secondary", label: "Low" },
  ignore: { variant: "outline", label: "Ignore" },
};

interface SeverityBadgeProps {
  severity: string;
  className?: string;
}

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  const config = severityConfig[severity] ?? severityConfig.ignore;

  return (
    <Badge
      variant={config.variant}
      className={cn(config.className, className)}
    >
      {config.label}
    </Badge>
  );
}
