/**
 * Alerts page — manage escalation alerts.
 *
 * Guards use this as a backup for Telegram to:
 *   - See pending alerts that need acknowledgment
 *   - Review recent resolved alerts
 *   - Check AI accuracy (false positive rate)
 */

import { AlertList } from "@/components/alerts/alert-list";

export default function AlertsPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Alerts</h1>
        <p className="text-muted-foreground">
          Manage escalation alerts and review AI accuracy
        </p>
      </div>
      <AlertList />
    </div>
  );
}
