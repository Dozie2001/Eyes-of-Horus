/**
 * Overview page — the guard's main screen.
 *
 * Layout (responsive grid):
 *   Mobile:  single column, all cards stacked
 *   Tablet:  2 columns — health + pending side by side
 *   Desktop: 3 columns for top row
 *
 * Top row: System Health | Camera Status | Pending Alerts
 * Bottom:  Recent Alerts (full width)
 *
 * Each card fetches its own data independently via usePoll.
 * If one card fails (e.g. escalation disabled), the others still work.
 */

import { SystemHealthCard } from "@/components/overview/system-health-card";
import { CameraStatusCards } from "@/components/overview/camera-status-card";
import { PendingAlertsCard } from "@/components/overview/pending-alerts-card";
import { RecentAlertsList } from "@/components/overview/recent-alerts-list";

export default function OverviewPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Overview</h1>
        <p className="text-muted-foreground">
          System status and recent activity
        </p>
      </div>

      {/* Top row: health + pending alerts */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        <SystemHealthCard />
        <PendingAlertsCard />
      </div>

      {/* Camera status cards (renders its own grid internally) */}
      <div>
        <h2 className="mb-3 text-lg font-semibold">Cameras</h2>
        <CameraStatusCards />
      </div>

      {/* Recent alerts (full width) */}
      <RecentAlertsList />
    </div>
  );
}
