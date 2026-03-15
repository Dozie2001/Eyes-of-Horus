import { SystemHealthCard } from "@/components/overview/system-health-card";
import { CameraStatusCards } from "@/components/overview/camera-status-card";
import { PendingAlertsCard } from "@/components/overview/pending-alerts-card";
import { RecentAlertsList } from "@/components/overview/recent-alerts-list";

export default function OverviewPage() {
  return (
    <div className="flex flex-col gap-10">
      {/* Page header — quiet, typographic */}
      <div>
        <h1 className="text-xl font-light tracking-tight">Overview</h1>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          System status and recent activity
        </p>
      </div>

      {/* Primary metrics — health + pending side by side */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <SystemHealthCard />
        <PendingAlertsCard />
      </div>

      {/* Cameras */}
      <div>
        <h2 className="mb-4 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          Cameras
        </h2>
        <CameraStatusCards />
      </div>

      {/* Recent alerts */}
      <div>
        <h2 className="mb-4 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          Recent alerts
        </h2>
        <RecentAlertsList />
      </div>
    </div>
  );
}
