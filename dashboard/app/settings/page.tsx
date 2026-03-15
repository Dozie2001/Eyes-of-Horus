import { QuietHoursForm } from "@/components/settings/quiet-hours-form";
import { SystemInfo } from "@/components/settings/system-info";

export default function SettingsPage() {
  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-xl font-light tracking-tight">Settings</h1>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          System configuration
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <QuietHoursForm />
        <SystemInfo />
      </div>
    </div>
  );
}
