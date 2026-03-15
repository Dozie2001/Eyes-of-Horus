import { AlertList } from "@/components/alerts/alert-list";

export default function AlertsPage() {
  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-xl font-light tracking-tight">Alerts</h1>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          Escalation management and AI accuracy
        </p>
      </div>
      <AlertList />
    </div>
  );
}
