import { SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <AppSidebar />
      <main className="flex-1 overflow-auto">
        <div className="flex items-center gap-3 border-b border-border/40 px-4 py-3 md:hidden">
          <SidebarTrigger />
          <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
            Eyes of Horus
          </span>
        </div>
        <div className="p-5 md:p-8 lg:p-10">
          {children}
        </div>
      </main>
    </>
  );
}
