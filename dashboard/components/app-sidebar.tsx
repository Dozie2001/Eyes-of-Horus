"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  LayoutDashboard,
  Bell,
  Activity,
  Camera,
  Settings,
  Video,
  LogOut,
} from "lucide-react";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { createClient } from "@/lib/supabase/client";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";

const navItems = [
  { title: "Overview", href: "/", icon: LayoutDashboard },
  { title: "Live View", href: "/live", icon: Video },
  { title: "Alerts", href: "/alerts", icon: Bell },
  { title: "Events", href: "/events", icon: Activity },
  { title: "Cameras", href: "/cameras", icon: Camera },
  { title: "Settings", href: "/settings", icon: Settings },
];

const isCloudMode = process.env.NEXT_PUBLIC_MODE === "cloud";

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [userEmail, setUserEmail] = useState<string | null>(null);

  useEffect(() => {
    if (!isCloudMode) return;
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      setUserEmail(data.user?.email ?? null);
    });
  }, []);

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/login");
    router.refresh();
  }

  return (
    <Sidebar>
      <SidebarHeader className="px-5 pt-6 pb-8">
        <div className="flex flex-col gap-1">
          <span className="font-mono text-[10px] tracking-[0.3em] uppercase text-horus">
            eyes of horus
          </span>
          <span className="text-[10px] text-muted-foreground/60">
            Surveillance Intelligence
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => {
                const isActive =
                  item.href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(item.href);

                return (
                  <SidebarMenuItem key={item.href}>
                    <SidebarMenuButton
                      render={<Link href={item.href} />}
                      isActive={isActive}
                    >
                      <item.icon data-icon="inline-start" />
                      <span className="text-sm">{item.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="px-5 pb-5 space-y-3">
        {isCloudMode && userEmail && (
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] text-muted-foreground truncate max-w-[140px]">
              {userEmail}
            </span>
            <button
              onClick={handleLogout}
              className="text-muted-foreground/50 hover:text-foreground transition-colors"
              title="Sign out"
            >
              <LogOut className="size-3.5" />
            </button>
          </div>
        )}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-1.5 rounded-full bg-green-500/80" />
            <span className="font-mono text-[10px] text-muted-foreground/50">
              v0.1.0
            </span>
          </div>
          <ThemeToggle />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
