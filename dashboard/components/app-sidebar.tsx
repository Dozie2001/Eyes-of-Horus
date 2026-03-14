/**
 * Main navigation sidebar for Eyes of Horus.
 *
 * Uses shadcn's Sidebar component which handles:
 * - Desktop: fixed sidebar on the left
 * - Mobile: collapses to a sheet overlay triggered by SidebarTrigger
 *
 * This is a client component because usePathname() needs browser context
 * to know which page we're on (for highlighting the active nav item).
 */

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Bell,
  Activity,
  Camera,
  Settings,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";

/**
 * Navigation items. Each maps to a page route.
 *
 * The icon is a Lucide React component passed as an object —
 * shadcn rule: pass icons as objects, not string keys.
 * The data-icon attribute tells shadcn to handle icon sizing.
 */
const navItems = [
  { title: "Overview", href: "/", icon: LayoutDashboard },
  { title: "Alerts", href: "/alerts", icon: Bell },
  { title: "Events", href: "/events", icon: Activity },
  { title: "Cameras", href: "/cameras", icon: Camera },
  { title: "Settings", href: "/settings", icon: Settings },
];

export function AppSidebar() {
  // usePathname() returns the current URL path (e.g. "/alerts")
  // We compare it to each nav item's href to highlight the active one
  const pathname = usePathname();

  return (
    <Sidebar>
      <SidebarHeader className="p-4">
        <span className="text-lg font-semibold tracking-tight">
          Eyes of Horus
        </span>
        <span className="text-xs text-muted-foreground">
          AI CCTV Monitoring
        </span>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => {
                // "/" needs exact match, others use prefix match
                // so "/alerts/123" still highlights the "Alerts" link
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
                      <span>{item.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4">
        <span className="text-xs text-muted-foreground">v0.1.0</span>
      </SidebarFooter>
    </Sidebar>
  );
}
