/**
 * Client-side providers wrapper.
 *
 * React context providers (like QueryClientProvider) need "use client"
 * because they use React state internally. We put them in this separate
 * component so the root layout can stay as a server component.
 *
 * QueryClient is created ONCE outside the component — this prevents
 * creating a new client on every render (Vercel best practice: rerender-lazy-state-init).
 */

"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { SidebarProvider } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";

export function Providers({ children }: { children: React.ReactNode }) {
  // useState with initializer function ensures we create the QueryClient
  // only once per app load, not on every render
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Don't refetch when the user switches browser tabs —
            // our polling intervals handle freshness
            refetchOnWindowFocus: false,
            // Retry failed requests once (backend might be restarting)
            retry: 1,
            // Keep previous data while refetching (no loading flash)
            staleTime: 5_000,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <SidebarProvider>
          {children}
        </SidebarProvider>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
