/**
 * Alert list — tabbed view of pending and recent alerts.
 *
 * Three tabs:
 *   Pending → GET /escalation/pending (polled every 10s)
 *   Recent  → GET /escalation/recent (polled every 10s)
 *   Quality → GET /metrics/alert-quality (polled every 60s)
 *
 * Uses TanStack Query for data fetching:
 *   - useQuery: fetches + caches data, polls at intervals
 *   - useMutation: handles acknowledge/dismiss write operations
 *   - invalidateQueries: refreshes all alert lists after a mutation
 *
 * This means when you acknowledge an alert on this page, the Overview
 * page's pending count also updates automatically (same queryKey).
 */

"use client";

import { useState } from "react";
import { toast } from "sonner";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Bell, Clock, BarChart3 } from "lucide-react";
import { getPendingAlerts, getRecentAlerts, acknowledgeAlert, dismissAlert } from "@/lib/api";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCard } from "./alert-card";
import { AlertQualityCard } from "./alert-quality-card";

export function AlertList() {
  // useQueryClient gives us access to the cache — we use it to
  // invalidate (refresh) queries after mutations
  const queryClient = useQueryClient();

  const pending = useQuery({
    queryKey: ["escalation-pending"],
    queryFn: () => getPendingAlerts(),
    refetchInterval: 10_000,
  });

  const recent = useQuery({
    queryKey: ["escalation-recent"],
    queryFn: () => getRecentAlerts(50),
    refetchInterval: 10_000,
  });

  // useMutation handles write operations. On success, we:
  // 1. Show a toast notification
  // 2. Invalidate all escalation queries — this triggers a refetch
  //    for BOTH the alerts page AND the overview page's pending count
  //    (because they share the "escalation-pending" queryKey)
  const ackMutation = useMutation({
    mutationFn: acknowledgeAlert,
    onSuccess: () => {
      toast.success("Alert acknowledged");
      queryClient.invalidateQueries({ queryKey: ["escalation-pending"] });
      queryClient.invalidateQueries({ queryKey: ["escalation-recent"] });
    },
    onError: () => toast.error("Failed to acknowledge alert"),
  });

  const dismissMutation = useMutation({
    mutationFn: dismissAlert,
    onSuccess: () => {
      toast.success("Alert dismissed as false alarm");
      queryClient.invalidateQueries({ queryKey: ["escalation-pending"] });
      queryClient.invalidateQueries({ queryKey: ["escalation-recent"] });
      setDismissId(null);
    },
    onError: () => {
      toast.error("Failed to dismiss alert");
      setDismissId(null);
    },
  });

  // State for dismiss confirmation dialog
  const [dismissId, setDismissId] = useState<number | null>(null);

  return (
    <>
      <Tabs defaultValue="pending">
        <TabsList>
          <TabsTrigger value="pending">
            <Bell data-icon="inline-start" />
            Pending
            {(pending.data?.length ?? 0) > 0 && (
              <span className="ml-1 rounded-full bg-destructive px-1.5 text-xs text-destructive-foreground">
                {pending.data?.length}
              </span>
            )}
          </TabsTrigger>
          <TabsTrigger value="recent">
            <Clock data-icon="inline-start" />
            Recent
          </TabsTrigger>
          <TabsTrigger value="quality">
            <BarChart3 data-icon="inline-start" />
            Quality
          </TabsTrigger>
        </TabsList>

        {/* Pending alerts tab */}
        <TabsContent value="pending">
          {pending.isLoading ? (
            <div className="flex flex-col gap-4 pt-4">
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-32 w-full" />
            </div>
          ) : (pending.data?.length ?? 0) === 0 ? (
            <div className="flex flex-col items-center gap-2 py-12 text-center">
              <Bell className="size-10 text-muted-foreground" />
              <p className="text-lg font-medium">All clear</p>
              <p className="text-sm text-muted-foreground">
                No pending alerts. The AI is watching.
              </p>
            </div>
          ) : (
            <ScrollArea className="pt-4">
              <div className="flex flex-col gap-3">
                {pending.data?.map((alert) => (
                  <AlertCard
                    key={alert.id}
                    alert={alert}
                    onAcknowledge={(id) => ackMutation.mutate(id)}
                    onDismiss={(id) => setDismissId(id)}
                  />
                ))}
              </div>
            </ScrollArea>
          )}
        </TabsContent>

        {/* Recent alerts tab */}
        <TabsContent value="recent">
          {recent.isLoading ? (
            <div className="flex flex-col gap-4 pt-4">
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-32 w-full" />
            </div>
          ) : (recent.data?.length ?? 0) === 0 ? (
            <div className="flex flex-col items-center gap-2 py-12 text-center">
              <Clock className="size-10 text-muted-foreground" />
              <p className="text-lg font-medium">No history yet</p>
              <p className="text-sm text-muted-foreground">
                Resolved alerts will appear here.
              </p>
            </div>
          ) : (
            <ScrollArea className="pt-4">
              <div className="flex flex-col gap-3">
                {recent.data?.map((alert) => (
                  <AlertCard
                    key={alert.id}
                    alert={alert}
                    readOnly
                  />
                ))}
              </div>
            </ScrollArea>
          )}
        </TabsContent>

        {/* Quality metrics tab */}
        <TabsContent value="quality" className="pt-4">
          <AlertQualityCard />
        </TabsContent>
      </Tabs>

      {/* Dismiss confirmation dialog */}
      <AlertDialog open={dismissId !== null} onOpenChange={(open) => !open && setDismissId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Dismiss as false alarm?</AlertDialogTitle>
            <AlertDialogDescription>
              This marks the alert as a false alarm. This data helps the AI learn
              and reduce future false positives.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => dismissId && dismissMutation.mutate(dismissId)}>
              Dismiss
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
