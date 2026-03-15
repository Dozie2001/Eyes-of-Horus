"use client";

import { useState, useEffect } from "react";
import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { upsertCameraProfile, deleteCameraProfile } from "@/lib/api";
import type { CameraProfile } from "@/lib/types";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Trash2 } from "lucide-react";

interface CameraProfileDialogProps {
  cameraId: string;
  profile: CameraProfile | null;
  open: boolean;
  onClose: () => void;
}

export function CameraProfileDialog({
  cameraId,
  profile,
  open,
  onClose,
}: CameraProfileDialogProps) {
  const queryClient = useQueryClient();

  const [description, setDescription] = useState("");
  const [weekdayStart, setWeekdayStart] = useState("");
  const [weekdayEnd, setWeekdayEnd] = useState("");
  const [weekendStart, setWeekendStart] = useState("");
  const [weekendEnd, setWeekendEnd] = useState("");

  useEffect(() => {
    if (profile) {
      setDescription(profile.description);
      setWeekdayStart(profile.schedule?.weekday?.start ?? "");
      setWeekdayEnd(profile.schedule?.weekday?.end ?? "");
      setWeekendStart(profile.schedule?.weekend?.start ?? "");
      setWeekendEnd(profile.schedule?.weekend?.end ?? "");
    } else {
      setDescription("");
      setWeekdayStart("");
      setWeekdayEnd("");
      setWeekendStart("");
      setWeekendEnd("");
    }
  }, [profile, open]);

  const saveMutation = useMutation({
    mutationFn: () => {
      const schedule: Record<string, { start: string; end: string }> = {};
      if (weekdayStart && weekdayEnd) {
        schedule.weekday = { start: weekdayStart, end: weekdayEnd };
      }
      if (weekendStart && weekendEnd) {
        schedule.weekend = { start: weekendStart, end: weekendEnd };
      }
      return upsertCameraProfile(cameraId, {
        description,
        schedule: Object.keys(schedule).length > 0 ? schedule : null,
      });
    },
    onSuccess: () => {
      toast.success("Profile saved");
      queryClient.invalidateQueries({ queryKey: ["camera-profiles"] });
      onClose();
    },
    onError: () => toast.error("Failed to save profile"),
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteCameraProfile(cameraId),
    onSuccess: () => {
      toast.success("Profile deleted");
      queryClient.invalidateQueries({ queryKey: ["camera-profiles"] });
      onClose();
    },
    onError: () => toast.error("Failed to delete profile"),
  });

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="font-light">
            {cameraId}
          </DialogTitle>
          <DialogDescription className="font-mono text-[10px] uppercase tracking-widest">
            Camera profile
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-6 py-4">
          {/* Description */}
          <div>
            <label className="mb-2 block font-mono text-[9px] uppercase tracking-wider text-muted-foreground/50">
              Description
            </label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What does this camera see? What is normal activity?"
              className="min-h-[80px] resize-none font-mono text-xs"
            />
          </div>

          {/* Weekday schedule */}
          <div>
            <label className="mb-2 block font-mono text-[9px] uppercase tracking-wider text-muted-foreground/50">
              Weekday hours (Mon-Fri)
            </label>
            <div className="flex items-center gap-2">
              <Input
                type="time"
                value={weekdayStart}
                onChange={(e) => setWeekdayStart(e.target.value)}
                className="font-mono text-xs"
              />
              <span className="text-muted-foreground/30">—</span>
              <Input
                type="time"
                value={weekdayEnd}
                onChange={(e) => setWeekdayEnd(e.target.value)}
                className="font-mono text-xs"
              />
            </div>
          </div>

          {/* Weekend schedule */}
          <div>
            <label className="mb-2 block font-mono text-[9px] uppercase tracking-wider text-muted-foreground/50">
              Weekend hours (Sat-Sun)
            </label>
            <div className="flex items-center gap-2">
              <Input
                type="time"
                value={weekendStart}
                onChange={(e) => setWeekendStart(e.target.value)}
                className="font-mono text-xs"
              />
              <span className="text-muted-foreground/30">—</span>
              <Input
                type="time"
                value={weekendEnd}
                onChange={(e) => setWeekendEnd(e.target.value)}
                className="font-mono text-xs"
              />
            </div>
          </div>
        </div>

        <DialogFooter className="flex items-center justify-between gap-2">
          {profile && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => deleteMutation.mutate()}
              className="mr-auto text-destructive/60 hover:text-destructive"
            >
              <Trash2 data-icon="inline-start" />
              <span className="font-mono text-[10px] uppercase tracking-wider">
                Delete
              </span>
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="text-muted-foreground"
          >
            <span className="font-mono text-[10px] uppercase tracking-wider">
              Cancel
            </span>
          </Button>
          <Button
            size="sm"
            onClick={() => saveMutation.mutate()}
            className="bg-horus/20 text-horus hover:bg-horus/30"
          >
            <span className="font-mono text-[10px] uppercase tracking-wider">
              Save
            </span>
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
