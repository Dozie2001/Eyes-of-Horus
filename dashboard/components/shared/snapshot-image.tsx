/**
 * Snapshot image — loads event images from the backend.
 *
 * The backend mounts /snapshots/ pointing to data/events/.
 * Event/alert data stores paths like "data/events/webcam/summary_track5.jpg".
 * We strip the prefix and build the full URL.
 *
 * Shows a dark placeholder with a camera icon if no image is available
 * or if the image fails to load.
 */

"use client";

import { useState } from "react";
import { Camera } from "lucide-react";
import { snapshotUrl } from "@/lib/api";

interface SnapshotImageProps {
  path: string | null | undefined;
  alt?: string;
  className?: string;
}

export function SnapshotImage({ path, alt = "Event snapshot", className }: SnapshotImageProps) {
  const [failed, setFailed] = useState(false);

  if (!path || failed) {
    return (
      <div className={`flex items-center justify-center rounded-md bg-muted ${className ?? "aspect-video w-full"}`}>
        <Camera className="size-8 text-muted-foreground" />
      </div>
    );
  }

  return (
    // Using <img> instead of next/image because these are dynamic
    // URLs from the backend, not static assets we can optimize at build time
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={snapshotUrl(path)}
      alt={alt}
      className={`rounded-md object-cover ${className ?? "aspect-video w-full"}`}
      onError={() => setFailed(true)}
    />
  );
}
