"use client";

import { useState } from "react";
import { Camera, Play } from "lucide-react";
import { snapshotUrl, mediaUrl } from "@/lib/api";

interface SnapshotImageProps {
  path: string | null | undefined;
  videoPath?: string | null | undefined;
  alt?: string;
  className?: string;
}

export function SnapshotImage({
  path,
  videoPath,
  alt = "Event snapshot",
  className,
}: SnapshotImageProps) {
  const [imgFailed, setImgFailed] = useState(false);
  const [showVideo, setShowVideo] = useState(false);

  const hasVideo = videoPath && videoPath.length > 0;
  const hasImage = path && path.length > 0 && !imgFailed;

  // Show video player
  if (showVideo && hasVideo) {
    return (
      <video
        src={mediaUrl(videoPath)}
        controls
        autoPlay
        className={`rounded-md bg-black ${className ?? "aspect-video w-full"}`}
      />
    );
  }

  // Show snapshot with play overlay if video exists
  if (hasImage) {
    return (
      <div className={`relative ${className ?? "aspect-video w-full"}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={snapshotUrl(path)}
          alt={alt}
          className="size-full rounded-md object-cover"
          onError={() => setImgFailed(true)}
        />
        {hasVideo && (
          <button
            onClick={() => setShowVideo(true)}
            className="absolute inset-0 flex items-center justify-center rounded-md bg-black/30 opacity-0 transition-opacity hover:opacity-100"
          >
            <div className="flex size-10 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm">
              <Play className="size-4 text-white" />
            </div>
          </button>
        )}
      </div>
    );
  }

  // Placeholder
  return (
    <div
      className={`flex items-center justify-center rounded-md bg-muted/50 ${className ?? "aspect-video w-full"}`}
    >
      <Camera className="size-6 text-muted-foreground/30" />
    </div>
  );
}
