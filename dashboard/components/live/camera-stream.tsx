"use client";

import { useState } from "react";
import { Camera, AlertCircle } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface CameraStreamProps {
  cameraId: string;
  status: string;
  className?: string;
}

export function CameraStream({ cameraId, status, className }: CameraStreamProps) {
  const [error, setError] = useState(false);

  if (status !== "running") {
    return (
      <div className={`flex flex-col items-center justify-center gap-2 rounded-lg bg-muted/30 ${className ?? "aspect-video w-full"}`}>
        <AlertCircle className="size-6 text-muted-foreground/30" />
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/40">
          {status === "error" ? "Camera error" : "Camera offline"}
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`flex flex-col items-center justify-center gap-2 rounded-lg bg-muted/30 ${className ?? "aspect-video w-full"}`}>
        <Camera className="size-6 text-muted-foreground/30" />
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/40">
          Stream unavailable
        </span>
      </div>
    );
  }

  return (
    // MJPEG stream rendered as an <img> — the browser handles
    // multipart/x-mixed-replace natively, updating the image
    // as each frame arrives. No JavaScript video player needed.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`${API_URL}/cameras/${cameraId}/stream`}
      alt={`Live feed: ${cameraId}`}
      className={`rounded-lg bg-black object-contain ${className ?? "aspect-video w-full"}`}
      onError={() => setError(true)}
    />
  );
}
