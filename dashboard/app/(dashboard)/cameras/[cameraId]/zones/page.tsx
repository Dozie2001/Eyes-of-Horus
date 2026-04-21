"use client";

import { use } from "react";
import { ZoneEditor } from "@/components/zones/zone-editor";

export default function ZoneEditorPage({
  params,
}: {
  params: Promise<{ cameraId: string }>;
}) {
  const { cameraId } = use(params);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-light tracking-tight">Zone Editor</h1>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground/50">
          {cameraId} — click on the frame to segment regions with SAM3
        </p>
      </div>
      <ZoneEditor cameraId={cameraId} />
    </div>
  );
}
