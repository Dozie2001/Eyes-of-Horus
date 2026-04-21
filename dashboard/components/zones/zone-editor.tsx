"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getZones,
  createZone,
  deleteZone,
  segmentZone,
  cameraFrameUrl,
} from "@/lib/api";
import type { Zone } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, RefreshCw, Trash2 } from "lucide-react";

const ZONE_COLORS = [
  "#ef4444",
  "#f59e0b",
  "#10b981",
  "#3b82f6",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
  "#6366f1",
  "#84cc16",
];

function getCentroid(points: number[][]): [number, number] {
  if (points.length === 0) return [0, 0];
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p[0];
    sy += p[1];
  }
  return [sx / points.length, sy / points.length];
}

interface ZoneEditorProps {
  cameraId: string;
}

export function ZoneEditor({ cameraId }: ZoneEditorProps) {
  const queryClient = useQueryClient();
  const imgRef = useRef<HTMLImageElement>(null);
  const [frameSrc, setFrameSrc] = useState("");
  const [frameSize, setFrameSize] = useState({ width: 1920, height: 1080 });
  const [segmenting, setSegmenting] = useState(false);
  const [clickPoint, setClickPoint] = useState<{ x: number; y: number } | undefined>();
  const [pendingPolygons, setPendingPolygons] = useState<number[][][]>([]);
  const [zoneName, setZoneName] = useState("");
  const [zoneSeverity, setZoneSeverity] = useState("");
  const [alertOnEntry, setAlertOnEntry] = useState(false);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const zones = useQuery({
    queryKey: ["zones", cameraId],
    queryFn: () => getZones(cameraId),
  });

  const loadFrame = useCallback(() => {
    setFrameSrc(cameraFrameUrl(cameraId));
    setClickPoint(undefined);
    setPendingPolygons([]);
    setError("");
  }, [cameraId]);

  useEffect(() => {
    loadFrame();
  }, [loadFrame]);

  function handleImageLoad() {
    const img = imgRef.current;
    if (!img) return;
    setFrameSize({ width: img.naturalWidth, height: img.naturalHeight });
  }

  async function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    const img = imgRef.current;
    if (!img || segmenting) return;

    const rect = img.getBoundingClientRect();
    const scaleX = frameSize.width / rect.width;
    const scaleY = frameSize.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    setClickPoint({ x, y });
    setPendingPolygons([]);
    setError("");
    setSegmenting(true);

    // Convert the displayed frame to base64 via canvas
    const canvas = document.createElement("canvas");
    canvas.width = frameSize.width;
    canvas.height = frameSize.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, frameSize.width, frameSize.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    const base64 = dataUrl.split(",")[1];

    try {
      const result = await segmentZone({
        image_base64: base64,
        point: { x, y },
      });

      if (result.polygon_count === 0) {
        setError("SAM3 returned no segments. Try clicking a different area.");
      } else {
        setPendingPolygons(result.polygons);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Segmentation failed");
    } finally {
      setSegmenting(false);
    }
  }

  async function handleSave() {
    if (!zoneName.trim()) {
      setError("Enter a zone name");
      return;
    }
    if (pendingPolygons.length === 0) return;

    setSaving(true);
    setError("");

    const zone: Zone = {
      name: zoneName.trim(),
      points: pendingPolygons[0],
      zone_type: "polygon",
      alert_on_entry: alertOnEntry,
    };
    if (zoneSeverity) {
      zone.severity_override = zoneSeverity;
    }

    try {
      await createZone(cameraId, zone);
      setPendingPolygons([]);
      setClickPoint(undefined);
      setZoneName("");
      setZoneSeverity("");
      setAlertOnEntry(false);
      queryClient.invalidateQueries({ queryKey: ["zones", cameraId] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  function handleCancel() {
    setPendingPolygons([]);
    setClickPoint(undefined);
    setError("");
  }

  async function handleDelete(name: string) {
    if (!confirm(`Delete zone "${name}"?`)) return;
    try {
      await deleteZone(cameraId, name);
      queryClient.invalidateQueries({ queryKey: ["zones", cameraId] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  const savedZones = zones.data ?? [];
  const hasPending = pendingPolygons.length > 0;

  return (
    <div className="flex gap-6">
      {/* Left: frame + overlay */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-3">
          <Button variant="outline" size="sm" onClick={loadFrame}>
            <RefreshCw data-icon="inline-start" className="size-3.5" />
            Refresh frame
          </Button>
        </div>

        <div
          className="relative inline-block cursor-crosshair select-none rounded-lg overflow-hidden bg-black"
          onClick={handleClick}
        >
          <img
            ref={imgRef}
            src={frameSrc}
            alt={`Camera ${cameraId}`}
            className="block max-w-full h-auto"
            onLoad={handleImageLoad}
            crossOrigin="anonymous"
          />

          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${frameSize.width} ${frameSize.height}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Saved zones */}
            {savedZones.map((zone, i) => {
              const color = ZONE_COLORS[i % ZONE_COLORS.length];
              const pts = zone.points.map((p) => `${p[0]},${p[1]}`).join(" ");
              const [cx, cy] = getCentroid(zone.points);
              const fontSize = Math.max(16, frameSize.width / 60);
              return (
                <g key={zone.name}>
                  <polygon
                    points={pts}
                    fill={color}
                    fillOpacity={0.25}
                    stroke={color}
                    strokeWidth={2}
                  />
                  <text
                    x={cx}
                    y={cy}
                    textAnchor="middle"
                    fill="#fff"
                    fontSize={fontSize}
                    fontWeight="bold"
                    paintOrder="stroke"
                    stroke="#000"
                    strokeWidth={3}
                  >
                    {zone.name}
                  </text>
                </g>
              );
            })}

            {/* Pending polygon */}
            {hasPending && (
              <polygon
                points={pendingPolygons[0]
                  .map((p) => `${p[0]},${p[1]}`)
                  .join(" ")}
                fill="rgba(59, 130, 246, 0.3)"
                stroke="#3b82f6"
                strokeWidth={2}
                strokeDasharray="6 3"
              />
            )}

            {/* Click marker */}
            {clickPoint && (
              <circle
                cx={clickPoint.x}
                cy={clickPoint.y}
                r={Math.max(8, frameSize.width / 150)}
                fill="#ef4444"
                stroke="#fff"
                strokeWidth={2}
              />
            )}
          </svg>

          {/* Loading overlay at click point */}
          {segmenting && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/20">
              <div className="flex items-center gap-2 rounded-lg bg-black/70 px-4 py-2 text-sm">
                <Loader2 className="size-4 animate-spin" />
                Segmenting with SAM3...
              </div>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <p className="mt-2 text-sm text-destructive">{error}</p>
        )}

        {/* Instructions or save form */}
        {hasPending ? (
          <div className="mt-3 rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-medium mb-3">Save Zone</h3>
            <div className="flex flex-wrap items-end gap-3">
              <div>
                <Label htmlFor="zone-name" className="text-xs text-muted-foreground">
                  Name
                </Label>
                <Input
                  id="zone-name"
                  value={zoneName}
                  onChange={(e) => setZoneName(e.target.value)}
                  placeholder="e.g. Front Gate"
                  className="mt-1 w-48"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSave();
                    if (e.key === "Escape") handleCancel();
                  }}
                />
              </div>
              <div>
                <Label htmlFor="zone-severity" className="text-xs text-muted-foreground">
                  Severity
                </Label>
                <select
                  id="zone-severity"
                  value={zoneSeverity}
                  onChange={(e) => setZoneSeverity(e.target.value)}
                  className="mt-1 block h-9 rounded-md border border-input bg-transparent px-3 text-sm"
                >
                  <option value="">Inherit</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground pt-5">
                <input
                  type="checkbox"
                  checked={alertOnEntry}
                  onChange={(e) => setAlertOnEntry(e.target.checked)}
                />
                Alert on entry
              </label>
              <div className="flex gap-2 pt-5">
                <Button size="sm" onClick={handleSave} disabled={saving}>
                  {saving ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    "Save"
                  )}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleCancel}
                >
                  Cancel
                </Button>
              </div>
            </div>
          </div>
        ) : (
          !segmenting && (
            <p className="mt-3 text-sm text-muted-foreground">
              Click anywhere on the frame to segment that region with SAM3.
            </p>
          )
        )}
      </div>

      {/* Right: zone list */}
      <div className="w-64 flex-shrink-0">
        <h2 className="text-sm font-medium mb-3">Zones</h2>
        {savedZones.length === 0 ? (
          <p className="text-xs text-muted-foreground/50">No zones defined.</p>
        ) : (
          <div className="space-y-2">
            {savedZones.map((zone, i) => {
              const color = ZONE_COLORS[i % ZONE_COLORS.length];
              return (
                <div
                  key={zone.name}
                  className="rounded-lg border border-border bg-card p-3"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="inline-block size-2.5 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-sm font-medium">{zone.name}</span>
                  </div>
                  {zone.severity_override && (
                    <span className="mt-1 inline-block font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                      {zone.severity_override}
                    </span>
                  )}
                  {zone.alert_on_entry && (
                    <span className="mt-1 ml-2 inline-block font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                      entry alert
                    </span>
                  )}
                  <div className="mt-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs text-destructive/70 hover:text-destructive"
                      onClick={() => handleDelete(zone.name)}
                    >
                      <Trash2 className="size-3" />
                      Delete
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
