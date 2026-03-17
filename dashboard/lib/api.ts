/**
 * API client for the Eyes of Horus backend.
 *
 * Every backend endpoint gets a typed function here.
 * Components import from this file instead of calling fetch() directly.
 */

import type {
  HealthResponse,
  PipelineStatusResponse,
  EventRecord,
  EventSummary,
  AgentDecision,
  EscalationAlert,
  CameraProfile,
  QuietHours,
  AlertQualityMetrics,
  RoleMember,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Base fetch wrapper. Throws on non-OK responses.
 */
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// --- System ---

export function getHealth() {
  return apiFetch<HealthResponse>("/health");
}

export function getPipelineStatus() {
  return apiFetch<PipelineStatusResponse>("/pipeline/status");
}

// --- Events ---

export function getEvents(limit = 50, cameraId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<EventRecord[]>(`/events?${params}`);
}

export function getEventsSummary(cameraId?: string) {
  const params = new URLSearchParams();
  if (cameraId) params.set("camera_id", cameraId);
  const qs = params.toString();
  return apiFetch<EventSummary>(`/events/summary${qs ? `?${qs}` : ""}`);
}

export function getEventsByType(type: string, limit = 50, cameraId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<EventRecord[]>(`/events/type/${type}?${params}`);
}

export function getEventsByTrack(trackId: number, limit = 50, cameraId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<EventRecord[]>(`/events/track/${trackId}?${params}`);
}

// --- Agent ---

export function getDecisions(limit = 50, cameraId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<AgentDecision[]>(`/agent/decisions?${params}`);
}

export function getAlerts(limit = 50, cameraId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<AgentDecision[]>(`/agent/alerts?${params}`);
}

// --- Escalation ---

export function getPendingAlerts(limit = 50, cameraId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<EscalationAlert[]>(`/escalation/pending?${params}`);
}

export function getRecentAlerts(limit = 50, cameraId?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<EscalationAlert[]>(`/escalation/recent?${params}`);
}

export function acknowledgeAlert(alertId: number) {
  return apiFetch<EscalationAlert>(`/escalation/${alertId}/acknowledge`, {
    method: "PUT",
  });
}

export function dismissAlert(alertId: number) {
  return apiFetch<EscalationAlert>(`/escalation/${alertId}/dismiss`, {
    method: "PUT",
  });
}

// --- Camera Profiles ---

export function getCameraProfiles() {
  return apiFetch<CameraProfile[]>("/cameras/profiles");
}

export function getCameraProfile(cameraId: string) {
  return apiFetch<CameraProfile>(`/cameras/profiles/${cameraId}`);
}

export function upsertCameraProfile(
  cameraId: string,
  data: { description: string; schedule?: Record<string, { start: string; end: string }> | null }
) {
  return apiFetch<CameraProfile>(`/cameras/profiles/${cameraId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function deleteCameraProfile(cameraId: string) {
  return apiFetch<{ deleted: string }>(`/cameras/profiles/${cameraId}`, {
    method: "DELETE",
  });
}

// --- Config ---

export function getQuietHours() {
  return apiFetch<QuietHours>("/config/quiet-hours");
}

export function setQuietHours(start: string | null, end: string | null) {
  return apiFetch<QuietHours>("/config/quiet-hours", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start, end }),
  });
}

// --- Metrics ---

export function getAlertQuality(days = 30, cameraId?: string) {
  const params = new URLSearchParams({ days: String(days) });
  if (cameraId) params.set("camera_id", cameraId);
  return apiFetch<AlertQualityMetrics>(`/metrics/alert-quality?${params}`);
}

// --- Roles ---

export function getRoleMembers() {
  return apiFetch<RoleMember[]>("/roles/members");
}

export function getActiveMembers() {
  return apiFetch<Record<string, RoleMember[]>>("/roles/members/active");
}

// --- Helpers ---

/**
 * Build the full URL for a snapshot image served by the backend.
 *
 * The backend mounts /snapshots/ pointing to data/events/.
 * Snapshot paths in the DB look like: "data/events/webcam/summary_track5.jpg"
 * We strip "data/events/" to get the relative path for the mount.
 */
/**
 * Build the full URL for a media file served by the backend.
 *
 * Handles both formats stored in the DB:
 *   - Relative: "data/events/webcam/file.jpg"
 *   - Absolute: "/Users/.../stang/data/events/webcam/file.jpg"
 *
 * Both get normalized to "/snapshots/webcam/file.jpg".
 */
function normalizeMediaPath(path: string): string {
  // Strip absolute path up to and including data/events/
  const idx = path.indexOf("data/events/");
  if (idx !== -1) {
    return path.slice(idx + "data/events/".length);
  }
  return path;
}

export function snapshotUrl(path: string): string {
  return `${API_URL}/snapshots/${normalizeMediaPath(path)}`;
}

export function mediaUrl(path: string): string {
  return `${API_URL}/snapshots/${normalizeMediaPath(path)}`;
}
