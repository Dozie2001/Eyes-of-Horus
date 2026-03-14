/**
 * TypeScript interfaces matching the backend API responses.
 *
 * Each interface corresponds to a Python _to_dict() method in the backend.
 * If you change a backend response shape, update the matching type here.
 */

// GET /health
export interface HealthResponse {
  status: string;
  site_id: string;
  site_name: string;
  event_count: number;
  cameras: Record<string, string>; // camera_id -> status string
}

// GET /pipeline/status -> cameras[camera_id]
export interface CameraStatus {
  status: string; // "stopped" | "starting" | "running" | "error"
  fps: number;
  frame_count: number;
  active_tracks: number;
  error: string | null;
}

// GET /pipeline/status
export interface PipelineStatusResponse {
  cameras: Record<string, CameraStatus>;
}

// GET /events -> each item
export interface EventRecord {
  id: number;
  camera_id: string;
  event_type: string;
  track_id: number;
  timestamp: string;
  duration_seconds: number;
  is_quiet_hours: boolean;
  nearby_objects: string[];
  extra: Record<string, unknown>;
  created_at: string;
}

// GET /events/summary
export interface EventSummary {
  counts: Record<string, number>;
  total: number;
}

// GET /agent/decisions -> each item
export interface AgentDecision {
  id: number;
  camera_id: string;
  event_type: string;
  track_id: number;
  alert: boolean;
  severity: string; // "ignore" | "low" | "medium" | "high"
  reason: string;
  recommendation: string;
  eval_duration_ms: number;
  created_at: string;
}

// GET /escalation/pending and /escalation/recent -> each item
export interface EscalationAlert {
  id: number;
  camera_id: string;
  decision_id: number;
  event_type: string;
  track_id: number;
  severity: string;
  current_level: number;
  max_level: number;
  status: string; // "pending" | "acknowledged" | "expired" | "dismissed"
  outcome: string; // "unresolved" | "true_alert" | "false_alarm"
  acknowledged_by: string;
  acknowledged_at: string | null;
  dismissed_by: string;
  snapshot_path: string;
  video_path: string;
  next_escalation_at: string | null;
  created_at: string;
}

// GET /cameras/profiles -> each item
export interface CameraProfile {
  camera_id: string;
  description: string;
  schedule: Record<string, { start: string; end: string }> | null;
  created_at: string;
  updated_at: string;
}

// GET /config/quiet-hours
export interface QuietHours {
  start: string | null;
  end: string | null;
  active: boolean;
}

// GET /metrics/alert-quality
export interface AlertQualityMetrics {
  period_days: number;
  total_alerts: number;
  true_alerts: number;
  false_alarms: number;
  unresolved: number;
  false_positive_rate: number;
  by_event_type: Record<
    string,
    {
      total: number;
      true: number;
      false: number;
      unresolved: number;
      fp_rate: number;
    }
  >;
  by_severity: Record<
    string,
    {
      total: number;
      true: number;
      false: number;
      unresolved: number;
      fp_rate: number;
    }
  >;
}

// GET /roles/members -> each item
export interface RoleMember {
  id: number;
  role: string;
  telegram_chat_id: string;
  telegram_username: string;
  invited_by: string;
  status: string;
  created_at: string;
  revoked_at: string | null;
}
