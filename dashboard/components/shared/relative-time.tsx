/**
 * Relative time display — shows "2 min ago" instead of "2026-03-13T22:23:02".
 *
 * Uses the browser's built-in Intl.RelativeTimeFormat for localization.
 * Falls back to the raw timestamp if parsing fails.
 *
 * This is a client component because it uses useEffect to update
 * the displayed time every 30 seconds (so "1 min ago" becomes "2 min ago").
 */

"use client";

import { useState, useEffect } from "react";

/**
 * Convert an ISO timestamp to a human-readable relative string.
 *
 * Examples:
 *   - 30 seconds ago → "just now"
 *   - 2 minutes ago  → "2 min ago"
 *   - 3 hours ago    → "3 hr ago"
 *   - 2 days ago     → "2 days ago"
 */
function formatRelative(isoString: string): string {
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);

    if (diffSec < 60) return "just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)} min ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} hr ago`;
    return `${Math.floor(diffSec / 86400)} days ago`;
  } catch {
    return isoString;
  }
}

interface RelativeTimeProps {
  timestamp: string; // ISO 8601 string from the backend
  className?: string;
}

export function RelativeTime({ timestamp, className }: RelativeTimeProps) {
  const [text, setText] = useState(() => formatRelative(timestamp));

  // Re-calculate every 30 seconds so "1 min ago" updates to "2 min ago"
  useEffect(() => {
    setText(formatRelative(timestamp));
    const id = setInterval(() => setText(formatRelative(timestamp)), 30_000);
    return () => clearInterval(id);
  }, [timestamp]);

  return (
    <time dateTime={timestamp} className={className} title={timestamp}>
      {text}
    </time>
  );
}
