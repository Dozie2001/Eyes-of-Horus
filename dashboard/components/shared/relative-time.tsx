"use client";

import { useState, useEffect } from "react";

/**
 * Parse a timestamp string as local time.
 *
 * The backend stores timezone-naive ISO strings (e.g. "2026-03-16T18:08:53").
 * new Date() treats these inconsistently — some browsers assume UTC, others local.
 * We parse manually to guarantee local time interpretation.
 */
function parseLocalTime(isoString: string): Date {
  // If it has timezone info (Z or +/-), let the browser handle it
  if (isoString.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(isoString)) {
    return new Date(isoString);
  }
  // Parse as local time: "2026-03-16T18:08:53.123456"
  const [datePart, timePart] = isoString.split("T");
  const [year, month, day] = datePart.split("-").map(Number);
  if (!timePart) return new Date(year, month - 1, day);
  const [h, m, s] = timePart.split(":").map(Number);
  return new Date(year, month - 1, day, h || 0, m || 0, Math.floor(s || 0));
}

function formatRelative(isoString: string): string {
  try {
    const date = parseLocalTime(isoString);
    const diffSec = Math.floor((Date.now() - date.getTime()) / 1000);

    if (diffSec < 0) return "just now";
    if (diffSec < 60) return "just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)} min ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} hr ago`;
    return `${Math.floor(diffSec / 86400)} days ago`;
  } catch {
    return isoString;
  }
}

interface RelativeTimeProps {
  timestamp: string;
  className?: string;
}

export function RelativeTime({ timestamp, className }: RelativeTimeProps) {
  const [text, setText] = useState(() => formatRelative(timestamp));

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
