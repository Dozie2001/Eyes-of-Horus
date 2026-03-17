"use client";

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThemeToggleProps {
  className?: string;
}

export function ThemeToggle({ className }: ThemeToggleProps) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Wait until after hydration to render theme-dependent UI
  useEffect(() => setMounted(true), []);

  if (!mounted) {
    // Placeholder with same dimensions to prevent layout shift
    return <div className={cn("h-7 w-14 rounded-full border border-border bg-secondary", className)} />;
  }

  const isDark = resolvedTheme === "dark";

  return (
    <div
      className={cn(
        "flex h-7 w-14 cursor-pointer items-center rounded-full p-1 transition-all duration-300",
        isDark
          ? "border border-border bg-secondary"
          : "border border-border bg-white",
        className
      )}
      onClick={() => setTheme(isDark ? "light" : "dark")}
      role="button"
      tabIndex={0}
    >
      <div className="flex w-full items-center justify-between">
        <div
          className={cn(
            "flex size-5 items-center justify-center rounded-full transition-transform duration-300",
            isDark
              ? "translate-x-0 bg-muted"
              : "translate-x-7 bg-gray-200"
          )}
        >
          {isDark ? (
            <Moon className="size-3 text-foreground" strokeWidth={1.5} />
          ) : (
            <Sun className="size-3 text-gray-700" strokeWidth={1.5} />
          )}
        </div>
        <div
          className={cn(
            "flex size-5 items-center justify-center rounded-full transition-transform duration-300",
            isDark ? "bg-transparent" : "-translate-x-7"
          )}
        >
          {isDark ? (
            <Sun className="size-3 text-muted-foreground/50" strokeWidth={1.5} />
          ) : (
            <Moon className="size-3 text-black" strokeWidth={1.5} />
          )}
        </div>
      </div>
    </div>
  );
}
