"use client";

import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import type { ProgressStageDef } from "@/lib/progressStages";
import { formatElapsed } from "@/hooks/useLiveProgress";
import { cn } from "@/lib/cn";

interface LiveProgressProps {
  stages: ProgressStageDef[];
  stageIndex: number;
  stageLabel: string;
  percent: number;
  elapsedMs: number;
  failed?: boolean;
  className?: string;
}

export function LiveProgress({
  stages,
  stageIndex,
  stageLabel,
  percent,
  elapsedMs,
  failed = false,
  className,
}: LiveProgressProps) {
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));

  return (
    <div className={cn("w-full space-y-5", className)}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          {!failed && clamped < 100 && (
            <Loader2 className="w-4 h-4 text-[#D8B4E2] animate-spin shrink-0" />
          )}
          <p className="text-sm font-semibold text-[#1A1A1A] truncate">{stageLabel}</p>
        </div>
        <div className="flex items-center gap-3 text-xs text-[#767676] shrink-0">
          <span>{formatElapsed(elapsedMs)}</span>
          <span className="font-semibold text-[#1A1A1A] tabular-nums">{clamped}%</span>
        </div>
      </div>

      <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
        <motion.div
          className={cn(
            "h-full rounded-full",
            failed ? "bg-red-400" : clamped >= 100 ? "bg-[#B4E2C6]" : "bg-[#1A1A1A]"
          )}
          initial={{ width: 0 }}
          animate={{ width: `${clamped}%` }}
          transition={{ duration: 0.45, ease: "easeOut" }}
        />
      </div>

      <ol className="space-y-2.5">
        {stages.map((stage, i) => {
          const done = i < stageIndex || (clamped >= 100 && i === stages.length - 1);
          const active = i === stageIndex && clamped < 100 && !failed;
          return (
            <li key={stage.id} className="flex items-center gap-3">
              <span
                className={cn(
                  "w-2 h-2 rounded-full shrink-0 transition-colors",
                  failed && active
                    ? "bg-red-400"
                    : done
                      ? "bg-[#B4E2C6]"
                      : active
                        ? "bg-[#D8B4E2] animate-pulse"
                        : "bg-gray-200"
                )}
              />
              <span
                className={cn(
                  "text-xs transition-colors",
                  done || active ? "text-[#1A1A1A] font-medium" : "text-gray-400"
                )}
              >
                {stage.label}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
