"use client";

import { motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
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
  /** Hide checklist — use in side panels so Live Progress is not duplicated. */
  compact?: boolean;
  className?: string;
}

export function LiveProgress({
  stages,
  stageIndex,
  stageLabel,
  percent,
  elapsedMs,
  failed = false,
  compact = false,
  className,
}: LiveProgressProps) {
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));

  return (
    <div className={cn("w-full space-y-4", className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2 min-w-0">
          {!failed && clamped < 100 && (
            <Loader2 className="w-4 h-4 text-[#D8B4E2] animate-spin shrink-0 mt-0.5" />
          )}
          <div className="min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-wider text-[#767676]">
              {failed ? "Failed" : clamped >= 100 ? "Completed" : "Current stage"}
            </p>
            <p className="text-sm font-semibold text-[#1A1A1A] leading-snug break-words">
              {stageLabel}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs text-[#767676] shrink-0 pt-0.5">
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

      {!compact && (
        <ol className="space-y-2">
          {stages.map((stage, i) => {
            const done = i < stageIndex || (clamped >= 100 && i === stages.length - 1);
            const active = i === stageIndex && clamped < 100 && !failed;
            const pending = !done && !active;
            return (
              <li
                key={stage.id}
                className={cn(
                  "flex items-center gap-3 transition-opacity",
                  pending && "opacity-40",
                  active && "opacity-100",
                  done && "opacity-70"
                )}
              >
                <span
                  className={cn(
                    "w-4 h-4 rounded-full shrink-0 flex items-center justify-center transition-colors",
                    failed && active
                      ? "bg-red-400"
                      : done
                        ? "bg-[#B4E2C6]"
                        : active
                          ? "bg-[#D8B4E2] animate-pulse"
                          : "bg-gray-200"
                  )}
                >
                  {done && <Check className="w-2.5 h-2.5 text-[#1A1A1A]" strokeWidth={3} />}
                </span>
                <span
                  className={cn(
                    "text-xs transition-colors",
                    active
                      ? "text-[#1A1A1A] font-semibold"
                      : done
                        ? "text-[#1A1A1A] font-medium line-through decoration-gray-300"
                        : "text-gray-400 font-normal"
                  )}
                >
                  {active ? stageLabel : stage.label}
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
