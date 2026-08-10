"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getStages,
  resolveStageIndex,
  type ProgressStageDef,
  type WorkflowKind,
} from "@/lib/progressStages";

export type LiveProgressState = {
  active: boolean;
  percent: number;
  stageIndex: number;
  stageLabel: string;
  stages: ProgressStageDef[];
  elapsedMs: number;
  completed: boolean;
  failed: boolean;
};

const initial = (kind: WorkflowKind): LiveProgressState => {
  const stages = getStages(kind);
  return {
    active: false,
    percent: 0,
    stageIndex: 0,
    stageLabel: stages[0]?.label || "",
    stages,
    elapsedMs: 0,
    completed: false,
    failed: false,
  };
};

export function useLiveProgress(kind: WorkflowKind) {
  const [state, setState] = useState<LiveProgressState>(() => initial(kind));
  const startRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const autoRef = useRef<number | null>(null);

  const clearTimers = useCallback(() => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    if (autoRef.current) window.clearInterval(autoRef.current);
    timerRef.current = null;
    autoRef.current = null;
  }, []);

  const start = useCallback(() => {
    clearTimers();
    startRef.current = Date.now();
    const stages = getStages(kind);
    setState({
      active: true,
      percent: stages[0]?.percent || 5,
      stageIndex: 0,
      stageLabel: stages[0]?.label || "",
      stages,
      elapsedMs: 0,
      completed: false,
      failed: false,
    });

    timerRef.current = window.setInterval(() => {
      if (startRef.current) {
        setState((s) => ({ ...s, elapsedMs: Date.now() - startRef.current! }));
      }
    }, 250);

    // Soft auto-advance ONLY for sync analysis (no authoritative backend stages).
    // Garment / try-on jobs poll real backend progress — auto-advance previously
    // jumped the UI to 94% "Saving result" while FLUX was still generating.
    if (kind === "analysis") {
      autoRef.current = window.setInterval(() => {
        setState((s) => {
          if (!s.active || s.completed || s.failed) return s;
          const nextIdx = Math.min(s.stageIndex + 1, s.stages.length - 2);
          if (nextIdx === s.stageIndex) return s;
          const stage = s.stages[nextIdx];
          return {
            ...s,
            stageIndex: nextIdx,
            stageLabel: stage.label,
            percent: Math.max(s.percent, stage.percent),
          };
        });
      }, 1800);
    }
  }, [kind, clearTimers]);

  const update = useCallback((progress: number, currentStep?: string) => {
    setState((s) => {
      const stages = s.stages.length ? s.stages : getStages(kind);
      const idx = resolveStageIndex(stages, progress, currentStep);
      const stage = stages[idx];
      // Backend progress is authoritative for async jobs. Never invent a higher
      // percent from the stage table (that caused the 94% Saving stall).
      const backendPct =
        typeof progress === "number" && Number.isFinite(progress) ? progress : 0;
      const nextPercent =
        kind === "analysis"
          ? Math.max(s.percent, Math.min(99, backendPct || stage.percent))
          : Math.max(s.percent, Math.min(99, backendPct > 0 ? backendPct : s.percent));
      return {
        ...s,
        active: true,
        stages,
        stageIndex: idx,
        stageLabel: currentStep && currentStep.length < 60 ? currentStep : stage.label,
        percent: nextPercent,
      };
    });
  }, [kind]);

  const complete = useCallback(() => {
    clearTimers();
    const elapsed = startRef.current ? Date.now() - startRef.current : 0;
    setState((s) => {
      const stages = s.stages.length ? s.stages : getStages(kind);
      return {
        ...s,
        active: false,
        completed: true,
        failed: false,
        percent: 100,
        stageIndex: stages.length - 1,
        stageLabel: stages[stages.length - 1]?.label || "Completed",
        stages,
        elapsedMs: elapsed || s.elapsedMs,
      };
    });
    return elapsed;
  }, [kind, clearTimers]);

  const fail = useCallback(() => {
    clearTimers();
    const elapsed = startRef.current ? Date.now() - startRef.current : 0;
    setState((s) => ({
      ...s,
      active: false,
      failed: true,
      completed: false,
      elapsedMs: elapsed || s.elapsedMs,
    }));
    return elapsed;
  }, [clearTimers]);

  const reset = useCallback(() => {
    clearTimers();
    startRef.current = null;
    setState(initial(kind));
  }, [kind, clearTimers]);

  useEffect(() => () => clearTimers(), [clearTimers]);

  return { ...state, start, update, complete, fail, reset };
}

export function formatElapsed(ms: number): string {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m > 0) return `${m}m ${s.toString().padStart(2, "0")}s`;
  return `${s}s`;
}
