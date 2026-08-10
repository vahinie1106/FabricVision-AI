"use client";

import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Download,
  Expand,
  FileJson,
  FileText,
  Package,
  ZoomIn,
  ZoomOut,
  X,
  Clock,
  Cpu,
  Maximize2,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { resolveMediaUrl } from "@/lib/resolveMediaUrl";
import { formatElapsed } from "@/hooks/useLiveProgress";
import { cn } from "@/lib/cn";

export type ResultMeta = {
  model?: string;
  timestamp?: string;
  resolution?: string;
  durationMs?: number;
  promptSummary?: string;
  extra?: Record<string, string | number | undefined>;
};

interface ResultCardProps {
  imageUrl: string;
  title?: string;
  meta?: ResultMeta;
  className?: string;
  children?: React.ReactNode;
}

export function ResultCard({
  imageUrl,
  title = "Generated Result",
  meta,
  className,
  children,
}: ResultCardProps) {
  const [zoom, setZoom] = useState(1);
  const [fullscreen, setFullscreen] = useState(false);
  const src = resolveMediaUrl(imageUrl) || imageUrl;

  const toggleZoom = useCallback(() => {
    setZoom((z) => (z >= 1.75 ? 1 : Number((z + 0.25).toFixed(2))));
  }, []);

  return (
    <>
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.35 }}
        className={cn(
          "w-full bg-white rounded-2xl border border-gray-100 shadow-[0_8px_30px_rgba(0,0,0,0.04)] overflow-hidden flex flex-col",
          className
        )}
      >
        <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between gap-3">
          <h3 className="text-sm font-bold text-[#1A1A1A] truncate">{title}</h3>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(1, Number((z - 0.25).toFixed(2))))}
              className="p-2 rounded-lg text-[#767676] hover:bg-[#F7F5F0] hover:text-[#1A1A1A] transition-colors"
              aria-label="Zoom out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={toggleZoom}
              className="p-2 rounded-lg text-[#767676] hover:bg-[#F7F5F0] hover:text-[#1A1A1A] transition-colors"
              aria-label="Zoom in"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setFullscreen(true)}
              className="p-2 rounded-lg text-[#767676] hover:bg-[#F7F5F0] hover:text-[#1A1A1A] transition-colors"
              aria-label="View full resolution"
              title="View full resolution"
            >
              <Expand className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="relative bg-[#F7F5F0] aspect-[4/5] overflow-hidden flex items-center justify-center">
          {/* Full-resolution result: no lazy load, no CSS blur filters */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={title}
            className="max-w-full max-h-full object-contain transition-transform duration-300 origin-center"
            style={{ transform: `scale(${zoom})`, imageRendering: "auto" }}
            decoding="sync"
          />
        </div>
        <div className="px-4 py-2 border-t border-gray-50 flex items-center justify-between gap-2">
          <p className="text-[10px] text-[#767676]">
            {meta?.resolution ? `Native ${meta.resolution}` : "Native PNG"} · no CSS sharpening
          </p>
          <a
            href={src}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] font-semibold text-[#1A1A1A] hover:underline"
            title="Open the original backend PNG at native resolution"
          >
            View Full Resolution
          </a>
        </div>

        {(meta || children) && (
          <div className="p-4 space-y-3 border-t border-gray-100">
            {meta && (
              <div className="grid grid-cols-2 gap-3 text-xs">
                {meta.timestamp && (
                  <MetaRow icon={Clock} label="Generated" value={new Date(meta.timestamp).toLocaleString()} />
                )}
                {meta.model && <MetaRow icon={Cpu} label="Model" value={meta.model} />}
                {meta.resolution && (
                  <MetaRow icon={Maximize2} label="Resolution" value={meta.resolution} />
                )}
                {typeof meta.durationMs === "number" && (
                  <MetaRow icon={Clock} label="Duration" value={formatElapsed(meta.durationMs)} />
                )}
                {meta.extra &&
                  Object.entries(meta.extra).map(([k, v]) =>
                    v != null && v !== "" ? (
                      <MetaRow key={k} icon={Cpu} label={k} value={String(v)} />
                    ) : null
                  )}
                {meta.promptSummary && (
                  <div className="col-span-2">
                    <p className="text-[10px] uppercase tracking-wider text-[#767676] mb-1">Prompt</p>
                    <p className="text-xs text-[#1A1A1A] leading-relaxed line-clamp-3">{meta.promptSummary}</p>
                  </div>
                )}
              </div>
            )}
            {children}
          </div>
        )}
      </motion.div>

      <AnimatePresence>
        {fullscreen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[90] bg-black/90 flex items-center justify-center p-4"
            onClick={() => setFullscreen(false)}
          >
            <button
              type="button"
              className="absolute top-4 right-4 p-2 rounded-full bg-white/10 text-white hover:bg-white/20"
              onClick={() => setFullscreen(false)}
              aria-label="Close fullscreen"
            >
              <X className="w-6 h-6" />
            </button>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={src}
              alt={title}
              className="max-w-full max-h-full object-contain"
              onClick={(e) => e.stopPropagation()}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function MetaRow({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-2 min-w-0">
      <Icon className="w-3.5 h-3.5 text-[#767676] mt-0.5 shrink-0" />
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wider text-[#767676]">{label}</p>
        <p className="text-xs font-medium text-[#1A1A1A] truncate">{value}</p>
      </div>
    </div>
  );
}

export type DownloadAction = {
  id: string;
  label: string;
  icon?: "png" | "json" | "txt" | "zip" | "prompt";
  onClick: () => void | Promise<void>;
  disabled?: boolean;
};

const downloadIcons = {
  png: Download,
  json: FileJson,
  txt: FileText,
  zip: Package,
  prompt: FileText,
};

interface DownloadCenterProps {
  actions: DownloadAction[];
  className?: string;
}

export function DownloadCenter({ actions, className }: DownloadCenterProps) {
  const [busy, setBusy] = useState<string | null>(null);

  return (
    <div className={cn("space-y-2", className)}>
      <p className="text-[10px] font-bold uppercase tracking-wider text-[#767676] mb-2">Download Center</p>
      {actions.map((action) => {
        const Icon = downloadIcons[action.icon || "png"];
        return (
          <Button
            key={action.id}
            variant={action.id === actions[0]?.id ? "primary" : "secondary"}
            className="w-full py-2.5 text-sm justify-start"
            disabled={action.disabled || busy === action.id}
            isLoading={busy === action.id}
            onClick={async () => {
              try {
                setBusy(action.id);
                await action.onClick();
              } finally {
                setBusy(null);
              }
            }}
          >
            <Icon className="w-4 h-4 mr-2" />
            {action.label}
          </Button>
        );
      })}
    </div>
  );
}
