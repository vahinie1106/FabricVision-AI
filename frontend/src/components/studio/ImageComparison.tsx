"use client";

import { useRef, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface ImageComparisonProps {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export function ImageComparison({
  beforeSrc,
  afterSrc,
  beforeLabel = "Original Fabric",
  afterLabel = "Generated Garment",
  className,
}: ImageComparisonProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(50);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.max(2, Math.min(98, pct)));
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    updateFromClientX(e.clientX);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    updateFromClientX(e.clientX);
  };

  const onPointerUp = () => {
    dragging.current = false;
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={cn("w-full", className)}
    >
      <div className="flex justify-between text-[10px] uppercase tracking-wider text-[#767676] mb-2 px-1">
        <span>{beforeLabel}</span>
        <span>{afterLabel}</span>
      </div>
      <div
        ref={containerRef}
        className="relative w-full aspect-[4/5] rounded-2xl overflow-hidden select-none cursor-ew-resize bg-[#F7F5F0] border border-gray-100 touch-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        role="slider"
        aria-valuenow={Math.round(position)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Before after comparison"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={afterSrc} alt={afterLabel} className="absolute inset-0 w-full h-full object-contain" draggable={false} />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={beforeSrc}
          alt={beforeLabel}
          className="absolute inset-0 w-full h-full object-contain"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          draggable={false}
        />
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.35)]"
          style={{ left: `${position}%` }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 rounded-full bg-white shadow-lg flex items-center justify-center">
            <div className="flex gap-0.5">
              <span className="w-0.5 h-3 bg-[#1A1A1A] rounded-full" />
              <span className="w-0.5 h-3 bg-[#1A1A1A] rounded-full" />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
