"use client";

import { cn } from "@/lib/cn";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return <div className={cn("rounded-xl loading-shimmer", className)} aria-hidden />;
}
