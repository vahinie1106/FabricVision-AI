"use client";

import { LucideIcon } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
  compact?: boolean;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  actionLabel,
  onAction,
  className,
  compact = false,
}: EmptyStateProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "w-full flex flex-col items-center justify-center text-center bg-white border border-dashed border-gray-200 rounded-3xl",
        compact ? "min-h-[180px] p-6" : "min-h-[400px] p-12",
        className
      )}
    >
      <div
        className={cn(
          "rounded-full flex items-center justify-center mb-4 bg-[#F7F5F0]",
          compact ? "w-14 h-14" : "w-20 h-20 mb-6"
        )}
      >
        <Icon className={cn(compact ? "w-7 h-7" : "w-10 h-10", "text-[#767676]")} />
      </div>
      <h3 className={cn("font-bold text-[#1A1A1A] mb-2", compact ? "text-base" : "text-2xl mb-3")}>
        {title}
      </h3>
      <p
        className={cn(
          "text-[#767676] max-w-md mx-auto leading-relaxed",
          compact ? "text-xs mb-0" : "mb-8"
        )}
      >
        {description}
      </p>
      {actionLabel && onAction && (
        <Button onClick={onAction} className="px-8 py-3 mt-4">
          {actionLabel}
        </Button>
      )}
    </motion.div>
  );
}
