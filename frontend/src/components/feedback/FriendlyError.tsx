"use client";

import { AlertCircle } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/Button";
import type { FriendlyError as FriendlyErrorType } from "@/lib/friendlyErrors";
import { cn } from "@/lib/cn";

interface FriendlyErrorProps {
  error: FriendlyErrorType;
  onRetry?: () => void;
  className?: string;
}

export function FriendlyError({ error, onRetry, className }: FriendlyErrorProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "w-full rounded-2xl border border-red-100 bg-red-50/80 p-5 flex flex-col sm:flex-row sm:items-center gap-4",
        className
      )}
    >
      <div className="flex items-start gap-3 flex-1 min-w-0">
        <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-semibold text-[#1A1A1A]">{error.title}</h4>
          <p className="text-sm text-[#767676] mt-1 leading-relaxed">{error.message}</p>
        </div>
      </div>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry} className="shrink-0 py-2.5 px-5 text-sm">
          Retry
        </Button>
      )}
    </motion.div>
  );
}
