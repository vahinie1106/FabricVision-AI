"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Clock, History, Trash2, PanelRightClose, PanelRightOpen } from "lucide-react";
import type { HistoryItem } from "@/hooks/useGenerationHistory";
import { EmptyState } from "@/components/feedback/EmptyState";
import { cn } from "@/lib/cn";

interface HistorySidebarProps {
  items: HistoryItem[];
  open: boolean;
  onToggle: () => void;
  onSelect: (item: HistoryItem) => void;
  onRemove?: (id: string) => void;
  activeId?: string | null;
  className?: string;
}

export function HistorySidebar({
  items,
  open,
  onToggle,
  onSelect,
  onRemove,
  activeId,
  className,
}: HistorySidebarProps) {
  return (
    <div className={cn("relative shrink-0", className)}>
      <button
        type="button"
        onClick={onToggle}
        className="absolute -left-10 top-0 z-20 p-2 rounded-xl bg-white border border-gray-200 shadow-sm text-[#767676] hover:text-[#1A1A1A] hover:bg-[#F7F5F0] transition-colors"
        aria-label={open ? "Close history" : "Open history"}
        title="Generation History"
      >
        {open ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="h-full bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden flex flex-col"
          >
            <div className="p-4 border-b border-gray-100 flex items-center gap-2 shrink-0">
              <History className="w-4 h-4 text-[#1A1A1A]" />
              <h3 className="text-sm font-bold text-[#1A1A1A]">History</h3>
              <span className="ml-auto text-[10px] text-[#767676] tabular-nums">{items.length}</span>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {items.length === 0 ? (
                <EmptyState
                  icon={History}
                  title="No history available."
                  description="Completed generations will appear here for quick restore."
                  className="min-h-[200px] border-0 shadow-none p-6"
                  compact
                />
              ) : (
                items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onSelect(item)}
                    className={cn(
                      "w-full text-left rounded-xl border p-2 flex gap-3 transition-all hover:border-[#1A1A1A]/30 hover:bg-[#FDFCFB]",
                      activeId === item.id ? "border-[#1A1A1A] bg-[#F7F5F0]" : "border-gray-100"
                    )}
                  >
                    <div className="w-14 h-14 rounded-lg overflow-hidden bg-[#F7F5F0] shrink-0">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={item.thumbnailUrl}
                        alt=""
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-semibold text-[#1A1A1A] truncate">{item.title}</p>
                      <p className="text-[10px] text-[#767676] mt-0.5 truncate">{item.model}</p>
                      <div className="flex items-center gap-1 mt-1 text-[10px] text-[#767676]">
                        <Clock className="w-3 h-3" />
                        <span>{new Date(item.createdAt).toLocaleString()}</span>
                      </div>
                      <span
                        className={cn(
                          "inline-block mt-1 text-[9px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded",
                          item.status === "completed"
                            ? "bg-[#B4E2C6]/30 text-green-800"
                            : "bg-red-50 text-red-600"
                        )}
                      >
                        {item.status}
                      </span>
                    </div>
                    {onRemove && (
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={(e) => {
                          e.stopPropagation();
                          onRemove(item.id);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.stopPropagation();
                            onRemove(item.id);
                          }
                        }}
                        className="p-1 self-start text-gray-300 hover:text-red-500 transition-colors"
                        aria-label="Remove from history"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </span>
                    )}
                  </button>
                ))
              )}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}
