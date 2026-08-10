"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Info, AlertTriangle, XCircle, X } from "lucide-react";

export type ToastType = "success" | "error" | "info" | "warning";

export type ToastItem = {
  id: string;
  type: ToastType;
  title: string;
  description?: string;
  duration?: number;
};

type ToastContextValue = {
  toast: (input: Omit<ToastItem, "id">) => void;
  success: (title: string, description?: string) => void;
  error: (title: string, description?: string) => void;
  info: (title: string, description?: string) => void;
  dismiss: (id: string) => void;
};

const ToastContext = createContext<ToastContextValue | null>(null);

const icons = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
  warning: AlertTriangle,
};

const styles = {
  success: "border-[#B4E2C6] bg-white",
  error: "border-red-200 bg-white",
  info: "border-blue-200 bg-white",
  warning: "border-amber-200 bg-white",
};

const iconStyles = {
  success: "text-green-600",
  error: "text-red-500",
  info: "text-blue-500",
  warning: "text-amber-600",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback(
    (input: Omit<ToastItem, "id">) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const item: ToastItem = { duration: 3500, ...input, id };
      setToasts((prev) => [...prev.slice(-4), item]);
      window.setTimeout(() => dismiss(id), item.duration);
    },
    [dismiss]
  );

  const value = useMemo<ToastContextValue>(
    () => ({
      toast,
      success: (title, description) => toast({ type: "success", title, description }),
      error: (title, description) => toast({ type: "error", title, description }),
      info: (title, description) => toast({ type: "info", title, description }),
      dismiss,
    }),
    [toast, dismiss]
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed bottom-6 right-6 z-[100] flex flex-col gap-3 w-[min(100vw-2rem,380px)] pointer-events-none">
        <AnimatePresence>
          {toasts.map((t) => {
            const Icon = icons[t.type];
            return (
              <motion.div
                key={t.id}
                initial={{ opacity: 0, y: 16, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.96 }}
                transition={{ duration: 0.22 }}
                className={`pointer-events-auto rounded-2xl border shadow-[0_12px_40px_rgba(0,0,0,0.1)] p-4 flex gap-3 ${styles[t.type]}`}
              >
                <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${iconStyles[t.type]}`} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-[#1A1A1A]">{t.title}</p>
                  {t.description && (
                    <p className="text-xs text-[#767676] mt-0.5 leading-relaxed">{t.description}</p>
                  )}
                </div>
                <button
                  onClick={() => dismiss(t.id)}
                  className="text-gray-400 hover:text-[#1A1A1A] transition-colors shrink-0"
                  aria-label="Dismiss"
                >
                  <X className="w-4 h-4" />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}
