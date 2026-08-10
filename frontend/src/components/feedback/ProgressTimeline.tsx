"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";

export type TimelineStep = {
  id: string;
  label: string;
  status: "waiting" | "processing" | "completed" | "failed";
};

interface ProgressTimelineProps {
  steps: TimelineStep[];
}

export function ProgressTimeline({ steps }: ProgressTimelineProps) {
  return (
    <div className="w-full flex flex-col space-y-6">
      {steps.map((step, index) => (
        <motion.div
          key={step.id}
          className="flex items-start"
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.05 }}
        >
          <div className="flex flex-col items-center mr-4">
            <div className="w-8 h-8 flex items-center justify-center bg-white rounded-full">
              {step.status === "completed" && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 300, damping: 20 }}
                >
                  <CheckCircle2 className="w-6 h-6 text-[#B4E2C6]" />
                </motion.div>
              )}
              {step.status === "processing" && (
                <Loader2 className="w-6 h-6 text-[#D8B4E2] animate-spin" />
              )}
              {step.status === "failed" && (
                <XCircle className="w-6 h-6 text-red-500" />
              )}
              {step.status === "waiting" && (
                <Circle className="w-6 h-6 text-gray-200" />
              )}
            </div>
            {index !== steps.length - 1 && (
              <div
                className={`w-[2px] h-8 mt-2 rounded-full ${
                  step.status === "completed" ? "bg-[#B4E2C6]" : "bg-gray-100"
                }`}
              />
            )}
          </div>
          <div className="flex flex-col pt-1">
            <p
              className={`text-sm font-semibold uppercase tracking-wider ${
                step.status === "waiting" ? "text-gray-400" : "text-[#1A1A1A]"
              }`}
            >
              {step.id}
            </p>
            <p
              className={`text-base font-medium ${
                step.status === "waiting" ? "text-gray-400" : "text-[#767676]"
              }`}
            >
              {step.label}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
