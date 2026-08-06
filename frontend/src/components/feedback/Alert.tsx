"use client";

import { AlertCircle, CheckCircle, Info, XCircle } from "lucide-react";
import { ReactNode } from "react";

export type AlertType = "success" | "error" | "info" | "warning";

interface AlertProps {
  type: AlertType;
  title?: string;
  children: ReactNode;
  className?: string;
}

const alertConfig = {
  success: {
    icon: CheckCircle,
    styles: "bg-[#B4E2C6]/20 border-[#B4E2C6] text-[#1A1A1A]",
    iconColor: "text-green-600",
  },
  error: {
    icon: XCircle,
    styles: "bg-red-50 border-red-200 text-red-900",
    iconColor: "text-red-500",
  },
  info: {
    icon: Info,
    styles: "bg-blue-50 border-blue-200 text-blue-900",
    iconColor: "text-blue-500",
  },
  warning: {
    icon: AlertCircle,
    styles: "bg-yellow-50 border-yellow-200 text-yellow-900",
    iconColor: "text-yellow-600",
  },
};

export function Alert({ type, title, children, className = "" }: AlertProps) {
  const config = alertConfig[type];
  const Icon = config.icon;

  return (
    <div className={`p-4 rounded-xl border flex items-start space-x-3 ${config.styles} ${className}`}>
      <Icon className={`w-5 h-5 flex-shrink-0 mt-0.5 ${config.iconColor}`} />
      <div>
        {title && <h4 className="font-semibold mb-1">{title}</h4>}
        <div className="text-sm opacity-90 leading-relaxed">{children}</div>
      </div>
    </div>
  );
}
