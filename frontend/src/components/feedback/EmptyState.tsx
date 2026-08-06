import { LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
}

export function EmptyState({ icon: Icon, title, description, actionLabel, onAction }: EmptyStateProps) {
  return (
    <div className="w-full min-h-[400px] flex flex-col items-center justify-center p-12 text-center bg-white border border-dashed border-gray-200 rounded-3xl">
      <div className="w-20 h-20 bg-[#F7F5F0] rounded-full flex items-center justify-center mb-6">
        <Icon className="w-10 h-10 text-[#767676]" />
      </div>
      <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">{title}</h3>
      <p className="text-[#767676] max-w-md mx-auto mb-8 leading-relaxed">
        {description}
      </p>
      {actionLabel && onAction && (
        <Button onClick={onAction} className="px-8 py-3">
          {actionLabel}
        </Button>
      )}
    </div>
  );
}
