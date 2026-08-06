import { ButtonHTMLAttributes, forwardRef } from "react";
import { Loader2 } from "lucide-react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "destructive";
  isLoading?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = "", variant = "primary", isLoading, children, disabled, ...props }, ref) => {
    const baseStyles = "inline-flex items-center justify-center rounded-xl font-semibold transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2";
    
    const variants = {
      primary: "bg-[#1A1A1A] text-white hover:bg-[#333333] hover:scale-[1.02] shadow-[0_4px_12px_rgba(26,26,26,0.2)] focus:ring-[#1A1A1A]",
      secondary: "bg-[#F7F5F0] text-[#1A1A1A] hover:bg-[#EBE7DF] border border-transparent focus:ring-[#EBE7DF]",
      ghost: "bg-transparent text-[#767676] hover:text-[#1A1A1A] hover:bg-[#F7F5F0]",
      destructive: "bg-[#E2B4B4] text-white hover:bg-[#D69A9A] focus:ring-[#E2B4B4]",
    };

    const disabledStyles = disabled || isLoading ? "opacity-50 cursor-not-allowed transform-none hover:scale-100" : "";

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={`${baseStyles} ${variants[variant]} ${disabledStyles} px-6 py-3 ${className}`}
        {...props}
      >
        {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";

export { Button };
