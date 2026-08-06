"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { name: "Home", href: "/" },
    { name: "AI Studio", href: "/studio" },
    { name: "Projects", href: "/projects" },
    { name: "About", href: "/about" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full backdrop-blur-md bg-[#FDFCFB]/80 border-b border-gray-200 shadow-[0px_4px_24px_rgba(0,0,0,0.02)]">
      <div className="max-w-[1440px] mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-2">
          <span className="font-[family-name:var(--font-outfit)] text-xl font-semibold tracking-tight text-[#1A1A1A]">
            FabricVision<span className="text-[#D8B4E2]">-AI</span>
          </span>
        </Link>

        <nav className="hidden md:flex space-x-8">
          {links.map((link) => {
            const isActive = pathname === link.href || pathname.startsWith(link.href + "/");
            return (
              <Link
                key={link.name}
                href={link.href}
                className="relative text-sm font-medium transition-colors hover:text-[#1A1A1A]"
              >
                <span className={isActive ? "text-[#1A1A1A]" : "text-[#767676]"}>
                  {link.name}
                </span>
                {isActive && (
                  <motion.div
                    layoutId="navbar-indicator"
                    className="absolute -bottom-[21px] left-0 right-0 h-[2px] bg-[#1A1A1A]"
                    initial={false}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center space-x-4">
          <button className="text-sm font-medium text-[#333333] hover:text-[#1A1A1A]">
            Log in
          </button>
          <button className="text-sm font-semibold bg-[#1A1A1A] text-white px-4 py-2 rounded-xl hover:bg-[#333333] transition-transform hover:scale-[1.02] shadow-[0_4px_12px_rgba(26,26,26,0.2)]">
            Start Designing
          </button>
        </div>
      </div>
    </header>
  );
}
