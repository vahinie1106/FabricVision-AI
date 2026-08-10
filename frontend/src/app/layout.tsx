import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { ToastProvider } from "@/hooks/useToast";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });

export const metadata: Metadata = {
  title: "FabricVision-AI | Premium Fashion Intelligence",
  description: "AI-powered fashion design platform by Fabriplay",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${outfit.variable} antialiased min-h-screen flex flex-col bg-[#FDFCFB]`}>
        <ToastProvider>
          <Navbar />
          <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
            {children}
          </main>
          <Footer />
        </ToastProvider>
      </body>
    </html>
  );
}
