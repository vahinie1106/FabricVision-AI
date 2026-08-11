"use client";

import Link from "next/link";
import { ArrowRight, Sparkles, Wand2, Layers, SearchCode, MoveRight } from "lucide-react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function Home() {
  return (
    <div className="flex flex-col items-center">
      {/* Hero Section */}
      <section className="w-full max-w-[1200px] flex flex-col lg:flex-row items-center justify-between mt-12 mb-32 gap-16">
        
        {/* Left: Text & CTA */}
        <div className="w-full lg:w-1/2 text-left">
          <motion.div 
            initial={{ opacity: 1, y: 0 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center space-x-2 bg-white px-4 py-2 rounded-full border border-gray-200 shadow-sm mb-8"
          >
            <Sparkles className="w-4 h-4 text-[#D8B4E2]" />
            <span className="text-xs font-semibold uppercase tracking-wider text-[#767676]">FabricVision-AI 2.0</span>
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 1, y: 0 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl lg:text-6xl font-bold text-[#1A1A1A] leading-tight mb-6"
          >
            Transform Fabrics Into <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#1A1A1A] to-[#767676]">
              Intelligent Fashion
            </span>
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 1, y: 0 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg text-[#767676] max-w-lg mb-10 leading-relaxed"
          >
            The professional workspace for fashion intelligence. Generate photorealistic garments, perform virtual try-ons, and extract semantic metadata with state-of-the-art AI.
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 1, y: 0 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center space-y-4 sm:space-y-0 sm:space-x-4"
          >
            <Link href="/studio/custom-garment" className="w-full sm:w-auto">
              <Button className="w-full py-4 px-8 text-base">Start Creating</Button>
            </Link>
            <Link href="/about" className="w-full sm:w-auto">
              <Button variant="secondary" className="w-full py-4 px-8 text-base bg-white shadow-sm border border-gray-200">
                Explore Technology <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
          </motion.div>
        </div>

        {/* Right: Interactive AI Demo Area */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="w-full lg:w-1/2 relative min-h-[400px] flex items-center justify-center"
        >
          <div className="absolute inset-0 bg-gradient-to-tr from-[#FDFCFB] to-[#F7F5F0] rounded-3xl -z-10 transform rotate-3 scale-105" />
          
          <Card className="w-full max-w-md p-6 bg-white/90 backdrop-blur-xl border border-white/50 shadow-[0_20px_40px_rgba(0,0,0,0.08)] relative z-10">
            <div className="flex flex-col space-y-6">
              
              {/* Step 1: Input */}
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 rounded-xl bg-gray-200 bg-[url('https://www.transparenttextures.com/patterns/black-linen.png')] flex-shrink-0" />
                <div>
                  <p className="text-xs font-bold uppercase tracking-wider text-[#767676] mb-1">Input Concept</p>
                  <p className="text-sm font-medium text-[#1A1A1A]">Raw Linen Fabric</p>
                </div>
              </div>
              
              {/* Step 2: Processing */}
              <div className="flex justify-center">
                <div className="px-4 py-2 bg-[#F7F5F0] rounded-full flex items-center space-x-2 shadow-inner border border-gray-100">
                  <div className="w-2 h-2 bg-[#D8B4E2] rounded-full animate-pulse" />
                  <span className="text-xs font-semibold text-[#1A1A1A]">FLUX Synthesis Active</span>
                </div>
              </div>
              
              {/* Step 3: Output */}
              <div className="w-full h-48 bg-gradient-to-br from-gray-100 to-gray-200 rounded-xl relative overflow-hidden flex flex-col items-center justify-center">
                 <Sparkles className="w-8 h-8 text-white/50 mb-2" />
                 <p className="text-sm font-semibold text-gray-400">Generated AI Garment</p>
              </div>

            </div>
          </Card>
        </motion.div>
      </section>

      {/* Feature Showcase Grid */}
      <section className="w-full max-w-[1200px] grid grid-cols-1 md:grid-cols-3 gap-8 pb-32">
        <Link href="/studio/custom-garment" className="group">
          <Card hoverable className="h-full p-8 flex flex-col overflow-hidden relative">
            <div className="absolute top-0 right-0 p-8 opacity-0 group-hover:opacity-100 transform translate-x-4 group-hover:translate-x-0 transition-all duration-300">
              <MoveRight className="w-6 h-6 text-[#1A1A1A]" />
            </div>
            <div className="w-14 h-14 bg-[#F7F5F0] rounded-2xl flex items-center justify-center mb-8 text-[#1A1A1A] group-hover:bg-[#1A1A1A] group-hover:text-white transition-colors duration-300">
              <Wand2 className="w-7 h-7" />
            </div>
            <h3 className="text-2xl font-bold mb-4 text-[#1A1A1A]">AI Fashion Intelligence</h3>
            <p className="text-[#767676] text-base leading-relaxed flex-1">
              Upload standard fabric textures and transform them into photorealistic, high-end garments using localized FLUX models.
            </p>
          </Card>
        </Link>
        
        <Link href="/studio/virtual-tryon" className="group">
          <Card hoverable className="h-full p-8 flex flex-col overflow-hidden relative">
            <div className="absolute top-0 right-0 p-8 opacity-0 group-hover:opacity-100 transform translate-x-4 group-hover:translate-x-0 transition-all duration-300">
              <MoveRight className="w-6 h-6 text-[#1A1A1A]" />
            </div>
            <div className="w-14 h-14 bg-[#F7F5F0] rounded-2xl flex items-center justify-center mb-8 text-[#1A1A1A] group-hover:bg-[#1A1A1A] group-hover:text-white transition-colors duration-300">
              <Layers className="w-7 h-7" />
            </div>
            <h3 className="text-2xl font-bold mb-4 text-[#1A1A1A]">Virtual Try-On</h3>
            <p className="text-[#767676] text-base leading-relaxed flex-1">
              Seamless digital fitting. Map generated garments directly onto target personas with remarkable structural accuracy.
            </p>
          </Card>
        </Link>
        
        <Link href="/studio/semantic-analysis" className="group">
          <Card hoverable className="h-full p-8 flex flex-col relative overflow-hidden bg-gradient-to-br from-white to-[#FDFCFB]">
            <div className="absolute top-6 right-6 bg-gradient-to-r from-[#D8B4E2] to-[#B4E2C6] text-[#1A1A1A] text-[10px] uppercase font-bold px-3 py-1 rounded shadow-sm">
              Coming Soon
            </div>
            <div className="w-14 h-14 bg-[#F7F5F0] rounded-2xl flex items-center justify-center mb-8 text-[#1A1A1A] group-hover:bg-[#1A1A1A] group-hover:text-white transition-colors duration-300">
              <SearchCode className="w-7 h-7" />
            </div>
            <h3 className="text-2xl font-bold mb-4 text-[#1A1A1A]">Semantic Extraction</h3>
            <p className="text-[#767676] text-base leading-relaxed flex-1">
              Deep metadata extraction. Our Qwen2.5-VL engine will automatically tag, categorize, and understand complex fashion catalogs.
            </p>
          </Card>
        </Link>
      </section>
    </div>
  );
}
