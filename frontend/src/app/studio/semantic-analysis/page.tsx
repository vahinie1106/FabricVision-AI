"use client";

import { useState } from "react";
import { Sparkles, BrainCircuit, Tags, ListTree, ChevronRight } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ImageDropzone } from "@/components/ui/ImageDropzone";
import Link from "next/link";
import { motion } from "framer-motion";

export default function SemanticAnalysis() {
  const [garmentImage, setGarmentImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [metadata, setMetadata] = useState<any>(null);

  const handleAnalyze = async (file: File | null) => {
    if (!file) return;
    setGarmentImage(file);
    setIsProcessing(true);
    setMetadata(null);

    try {
      const { AnalysisService } = await import("@/services/analysisService");
      const response = await AnalysisService.analyzeGarment({ garmentImage: file });
      // The backend pipeline returns metadata nested inside response.metadata.metadata
      const actualMetadata = response.metadata?.metadata || response.metadata || response;
      setMetadata(actualMetadata);
    } catch (err) {
      console.error(err);
      alert("Failed to analyze image");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="w-full max-w-[1200px] mx-auto min-h-[calc(100vh-140px)] flex flex-col pt-12 pb-24">
      
      <div className="text-center max-w-3xl mx-auto mb-16">
        <h1 className="text-4xl md:text-5xl font-bold mb-6 text-[#1A1A1A]">Fashion Intelligence Engine</h1>
        <p className="text-lg text-[#767676] leading-relaxed">
          Powered by Qwen2.5-VL, the Semantic Analysis module automatically reverse-engineers garment DNA from any fashion image.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
        
        {/* Left: Upload and Interactive Preview */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-tr from-[#FDFCFB] to-[#D8B4E2]/10 rounded-[3rem] -z-10 transform rotate-2 scale-105" />
          <Card className="p-8 bg-white/90 backdrop-blur-xl border border-white/50 shadow-[0_20px_40px_rgba(0,0,0,0.08)] relative overflow-hidden flex flex-col gap-6">
            
            <ImageDropzone label="Upload Garment Image" onImageSelected={handleAnalyze} />

            <div className="w-full flex space-x-6 min-h-[250px]">
               <div className="w-1/2 bg-gray-100 rounded-2xl relative overflow-hidden flex items-center justify-center">
                 {garmentImage ? (
                   // eslint-disable-next-line @next/next/no-img-element
                   <img src={URL.createObjectURL(garmentImage)} alt="Uploaded Garment" className="w-full h-full object-cover" />
                 ) : (
                   <>
                     <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/black-linen.png')] mix-blend-overlay opacity-30" />
                     <span className="text-gray-400 font-medium z-10 text-sm">Example Image</span>
                   </>
                 )}
                 {isProcessing && (
                   <motion.div 
                     className="absolute top-0 left-0 right-0 h-1 bg-[#D8B4E2] shadow-[0_0_15px_#D8B4E2]"
                     animate={{ top: ["0%", "100%", "0%"] }}
                     transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                   />
                 )}
               </div>
               
               <div className="w-1/2 flex flex-col space-y-4 max-h-[300px] overflow-y-auto pr-2">
                  <div className="flex items-center space-x-2 sticky top-0 bg-white/90 backdrop-blur-sm py-1 z-10">
                    <BrainCircuit className="w-4 h-4 text-[#B4E2C6]" />
                    <span className="text-xs font-bold uppercase text-[#1A1A1A]">Extracted Data</span>
                  </div>
                  
                  {!metadata && !isProcessing && (
                    <div className="text-sm text-gray-400 mt-4">Upload an image to see semantic data.</div>
                  )}

                  {isProcessing && (
                    <div className="space-y-4 mt-2 opacity-50 animate-pulse">
                      <div className="bg-[#F7F5F0] rounded-xl p-3 space-y-2">
                        <div className="h-2 bg-gray-200 rounded w-full"></div>
                        <div className="h-2 bg-gray-200 rounded w-4/5"></div>
                      </div>
                      <div className="bg-[#F7F5F0] rounded-xl p-3 space-y-2">
                        <div className="h-2 bg-gray-200 rounded w-3/4"></div>
                        <div className="h-2 bg-gray-200 rounded w-5/6"></div>
                      </div>
                    </div>
                  )}

                  {metadata && (
                    <div className="space-y-3 mt-2 pb-4">
                      {metadata.garment_identity?.name && (
                        <div className="bg-[#F7F5F0] rounded-xl p-3">
                          <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Identity</div>
                          <div className="text-sm font-medium capitalize">{metadata.garment_identity.name}</div>
                        </div>
                      )}
                      
                      <div className="bg-[#F7F5F0] rounded-xl p-3">
                        <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Classification</div>
                        <div className="text-sm capitalize">{metadata.classification?.category} - {metadata.classification?.subcategory}</div>
                      </div>

                      <div className="bg-[#F7F5F0] rounded-xl p-3">
                        <div className="text-[10px] text-gray-500 uppercase font-bold mb-2">Attributes</div>
                        <div className="flex flex-wrap gap-2">
                          <span className="px-2 py-1 bg-white text-[10px] rounded border border-gray-100 text-gray-500 capitalize">{metadata.physical_attributes?.material || "Unknown Material"}</span>
                          <span className="px-2 py-1 bg-white text-[10px] rounded border border-gray-100 text-gray-500 capitalize">{metadata.shape_and_fit?.fit || "Unknown Fit"}</span>
                          <span className="px-2 py-1 bg-white text-[10px] rounded border border-gray-100 text-gray-500 capitalize">{metadata.style?.occasion || "Unknown Style"}</span>
                        </div>
                      </div>
                      
                      <div className="bg-[#F7F5F0] rounded-xl p-3 break-all">
                         <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Raw JSON</div>
                         <pre className="text-[9px] text-gray-600 overflow-hidden">{JSON.stringify(metadata, null, 2).slice(0, 200)}...</pre>
                      </div>
                    </div>
                  )}
               </div>
            </div>
          </Card>
        </div>

        {/* Right: Feature Explanations */}
        <div className="flex flex-col space-y-8">
           
           <div className="flex items-start">
             <div className="w-12 h-12 rounded-xl bg-[#F7F5F0] flex items-center justify-center mr-4 flex-shrink-0">
               <Tags className="w-6 h-6 text-[#1A1A1A]" />
             </div>
             <div>
               <h3 className="text-xl font-bold text-[#1A1A1A] mb-2">Automated Classification</h3>
               <p className="text-[#767676] text-sm leading-relaxed">Instantly categorize complex fashion items with human-level accuracy. Identifies silhouette, cut, and fit automatically.</p>
             </div>
           </div>

           <div className="flex items-start">
             <div className="w-12 h-12 rounded-xl bg-[#F7F5F0] flex items-center justify-center mr-4 flex-shrink-0">
               <ListTree className="w-6 h-6 text-[#1A1A1A]" />
             </div>
             <div>
               <h3 className="text-xl font-bold text-[#1A1A1A] mb-2">Intelligent Taxonomy</h3>
               <p className="text-[#767676] text-sm leading-relaxed">Extracts structured JSON metadata directly from pixels, bridging the gap between visual design and catalog management.</p>
             </div>
           </div>
           
           <div className="pt-6 border-t border-gray-100">
              <p className="text-sm font-semibold text-[#1A1A1A] mb-4">View our API documentation to integrate directly into your workflows.</p>
              <div className="flex space-x-3">
                <Button variant="secondary">API Documentation</Button>
              </div>
           </div>

        </div>

      </div>
    </div>
  );
}
