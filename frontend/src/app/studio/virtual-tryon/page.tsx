"use client";

import { useState } from "react";
import { ImageDropzone } from "@/components/ui/ImageDropzone";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/feedback/Skeleton";
import { CheckCircle2, Download, SlidersHorizontal, Image as ImageIcon, Shirt, User } from "lucide-react";

export default function VirtualTryOn() {
  const [garmentImage, setGarmentImage] = useState<File | null>(null);
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDone, setIsDone] = useState(false);
  
  const [fitPreference, setFitPreference] = useState("Maintain Source Fit");
  const [backgroundAction, setBackgroundAction] = useState("Keep Original");
  
  const [activeStep, setActiveStep] = useState(1);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  const handleTryOn = async () => {
    if (!garmentImage || !personImage) return;

    setIsProcessing(true);
    setIsDone(false);
    setResultUrl(null);

    try {
      const { GenerationService } = await import("@/services/generationService");
      const response = await GenerationService.executeVirtualTryOn({
        garmentImage,
        personImage,
        fitPreference,
        backgroundAction
      });

      setResultUrl(response.resultUrl || null);
      setIsDone(true);
    } catch (err) {
      console.error(err);
      alert("Failed to generate try-on");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="w-full flex flex-col lg:flex-row gap-8 max-w-[1400px] mx-auto min-h-[calc(100vh-140px)]">
      
      {/* LEFT: Wizard Steps */}
      <div className="w-full lg:w-1/3 flex flex-col gap-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Virtual Try-On</h1>
          <p className="text-[#767676] text-sm">High-fidelity digital fitting powered by CatVTON.</p>
        </div>

        <div className="flex-1 space-y-6">
          {/* Step 1 */}
          <Card className={`p-6 transition-all duration-300 ${activeStep === 1 ? 'ring-2 ring-[#1A1A1A]' : 'opacity-60'}`} onClick={() => setActiveStep(1)}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold flex items-center text-[#1A1A1A]">
                <div className="w-8 h-8 rounded-full bg-[#1A1A1A] text-white flex items-center justify-center mr-3 text-sm">1</div>
                Source Garment
              </h3>
              {garmentImage && <CheckCircle2 className="w-5 h-5 text-green-500" />}
            </div>
            {activeStep === 1 && (
              <div className="mt-4">
                <ImageDropzone label="Upload Garment" onImageSelected={(f) => { setGarmentImage(f); if(f) setActiveStep(2); }} />
              </div>
            )}
          </Card>

          {/* Step 2 */}
          <Card className={`p-6 transition-all duration-300 ${activeStep === 2 ? 'ring-2 ring-[#1A1A1A]' : 'opacity-60'}`} onClick={() => setActiveStep(2)}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold flex items-center text-[#1A1A1A]">
                <div className="w-8 h-8 rounded-full bg-[#1A1A1A] text-white flex items-center justify-center mr-3 text-sm">2</div>
                Target Persona
              </h3>
              {personImage && <CheckCircle2 className="w-5 h-5 text-green-500" />}
            </div>
            {activeStep === 2 && (
              <div className="mt-4">
                <ImageDropzone label="Upload Model/Person" onImageSelected={(f) => { setPersonImage(f); if(f) setActiveStep(3); }} />
              </div>
            )}
          </Card>

          {/* Step 3 */}
          <Card className={`p-6 transition-all duration-300 ${activeStep === 3 ? 'ring-2 ring-[#1A1A1A]' : 'opacity-60'}`} onClick={() => setActiveStep(3)}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold flex items-center text-[#1A1A1A]">
                <div className="w-8 h-8 rounded-full bg-[#1A1A1A] text-white flex items-center justify-center mr-3 text-sm">3</div>
                Adjustments
              </h3>
            </div>
            {activeStep === 3 && (
              <div className="space-y-4 mt-2">
                <div>
                  <label className="block text-xs font-semibold text-[#767676] mb-1">Fit Preference</label>
                  <select 
                    className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                    value={fitPreference}
                    onChange={(e) => setFitPreference(e.target.value)}
                  >
                    <option>Maintain Source Fit</option>
                    <option>Adapt to Persona Body</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-[#767676] mb-1">Background Action</label>
                  <select 
                    className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                    value={backgroundAction}
                    onChange={(e) => setBackgroundAction(e.target.value)}
                  >
                    <option>Keep Original</option>
                    <option>Remove Background</option>
                    <option>Studio Backdrop</option>
                  </select>
                </div>
              </div>
            )}
          </Card>
        </div>

        <Button 
          className="w-full py-4 text-base" 
          onClick={handleTryOn}
          disabled={!garmentImage || !personImage || isProcessing || isDone}
          isLoading={isProcessing}
        >
          {isProcessing ? "Mapping Garment..." : "Generate Try-On"}
        </Button>
      </div>

      {/* RIGHT: Results Workspace */}
      <div className="w-full lg:w-2/3 bg-[#F7F5F0] rounded-2xl border border-gray-200 p-8 flex flex-col items-center justify-center relative overflow-hidden">
        
        {!isProcessing && !isDone && (
          <div className="flex flex-col items-center text-center opacity-40">
            <SlidersHorizontal className="w-16 h-16 mb-4 text-[#1A1A1A]" />
            <p className="text-xl font-bold text-[#1A1A1A]">Awaiting Inputs</p>
            <p className="text-sm mt-2 max-w-sm">Upload a garment and a target persona on the left to begin the virtual fitting process.</p>
          </div>
        )}

        {isProcessing && (
          <div className="flex flex-col items-center w-full max-w-md">
            <Skeleton className="w-full aspect-[3/4] rounded-2xl shadow-xl" />
            <div className="mt-8 flex items-center space-x-3 bg-white px-6 py-3 rounded-full shadow-sm">
              <div className="w-4 h-4 border-2 border-[#1A1A1A] border-t-transparent rounded-full animate-spin" />
              <span className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A]">CatVTON Geometry Mapping...</span>
            </div>
          </div>
        )}

        {isDone && (
          <div className="w-full h-full flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-[#1A1A1A]">Result</h2>
              <div className="flex space-x-3">
                <Button variant="secondary"><Download className="w-4 h-4 mr-2"/> Export</Button>
                <Button onClick={() => {setIsDone(false); setActiveStep(1); setGarmentImage(null); setPersonImage(null);}}>Reset</Button>
              </div>
            </div>
            
            <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6 min-h-[500px]">
              
              {/* Inputs Summary */}
              <div className="flex flex-col gap-6">
                <div className="flex-1 bg-white rounded-2xl border border-gray-100 p-4 flex flex-col">
                  <div className="flex items-center text-xs font-bold uppercase text-[#767676] mb-3"><Shirt className="w-4 h-4 mr-2" /> Source</div>
                  <div className="flex-1 bg-gray-100 rounded-xl flex items-center justify-center">Image Placeholder</div>
                </div>
                <div className="flex-1 bg-white rounded-2xl border border-gray-100 p-4 flex flex-col">
                  <div className="flex items-center text-xs font-bold uppercase text-[#767676] mb-3"><User className="w-4 h-4 mr-2" /> Persona</div>
                  <div className="flex-1 bg-gray-100 rounded-xl flex items-center justify-center">Image Placeholder</div>
                </div>
              </div>

              {/* Final Output */}
              <div className="bg-white rounded-2xl border border-gray-100 p-4 flex flex-col shadow-sm group relative overflow-hidden">
                <div className="flex items-center text-xs font-bold uppercase text-[#D8B4E2] mb-3"><ImageIcon className="w-4 h-4 mr-2" /> Final Composition</div>
                <div className="flex-1 bg-gradient-to-br from-gray-200 to-gray-50 rounded-xl flex items-center justify-center relative overflow-hidden">
                  {resultUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={"http://127.0.0.1:8000" + resultUrl} alt="Try-On Result" className="w-full h-full object-cover z-20" />
                  ) : (
                    <>
                      <span className="text-gray-400 font-medium z-10">High-Res Try-On (CatVTON)</span>
                      <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/black-linen.png')] mix-blend-overlay opacity-20" />
                    </>
                  )}
                </div>
              </div>

            </div>
          </div>
        )}

      </div>

    </div>
  );
}
