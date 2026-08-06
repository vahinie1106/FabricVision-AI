"use client";

import { useState } from "react";
import { ImageDropzone } from "@/components/ui/ImageDropzone";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/feedback/Skeleton";
import { ProgressTimeline, TimelineStep } from "@/components/feedback/ProgressTimeline";
import { Sparkles, Download, Wand2, RefreshCw } from "lucide-react";

export default function CustomGarmentGenerator() {
  const [fabricImage, setFabricImage] = useState<File | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDone, setIsDone] = useState(false);

  // Identity Fields
  const [gender, setGender] = useState("Women");
  const [season, setSeason] = useState("Summer");
  const [occasion, setOccasion] = useState("Casual");

  // Garment Configuration Fields
  const [garmentType, setGarmentType] = useState("Dress");
  const [fit, setFit] = useState("Slim Fit");
  const [style, setStyle] = useState("Casual");

  // Physical Fields (UI exposes Fabric & Color; Material & Texture remain hidden state values)
  const [fabric, setFabric] = useState("Cotton");
  const [material] = useState("Cotton"); // Hidden from UI MVP
  const [texture] = useState("Smooth");  // Hidden from UI MVP
  const [color, setColor] = useState("White");

  // Construction Fields
  const [sleeve, setSleeve] = useState("Short Sleeve");
  const [neckline, setNeckline] = useState("Round Neck");

  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<any>(null);

  const [steps, setSteps] = useState<TimelineStep[]>([
    { id: "01", label: "Queued", status: "waiting" },
    { id: "02", label: "Processing", status: "waiting" },
    { id: "03", label: "Completed", status: "waiting" },
  ]);

  const handleGenerate = async () => {
    if (!fabricImage) return;

    setIsGenerating(true);
    setIsDone(false);
    setResultUrl(null);
    setMetadata(null);

    setSteps([
      { id: "01", label: "Queued", status: "processing" },
      { id: "02", label: "Processing", status: "waiting" },
      { id: "03", label: "Completed", status: "waiting" },
    ]);

    try {
      const { GenerationService } = await import("@/services/generationService");

      const response = await GenerationService.generateCustomGarment(
        {
          fabricImage,
          garmentType,
          fit,
          style,
          gender,
          season,
          occasion,
          fabric,
          material,
          texture,
          color,
          sleeve,
          neckline,
        },
        (progress, currentStep) => {
          if (progress > 0 && progress < 100) {
            setSteps([
              { id: "01", label: "Queued", status: "completed" },
              { id: "02", label: currentStep || "Processing", status: "processing" },
              { id: "03", label: "Completed", status: "waiting" },
            ]);
          }
        }
      );

      setSteps([
        { id: "01", label: "Queued", status: "completed" },
        { id: "02", label: "Processing", status: "completed" },
        { id: "03", label: "Completed", status: "completed" },
      ]);

      setResultUrl(response.resultUrl || null);
      setMetadata(response.metadata || null);
      setIsDone(true);
    } catch (err) {
      console.error(err);
      setSteps([
        { id: "01", label: "Queued", status: "failed" },
        { id: "02", label: "Processing", status: "failed" },
        { id: "03", label: "Completed", status: "failed" },
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="w-full flex flex-col lg:flex-row h-[calc(100vh-140px)] gap-6 overflow-hidden">

      {/* LEFT PANEL - Controls */}
      <div className="w-full lg:w-1/3 min-w-[340px] flex flex-col h-full bg-white rounded-2xl border border-gray-200 shadow-sm overflow-y-auto">
        <div className="p-6 border-b border-gray-100 sticky top-0 bg-white z-10">
          <h2 className="text-xl font-bold flex items-center">
            <Wand2 className="w-5 h-5 mr-2 text-[#1A1A1A]" />
            Custom Garment
          </h2>
          <p className="text-xs text-[#767676] mt-1">Configure complete fashion metadata parameters.</p>
        </div>

        <div className="p-6 flex-1 space-y-8">
          {/* WORKFLOW STEP 1: Base Fabric */}
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">1. Base Fabric</h3>
            <ImageDropzone label="Upload Texture" onImageSelected={setFabricImage} />
          </div>

          {/* WORKFLOW STEP 2: Identity */}
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">2. Identity</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Gender</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                >
                  <option>Men</option>
                  <option>Women</option>
                  <option>Unisex</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Season</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={season}
                  onChange={(e) => setSeason(e.target.value)}
                >
                  <option>Summer</option>
                  <option>Winter</option>
                  <option>Spring</option>
                  <option>Autumn</option>
                  <option>All Season</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Occasion</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={occasion}
                  onChange={(e) => setOccasion(e.target.value)}
                >
                  <option>Casual</option>
                  <option>Formal</option>
                  <option>Party</option>
                  <option>Sports</option>
                  <option>Traditional</option>
                  <option>Business</option>
                </select>
              </div>
            </div>
          </div>

          {/* WORKFLOW STEP 3: Garment Configuration */}
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">3. Garment Configuration</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Garment Type</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={garmentType}
                  onChange={(e) => setGarmentType(e.target.value)}
                >
                  <option>Dress</option>
                  <option>Shirt</option>
                  <option>Trousers</option>
                  <option>Jacket</option>
                  <option>Kurti</option>
                  <option>Lehenga</option>
                  <option>Saree</option>
                  <option>Top</option>
                  <option>Skirt</option>
                  <option>Jumpsuit</option>
                  <option>Hoodie</option>
                  <option>Blazer</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Fit</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={fit}
                  onChange={(e) => setFit(e.target.value)}
                >
                  <option>Slim Fit</option>
                  <option>Regular</option>
                  <option>Oversized</option>
                  <option>Relaxed</option>
                  <option>Tailored</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Style</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={style}
                  onChange={(e) => setStyle(e.target.value)}
                >
                  <option>Casual</option>
                  <option>Formal</option>
                  <option>Avant-Garde</option>
                  <option>Minimalist</option>
                  <option>Vintage</option>
                  <option>Streetwear</option>
                  <option>Bohemian</option>
                </select>
              </div>
            </div>
          </div>

          {/* WORKFLOW STEP 4: Physical Attributes */}
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">4. Physical Attributes</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Fabric</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={fabric}
                  onChange={(e) => setFabric(e.target.value)}
                >
                  <option>Cotton</option>
                  <option>Polyester</option>
                  <option>Wool</option>
                  <option>Silk</option>
                  <option>Linen</option>
                  <option>Denim</option>
                  <option>Leather</option>
                  <option>Fleece</option>
                  <option>Nylon</option>
                  <option>Spandex</option>
                  <option>Velvet</option>
                  <option>Corduroy</option>
                  <option>Satin</option>
                  <option>Chiffon</option>
                  <option>Lace</option>
                  <option>Rayon</option>
                  <option>Viscose</option>
                  <option>Cashmere</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Color</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={color}
                  onChange={(e) => setColor(e.target.value)}
                >
                  <option>Black</option>
                  <option>White</option>
                  <option>Red</option>
                  <option>Blue</option>
                  <option>Green</option>
                  <option>Yellow</option>
                  <option>Navy Blue</option>
                  <option>Royal Blue</option>
                  <option>Maroon</option>
                  <option>Beige</option>
                  <option>Olive Green</option>
                  <option>Pastel Pink</option>
                  <option>Lavender</option>
                  <option>Cream</option>
                </select>
              </div>
            </div>
          </div>

          {/* WORKFLOW STEP 5: Construction */}
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">5. Construction</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Sleeve</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={sleeve}
                  onChange={(e) => setSleeve(e.target.value)}
                >
                  <option>Sleeveless</option>
                  <option>Cap Sleeve</option>
                  <option>Short Sleeve</option>
                  <option>Half Sleeve</option>
                  <option>Three Quarter Sleeve</option>
                  <option>Full Sleeve</option>
                  <option>Puff Sleeve</option>
                  <option>Bell Sleeve</option>
                  <option>Bishop Sleeve</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#767676] mb-1">Neckline</label>
                <select
                  className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
                  value={neckline}
                  onChange={(e) => setNeckline(e.target.value)}
                >
                  <option>Round Neck</option>
                  <option>V Neck</option>
                  <option>U Neck</option>
                  <option>Collar Neck</option>
                  <option>Boat Neck</option>
                  <option>Square Neck</option>
                  <option>Sweetheart Neck</option>
                  <option>Halter Neck</option>
                  <option>High Neck</option>
                  <option>Mandarin Collar</option>
                  <option>Off Shoulder</option>
                  <option>Keyhole Neck</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-gray-100 bg-[#FDFCFB] sticky bottom-0 z-10">
          <Button
            className="w-full py-4 text-sm uppercase tracking-wide"
            onClick={handleGenerate}
            disabled={!fabricImage || isGenerating}
            isLoading={isGenerating}
          >
            {isGenerating ? "Synthesizing..." : "Generate AI Concept"}
          </Button>
        </div>
      </div>

      {/* CENTER PANEL - Workspace Canvas */}
      <div className="w-full lg:flex-1 h-full bg-[#F7F5F0] rounded-2xl border border-gray-200 shadow-inner relative flex flex-col items-center justify-center overflow-hidden">

        {!isGenerating && !isDone && (
          <div className="flex flex-col items-center text-center p-8 opacity-50">
            <Sparkles className="w-16 h-16 text-gray-300 mb-4" />
            <p className="text-gray-400 font-medium">Your canvas is empty.<br />Upload a fabric and configure your settings to begin.</p>
          </div>
        )}

        {isGenerating && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/50 backdrop-blur-sm z-10">
            <div className="flex flex-col items-center">
              <Skeleton className="w-[400px] h-[500px] max-w-full rounded-2xl shadow-lg" />
              <p className="mt-6 font-semibold text-[#1A1A1A] animate-pulse uppercase tracking-wider">Processing with FLUX Kontext...</p>
            </div>
          </div>
        )}

        {isDone && (
          <div className="relative w-full h-full p-8 flex items-center justify-center">
            <div className="w-full max-w-[500px] aspect-[4/5] bg-gradient-to-br from-gray-200 to-gray-50 rounded-2xl shadow-2xl overflow-hidden relative group">
              {resultUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={resultUrl.startsWith("http") ? resultUrl : `http://127.0.0.1:8000${resultUrl}`}
                  alt="Generated Garment"
                  className="w-full h-full object-cover rounded-2xl"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-gray-400 font-medium z-0">
                  No output returned
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* RIGHT PANEL - AI Insights */}
      <div className="w-full lg:w-[320px] flex flex-col h-full bg-white rounded-2xl border border-gray-200 shadow-sm overflow-y-auto">
        <div className="p-6 border-b border-gray-100">
          <h2 className="text-sm font-bold uppercase tracking-wider text-[#1A1A1A]">AI Insights</h2>
        </div>

        <div className="p-6 flex-1 flex flex-col space-y-8">

          {/* Progress Section */}
          <div className={isGenerating || isDone ? "opacity-100" : "opacity-30 pointer-events-none"}>
            <ProgressTimeline steps={steps} />
          </div>

          {/* Metadata Section */}
          <div className={`mt-auto ${isDone ? "opacity-100" : "opacity-30 pointer-events-none"}`}>
            <h3 className="text-xs font-bold uppercase tracking-wider text-[#767676] mb-4 border-b border-gray-100 pb-2">Generated Metadata</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-sm">
                <span className="text-[#767676]">Category</span>
                <span className="font-semibold text-[#1A1A1A] capitalize">{metadata?.category || garmentType}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-[#767676]">Fabric</span>
                <span className="font-semibold text-[#1A1A1A] capitalize">{metadata?.fabric || fabric}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-[#767676]">Style Affinity</span>
                <span className="font-semibold text-[#1A1A1A] capitalize">{metadata?.styleAffinity || style}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-[#767676]">Confidence</span>
                <span className="font-semibold text-green-600">{metadata?.confidenceScore ? `${(metadata.confidenceScore * 100).toFixed(0)}%` : "95%"}</span>
              </div>
            </div>

            <div className="mt-8 space-y-3">
              <Button className="w-full" disabled={!isDone}>
                <Download className="w-4 h-4 mr-2" /> Export Asset
              </Button>
              <Button variant="secondary" className="w-full" disabled={!isDone}>
                <RefreshCw className="w-4 h-4 mr-2" /> Send to Try-On
              </Button>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
