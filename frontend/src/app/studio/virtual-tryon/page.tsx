"use client";

import { useCallback, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ImageDropzone } from "@/components/ui/ImageDropzone";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { LiveProgress } from "@/components/feedback/LiveProgress";
import { EmptyState } from "@/components/feedback/EmptyState";
import { FriendlyError } from "@/components/feedback/FriendlyError";
import { ResultCard, DownloadCenter } from "@/components/studio/ResultCard";
import { HistorySidebar } from "@/components/studio/HistorySidebar";
import { useToast } from "@/hooks/useToast";
import { useLiveProgress } from "@/hooks/useLiveProgress";
import { useGenerationHistory, type HistoryItem } from "@/hooks/useGenerationHistory";
import { toFriendlyError, type FriendlyError as FriendlyErrorType } from "@/lib/friendlyErrors";
import { resolveMediaUrl } from "@/lib/resolveMediaUrl";
import { downloadFromUrl, downloadZip, fetchAsBlob } from "@/lib/download";
import {
  CheckCircle2,
  Image as ImageIcon,
  Shirt,
  User,
  SlidersHorizontal,
  RefreshCw,
} from "lucide-react";

export default function VirtualTryOn() {
  const toast = useToast();
  const progress = useLiveProgress("tryon");
  const { items: history, addItem, removeItem } = useGenerationHistory("virtual-tryon");

  const [garmentImage, setGarmentImage] = useState<File | null>(null);
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [garmentPreview, setGarmentPreview] = useState<string | null>(null);
  const [personPreview, setPersonPreview] = useState<string | null>(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [fitPreference, setFitPreference] = useState("Maintain Source Fit");
  const [backgroundAction, setBackgroundAction] = useState("Keep Original");
  const [activeStep, setActiveStep] = useState(1);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [durationMs, setDurationMs] = useState<number | undefined>();
  const [friendlyError, setFriendlyError] = useState<FriendlyErrorType | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [tryonMeta, setTryonMeta] = useState<Record<string, unknown> | null>(null);

  const resolvedResult = resolveMediaUrl(resultUrl) || resultUrl;

  const onGarment = useCallback(
    (file: File | null) => {
      setGarmentImage(file);
      if (garmentPreview) URL.revokeObjectURL(garmentPreview);
      if (file) {
        setGarmentPreview(URL.createObjectURL(file));
        setActiveStep(3);
        toast.success("Image uploaded", "Source garment ready.");
      } else {
        setGarmentPreview(null);
      }
    },
    [garmentPreview, toast]
  );

  const onPerson = useCallback(
    (file: File | null) => {
      setPersonImage(file);
      if (personPreview) URL.revokeObjectURL(personPreview);
      if (file) {
        setPersonPreview(URL.createObjectURL(file));
        setActiveStep(3);
        toast.success("Image uploaded", "Target persona ready.");
      } else {
        setPersonPreview(null);
      }
    },
    [personPreview, toast]
  );

  const handleTryOn = async () => {
    if (!garmentImage || !personImage) return;

    setIsProcessing(true);
    setIsDone(false);
    setResultUrl(null);
    setFriendlyError(null);
    setActiveHistoryId(null);
    progress.start();
    toast.info("Generation started", "CatVTON is mapping the garment.");

    try {
      const { GenerationService } = await import("@/services/generationService");
      const response = await GenerationService.executeVirtualTryOn(
        {
          garmentImage,
          personImage,
          fitPreference,
          backgroundAction,
        },
        (pct, step) => progress.update(pct, step)
      );

      const meta = response.metadata || {};
      if (!meta.was_real_catvton_used || meta.was_fallback_used) {
        throw Object.assign(
          new Error("Virtual try-on did not use real CatVTON inference."),
          { errorType: "CATVTON_FALLBACK", metadata: meta }
        );
      }

      const elapsed = progress.complete();
      const url = response.resultUrl || null;
      const ts = new Date().toISOString();

      setResultUrl(url);
      setGeneratedAt(ts);
      setDurationMs(elapsed);
      setTryonMeta(meta);
      setIsDone(true);

      if (meta.mask_quality_warning || meta.mask_source === "grabcut") {
        toast.info(
          "Real CatVTON inference",
          `Mask source: ${String(meta.mask_source)}. AutoMasker/DensePose unavailable — GrabCut cloth-region mask used.`
        );
      } else {
        toast.success("Real CatVTON inference", "Try-on result is ready.");
      }

      const thumb = resolveMediaUrl(url) || personPreview || garmentPreview || "";
      const entry = addItem({
        module: "virtual-tryon",
        thumbnailUrl: thumb,
        resultUrl: url || undefined,
        title: "Virtual Try-On",
        model: "CatVTON",
        status: "completed",
        durationMs: elapsed,
        createdAt: ts,
        promptSummary: `Real CatVTON · mask:${String(meta.mask_source || "unknown")} · ${fitPreference}`,
      });
      setActiveHistoryId(entry.id);
    } catch (err) {
      console.error(err);
      progress.fail();
      setIsDone(false);
      setResultUrl(null);
      setTryonMeta(null);
      const fe = toFriendlyError(err);
      setFriendlyError(fe);
      toast.error(fe.title, fe.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetAll = () => {
    setIsDone(false);
    setResultUrl(null);
    setActiveStep(1);
    setGarmentImage(null);
    setPersonImage(null);
    if (garmentPreview) URL.revokeObjectURL(garmentPreview);
    if (personPreview) URL.revokeObjectURL(personPreview);
    setGarmentPreview(null);
    setPersonPreview(null);
    setFriendlyError(null);
    setTryonMeta(null);
    progress.reset();
    setActiveHistoryId(null);
  };

  const restoreHistory = (item: HistoryItem) => {
    setActiveHistoryId(item.id);
    setResultUrl(item.resultUrl || item.thumbnailUrl);
    setGeneratedAt(item.createdAt);
    setDurationMs(item.durationMs);
    setIsDone(item.status === "completed");
    setIsProcessing(false);
    setFriendlyError(null);
    toast.info("History restored", item.title);
  };

  const downloadActions = [
    {
      id: "png",
      label: "Download PNG",
      icon: "png" as const,
      disabled: !resolvedResult,
      onClick: async () => {
        if (!resolvedResult) return;
        await downloadFromUrl(resolvedResult, `fabricvision-tryon-${Date.now()}.png`);
        toast.success("Download ready", "PNG saved to your device.");
      },
    },
    {
      id: "zip",
      label: "Download ZIP",
      icon: "zip" as const,
      disabled: !resolvedResult,
      onClick: async () => {
        if (!resolvedResult) return;
        const files: Array<{ name: string; blob: Blob }> = [
          { name: "tryon-result.png", blob: await fetchAsBlob(resolvedResult) },
        ];
        if (garmentPreview) {
          files.push({ name: "source-garment.png", blob: await fetchAsBlob(garmentPreview) });
        }
        if (personPreview) {
          files.push({ name: "persona.png", blob: await fetchAsBlob(personPreview) });
        }
        await downloadZip(files, `fabricvision-tryon-${Date.now()}.zip`);
        toast.success("Download ready", "ZIP package exported.");
      },
    },
  ];

  return (
    <div className="w-full flex flex-col lg:flex-row gap-6 lg:gap-8 max-w-[1400px] mx-auto min-h-[calc(100vh-140px)]">
      {/* LEFT: Wizard */}
      <div className="w-full lg:w-[380px] xl:w-1/3 flex flex-col gap-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Virtual Try-On</h1>
          <p className="text-[#767676] text-sm">
            Upload person → garment → preview → generate → download
          </p>
        </div>

        <div className="flex-1 space-y-4">
          <Card
            className={`p-5 sm:p-6 transition-all duration-300 ${activeStep === 1 ? "ring-2 ring-[#1A1A1A]" : "opacity-70"}`}
            onClick={() => setActiveStep(1)}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-bold flex items-center text-[#1A1A1A]">
                <div className="w-8 h-8 rounded-full bg-[#1A1A1A] text-white flex items-center justify-center mr-3 text-sm">
                  1
                </div>
                Upload Person
              </h3>
              {personImage && <CheckCircle2 className="w-5 h-5 text-green-500" />}
            </div>
            {/* Keep existing step labels but align workflow: person first in UX copy; garment stays step 1 in original - 
                Spec says Upload Person then Upload Garment. Reorder UI accordingly while keeping API the same. */}
            {activeStep === 1 && (
              <div className="mt-4" onClick={(e) => e.stopPropagation()}>
                <ImageDropzone
                  label="Upload Person"
                  onImageSelected={onPerson}
                  onValidationError={(msg) => toast.error("Invalid Image", msg)}
                  compact
                />
              </div>
            )}
          </Card>

          <Card
            className={`p-5 sm:p-6 transition-all duration-300 ${activeStep === 2 ? "ring-2 ring-[#1A1A1A]" : "opacity-70"}`}
            onClick={() => setActiveStep(2)}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-bold flex items-center text-[#1A1A1A]">
                <div className="w-8 h-8 rounded-full bg-[#1A1A1A] text-white flex items-center justify-center mr-3 text-sm">
                  2
                </div>
                Upload Garment
              </h3>
              {garmentImage && <CheckCircle2 className="w-5 h-5 text-green-500" />}
            </div>
            {activeStep === 2 && (
              <div className="mt-4" onClick={(e) => e.stopPropagation()}>
                <ImageDropzone
                  label="Upload Garment"
                  onImageSelected={onGarment}
                  onValidationError={(msg) => toast.error("Invalid Image", msg)}
                  compact
                />
              </div>
            )}
          </Card>

          <Card
            className={`p-5 sm:p-6 transition-all duration-300 ${activeStep === 3 ? "ring-2 ring-[#1A1A1A]" : "opacity-70"}`}
            onClick={() => setActiveStep(3)}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-bold flex items-center text-[#1A1A1A]">
                <div className="w-8 h-8 rounded-full bg-[#1A1A1A] text-white flex items-center justify-center mr-3 text-sm">
                  3
                </div>
                Preview & Adjust
              </h3>
            </div>
            {activeStep === 3 && (
              <div className="space-y-4 mt-2" onClick={(e) => e.stopPropagation()}>
                {(personPreview || garmentPreview) && (
                  <div className="grid grid-cols-2 gap-3 mb-2">
                    {personPreview && (
                      <div className="rounded-xl overflow-hidden aspect-square bg-[#F7F5F0]">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={personPreview} alt="Person" className="w-full h-full object-cover" />
                      </div>
                    )}
                    {garmentPreview && (
                      <div className="rounded-xl overflow-hidden aspect-square bg-[#F7F5F0]">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={garmentPreview} alt="Garment" className="w-full h-full object-cover" />
                      </div>
                    )}
                  </div>
                )}
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

        <div className="flex gap-3">
          <Button
            className="flex-1 py-4 text-base"
            onClick={handleTryOn}
            disabled={!garmentImage || !personImage || isProcessing}
            isLoading={isProcessing}
          >
            {isProcessing ? "Generating..." : isDone ? "Generate Again" : "Generate"}
          </Button>
          {isDone && (
            <Button variant="secondary" className="py-4 px-4" onClick={resetAll}>
              <RefreshCw className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>

      {/* RIGHT: Results */}
      <div className="w-full lg:flex-1 bg-[#F7F5F0] rounded-2xl border border-gray-200 p-5 sm:p-8 flex flex-col relative overflow-hidden min-h-[480px]">
        <AnimatePresence mode="wait">
          {!isProcessing && !isDone && !friendlyError && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex items-center justify-center"
            >
              <EmptyState
                icon={SlidersHorizontal}
                title="Upload a person and garment to begin."
                description="Follow the steps on the left to preview inputs, then generate a high-fidelity CatVTON composition."
                className="border-0 bg-transparent max-w-md"
              />
            </motion.div>
          )}

          {isProcessing && (
            <motion.div
              key="progress"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex items-center justify-center"
            >
              <div className="w-full max-w-md bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
                <LiveProgress
                  stages={progress.stages}
                  stageIndex={progress.stageIndex}
                  stageLabel={progress.stageLabel}
                  percent={progress.percent}
                  elapsedMs={progress.elapsedMs}
                  failed={progress.failed}
                />
              </div>
            </motion.div>
          )}

          {friendlyError && !isProcessing && (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex-1 flex items-center justify-center"
            >
              <FriendlyError error={friendlyError} onRetry={handleTryOn} className="max-w-lg" />
            </motion.div>
          )}

          {isDone && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full h-full flex flex-col gap-6 overflow-y-auto"
            >
              <div className="flex flex-wrap justify-between items-center gap-3">
                <h2 className="text-2xl font-bold text-[#1A1A1A]">Result</h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="flex flex-col gap-4">
                  <div className="bg-white rounded-2xl border border-gray-100 p-4 flex flex-col min-h-[160px]">
                    <div className="flex items-center text-xs font-bold uppercase text-[#767676] mb-3">
                      <User className="w-4 h-4 mr-2" /> Person
                    </div>
                    <div className="flex-1 rounded-xl overflow-hidden bg-gray-100 min-h-[120px]">
                      {personPreview ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={personPreview} alt="Person" className="w-full h-full object-cover" />
                      ) : (
                        <div className="h-full flex items-center justify-center text-xs text-gray-400">No preview</div>
                      )}
                    </div>
                  </div>
                  <div className="bg-white rounded-2xl border border-gray-100 p-4 flex flex-col min-h-[160px]">
                    <div className="flex items-center text-xs font-bold uppercase text-[#767676] mb-3">
                      <Shirt className="w-4 h-4 mr-2" /> Garment
                    </div>
                    <div className="flex-1 rounded-xl overflow-hidden bg-gray-100 min-h-[120px]">
                      {garmentPreview ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={garmentPreview} alt="Garment" className="w-full h-full object-cover" />
                      ) : (
                        <div className="h-full flex items-center justify-center text-xs text-gray-400">No preview</div>
                      )}
                    </div>
                  </div>
                  <DownloadCenter actions={downloadActions} />
                </div>

                <div>
                  {resolvedResult ? (
                    <ResultCard
                      imageUrl={resultUrl || ""}
                      title="Final Composition"
                      meta={{
                        model: tryonMeta?.was_real_catvton_used
                          ? "CatVTON (real inference)"
                          : "CatVTON",
                        timestamp: generatedAt || undefined,
                        durationMs,
                        promptSummary: [
                          tryonMeta?.was_real_catvton_used ? "Real CatVTON inference" : null,
                          tryonMeta?.mask_source
                            ? `mask:${String(tryonMeta.mask_source)}`
                            : null,
                          fitPreference,
                          backgroundAction,
                        ]
                          .filter(Boolean)
                          .join(" · "),
                        resolution: Array.isArray(tryonMeta?.resolution)
                          ? `${(tryonMeta!.resolution as number[])[0]}×${(tryonMeta!.resolution as number[])[1]}`
                          : "512×512",
                        extra: {
                          backend: tryonMeta?.inference_backend
                            ? String(tryonMeta.inference_backend)
                            : undefined,
                          steps: tryonMeta?.num_inference_steps
                            ? String(tryonMeta.num_inference_steps)
                            : undefined,
                        },
                      }}
                    >
                      {tryonMeta?.mask_source === "grabcut" && (
                        <p className="text-xs text-amber-800 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
                          Cloth-region GrabCut mask used (AutoMasker/DensePose not available). Result
                          is real CatVTON diffusion, not a paste blend — quality may still trail
                          official AutoMasker masks.
                        </p>
                      )}
                    </ResultCard>
                  ) : (
                    <div className="bg-white rounded-2xl border border-gray-100 p-4 flex flex-col min-h-[400px]">
                      <div className="flex items-center text-xs font-bold uppercase text-[#D8B4E2] mb-3">
                        <ImageIcon className="w-4 h-4 mr-2" /> Final Composition
                      </div>
                      <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
                        No output returned
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="absolute top-4 right-4">
          <HistorySidebar
            items={history}
            open={historyOpen}
            onToggle={() => setHistoryOpen((o) => !o)}
            onSelect={restoreHistory}
            onRemove={removeItem}
            activeId={activeHistoryId}
          />
        </div>
      </div>
    </div>
  );
}
