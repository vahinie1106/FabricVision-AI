"use client";

import { useCallback, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  BrainCircuit,
  Tags,
  ListTree,
  FileJson,
  RefreshCw,
  ScanSearch,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ImageDropzone } from "@/components/ui/ImageDropzone";
import { LiveProgress } from "@/components/feedback/LiveProgress";
import { EmptyState } from "@/components/feedback/EmptyState";
import { FriendlyError } from "@/components/feedback/FriendlyError";
import { DownloadCenter } from "@/components/studio/ResultCard";
import { HistorySidebar } from "@/components/studio/HistorySidebar";
import { useToast } from "@/hooks/useToast";
import { useLiveProgress } from "@/hooks/useLiveProgress";
import { useGenerationHistory, type HistoryItem } from "@/hooks/useGenerationHistory";
import { toFriendlyError, type FriendlyError as FriendlyErrorType } from "@/lib/friendlyErrors";
import { downloadJson, downloadText } from "@/lib/download";

function buildTxtReport(metadata: Record<string, unknown>): string {
  const lines: string[] = ["FabricVision-AI Semantic Analysis Report", "========================================", ""];
  const walk = (obj: unknown, indent = 0) => {
    if (obj == null) return;
    if (typeof obj !== "object") {
      lines.push(`${"  ".repeat(indent)}${String(obj)}`);
      return;
    }
    if (Array.isArray(obj)) {
      obj.forEach((v) => walk(v, indent + 1));
      return;
    }
    Object.entries(obj as Record<string, unknown>).forEach(([k, v]) => {
      if (v != null && typeof v === "object") {
        lines.push(`${"  ".repeat(indent)}${k}:`);
        walk(v, indent + 1);
      } else {
        lines.push(`${"  ".repeat(indent)}${k}: ${String(v)}`);
      }
    });
  };
  walk(metadata);
  return lines.join("\n");
}

export default function SemanticAnalysis() {
  const toast = useToast();
  const progress = useLiveProgress("analysis");
  const resetProgress = progress.reset;
  const { items: history, addItem, removeItem } = useGenerationHistory("semantic-analysis");

  const [garmentImage, setGarmentImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [durationMs, setDurationMs] = useState<number | undefined>();
  const [friendlyError, setFriendlyError] = useState<FriendlyErrorType | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);

  const onImageSelected = useCallback(
    (file: File | null) => {
      setGarmentImage(file);
      setMetadata(null);
      setFriendlyError(null);
      setActiveHistoryId(null);
      resetProgress();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (file) {
        setPreviewUrl(URL.createObjectURL(file));
        toast.success("Image uploaded", "Ready to analyze with Qwen2.5-VL.");
      } else {
        setPreviewUrl(null);
      }
    },
    [previewUrl, toast, resetProgress]
  );

  const handleAnalyze = async () => {
    if (!garmentImage) return;

    setIsProcessing(true);
    setMetadata(null);
    setFriendlyError(null);
    progress.start();
    toast.info("Generation started", "Semantic analysis in progress.");

    try {
      const { AnalysisService } = await import("@/services/analysisService");
      const response = await AnalysisService.analyzeGarment({ garmentImage });
      const actualMetadata =
        (response.metadata as { metadata?: Record<string, unknown> })?.metadata ||
        (response.metadata as Record<string, unknown>) ||
        (response as unknown as Record<string, unknown>);

      const elapsed = progress.complete();
      const ts = new Date().toISOString();
      setMetadata(actualMetadata);
      setGeneratedAt(ts);
      setDurationMs(elapsed);

      const title =
        String(
          (actualMetadata as { garment_identity?: { name?: string } })?.garment_identity?.name ||
            (actualMetadata as { classification?: { category?: string } })?.classification?.category ||
            "Analysis Result"
        );

      const entry = addItem({
        module: "semantic-analysis",
        thumbnailUrl: previewUrl || "",
        title,
        model: "Qwen2.5-VL",
        status: "completed",
        durationMs: elapsed,
        createdAt: ts,
        metadata: actualMetadata,
      });
      setActiveHistoryId(entry.id);
      toast.success("Generation completed", "Structured metadata is ready.");
    } catch (err) {
      console.error(err);
      progress.fail();
      const fe = toFriendlyError(err);
      setFriendlyError(fe);
      toast.error(fe.title, fe.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const restoreHistory = (item: HistoryItem) => {
    setActiveHistoryId(item.id);
    setMetadata((item.metadata as Record<string, unknown>) || null);
    setGeneratedAt(item.createdAt);
    setDurationMs(item.durationMs);
    if (item.thumbnailUrl) setPreviewUrl(item.thumbnailUrl);
    setFriendlyError(null);
    setIsProcessing(false);
    toast.info("History restored", item.title);
  };

  const downloadActions = [
    {
      id: "json",
      label: "Download JSON",
      icon: "json" as const,
      disabled: !metadata,
      onClick: async () => {
        await downloadJson(
          { generatedAt, model: "Qwen2.5-VL", durationMs, metadata },
          `fabricvision-analysis-${Date.now()}.json`
        );
        toast.success("JSON exported", "Metadata downloaded.");
      },
    },
    {
      id: "txt",
      label: "Download TXT Report",
      icon: "txt" as const,
      disabled: !metadata,
      onClick: async () => {
        await downloadText(buildTxtReport(metadata!), `fabricvision-analysis-${Date.now()}.txt`);
        toast.success("Download ready", "Text report exported.");
      },
    },
  ];

  const classification = metadata?.classification as
    | { category?: string; subcategory?: string }
    | undefined;
  const physical = metadata?.physical_attributes as { material?: string } | undefined;
  const shape = metadata?.shape_and_fit as { fit?: string } | undefined;
  const style = metadata?.style as { occasion?: string } | undefined;
  const identity = metadata?.garment_identity as { name?: string } | undefined;

  return (
    <div className="w-full max-w-[1200px] mx-auto min-h-[calc(100vh-140px)] flex flex-col pt-6 sm:pt-12 pb-16 sm:pb-24 relative">
      <div className="text-center max-w-3xl mx-auto mb-10 sm:mb-16 px-2">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 sm:mb-6 text-[#1A1A1A]">
          Fashion Intelligence Engine
        </h1>
        <p className="text-base sm:text-lg text-[#767676] leading-relaxed">
          Upload → Preview → Analyze → Structured Metadata → Download JSON
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-start">
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-tr from-[#FDFCFB] to-[#D8B4E2]/10 rounded-[3rem] -z-10 transform rotate-2 scale-105" />
          <Card className="p-5 sm:p-8 bg-white/90 backdrop-blur-xl border border-white/50 shadow-[0_20px_40px_rgba(0,0,0,0.08)] relative overflow-hidden flex flex-col gap-6">
            <ImageDropzone
              label="Upload Image"
              onImageSelected={onImageSelected}
              onValidationError={(msg) => toast.error("Invalid Image", msg)}
            />

            <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-4 min-h-[220px]">
              <div className="bg-gray-100 rounded-2xl relative overflow-hidden flex items-center justify-center min-h-[200px]">
                {previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrl} alt="Uploaded Garment" className="w-full h-full object-cover" />
                ) : (
                  <EmptyState
                    icon={ScanSearch}
                    title="Upload a fabric image to begin."
                    description="PNG, JPEG, or WEBP"
                    compact
                    className="border-0 bg-transparent min-h-[180px] p-4"
                  />
                )}
                {isProcessing && (
                  <motion.div
                    className="absolute top-0 left-0 right-0 h-1 bg-[#D8B4E2] shadow-[0_0_15px_#D8B4E2]"
                    animate={{ top: ["0%", "100%", "0%"] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  />
                )}
              </div>

              <div className="flex flex-col space-y-3 max-h-[320px] overflow-y-auto pr-1">
                <div className="flex items-center space-x-2 sticky top-0 bg-white/90 backdrop-blur-sm py-1 z-10">
                  <BrainCircuit className="w-4 h-4 text-[#B4E2C6]" />
                  <span className="text-xs font-bold uppercase text-[#1A1A1A]">Extracted Data</span>
                </div>

                <AnimatePresence mode="wait">
                  {!metadata && !isProcessing && !friendlyError && (
                    <motion.div key="empty-meta" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <p className="text-sm text-gray-400 mt-2">No metadata generated.</p>
                    </motion.div>
                  )}

                  {isProcessing && (
                    <motion.div
                      key="progress"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="mt-2"
                    >
                      <LiveProgress
                        stages={progress.stages}
                        stageIndex={progress.stageIndex}
                        stageLabel={progress.stageLabel}
                        percent={progress.percent}
                        elapsedMs={progress.elapsedMs}
                      />
                    </motion.div>
                  )}

                  {friendlyError && !isProcessing && (
                    <FriendlyError error={friendlyError} onRetry={handleAnalyze} />
                  )}

                  {metadata && (
                    <motion.div
                      key="meta"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-3 mt-1 pb-2"
                    >
                      {identity?.name && (
                        <div className="bg-[#F7F5F0] rounded-xl p-3">
                          <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Identity</div>
                          <div className="text-sm font-medium capitalize">{identity.name}</div>
                        </div>
                      )}

                      <div className="bg-[#F7F5F0] rounded-xl p-3">
                        <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Classification</div>
                        <div className="text-sm capitalize">
                          {classification?.category || "—"}
                          {classification?.subcategory ? ` — ${classification.subcategory}` : ""}
                        </div>
                      </div>

                      <div className="bg-[#F7F5F0] rounded-xl p-3">
                        <div className="text-[10px] text-gray-500 uppercase font-bold mb-2">Attributes</div>
                        <div className="flex flex-wrap gap-2">
                          <span className="px-2 py-1 bg-white text-[10px] rounded border border-gray-100 text-gray-500 capitalize">
                            {physical?.material || "Unknown Material"}
                          </span>
                          <span className="px-2 py-1 bg-white text-[10px] rounded border border-gray-100 text-gray-500 capitalize">
                            {shape?.fit || "Unknown Fit"}
                          </span>
                          <span className="px-2 py-1 bg-white text-[10px] rounded border border-gray-100 text-gray-500 capitalize">
                            {style?.occasion || "Unknown Style"}
                          </span>
                        </div>
                      </div>

                      <div className="bg-[#F7F5F0] rounded-xl p-3">
                        <div className="text-[10px] text-gray-500 uppercase font-bold mb-1 flex items-center gap-1">
                          <FileJson className="w-3 h-3" /> Structured Metadata
                        </div>
                        <pre className="text-[9px] text-gray-600 overflow-auto max-h-40 whitespace-pre-wrap break-all">
                          {JSON.stringify(metadata, null, 2)}
                        </pre>
                      </div>

                      {(generatedAt || durationMs != null) && (
                        <div className="text-[10px] text-[#767676] space-y-0.5">
                          {generatedAt && <p>Generated: {new Date(generatedAt).toLocaleString()}</p>}
                          {durationMs != null && <p>Duration: {(durationMs / 1000).toFixed(1)}s · Model: Qwen2.5-VL</p>}
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-3">
              <Button
                className="flex-1"
                onClick={handleAnalyze}
                disabled={!garmentImage || isProcessing}
                isLoading={isProcessing}
              >
                {isProcessing ? "Analyzing..." : metadata ? "Analyze Again" : "Analyze"}
              </Button>
              {metadata && (
                <Button
                  variant="secondary"
                  onClick={() => {
                    setMetadata(null);
                    progress.reset();
                  }}
                >
                  <RefreshCw className="w-4 h-4" />
                </Button>
              )}
            </div>

            {metadata && <DownloadCenter actions={downloadActions} />}
          </Card>
        </div>

        <div className="flex flex-col space-y-8">
          <div className="flex items-start">
            <div className="w-12 h-12 rounded-xl bg-[#F7F5F0] flex items-center justify-center mr-4 flex-shrink-0">
              <Tags className="w-6 h-6 text-[#1A1A1A]" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-[#1A1A1A] mb-2">Automated Classification</h3>
              <p className="text-[#767676] text-sm leading-relaxed">
                Instantly categorize complex fashion items with human-level accuracy. Identifies silhouette, cut, and
                fit automatically.
              </p>
            </div>
          </div>

          <div className="flex items-start">
            <div className="w-12 h-12 rounded-xl bg-[#F7F5F0] flex items-center justify-center mr-4 flex-shrink-0">
              <ListTree className="w-6 h-6 text-[#1A1A1A]" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-[#1A1A1A] mb-2">Intelligent Taxonomy</h3>
              <p className="text-[#767676] text-sm leading-relaxed">
                Extracts structured JSON metadata directly from pixels, bridging the gap between visual design and
                catalog management.
              </p>
            </div>
          </div>

          <div className="pt-6 border-t border-gray-100">
            <p className="text-sm font-semibold text-[#1A1A1A] mb-4">
              View our API documentation to integrate directly into your workflows.
            </p>
            <div className="flex space-x-3">
              <Button variant="secondary">API Documentation</Button>
            </div>
          </div>

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
