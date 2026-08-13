"use client";

import { useCallback, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ImageDropzone } from "@/components/ui/ImageDropzone";
import { Button } from "@/components/ui/Button";
import { LiveProgress } from "@/components/feedback/LiveProgress";
import { EmptyState } from "@/components/feedback/EmptyState";
import { FriendlyError } from "@/components/feedback/FriendlyError";
import { ResultCard, DownloadCenter } from "@/components/studio/ResultCard";
import { ImageComparison } from "@/components/studio/ImageComparison";
import { HistorySidebar } from "@/components/studio/HistorySidebar";
import { useToast } from "@/hooks/useToast";
import { useLiveProgress, formatElapsed } from "@/hooks/useLiveProgress";
import { useGenerationHistory, type HistoryItem } from "@/hooks/useGenerationHistory";
import { useFluxWarmupStatus } from "@/hooks/useFluxWarmupStatus";
import { toFriendlyError, type FriendlyError as FriendlyErrorType } from "@/lib/friendlyErrors";
import { resolveMediaUrl } from "@/lib/resolveMediaUrl";
import {
  downloadFromUrl,
  downloadJson,
  downloadText,
  downloadZip,
  fetchAsBlob,
} from "@/lib/download";
import { Shirt, Sparkles, RefreshCw, Columns2 } from "lucide-react";
import type { GenerationMode } from "@/types";

export default function CustomGarmentGenerator() {
  const toast = useToast();
  const progress = useLiveProgress("garment");
  const { items: history, addItem, removeItem } = useGenerationHistory("custom-garment");
  const fluxWarmup = useFluxWarmupStatus(2000);

  const [fabricImage, setFabricImage] = useState<File | null>(null);
  const [fabricPreview, setFabricPreview] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(true);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [friendlyError, setFriendlyError] = useState<FriendlyErrorType | null>(null);

  const [gender, setGender] = useState("Women");
  const [season, setSeason] = useState("Summer");
  const [occasion, setOccasion] = useState("Casual");
  const [garmentType, setGarmentType] = useState("Dress");
  const [fit, setFit] = useState("Slim Fit");
  const [style, setStyle] = useState("Casual");
  const [fabric, setFabric] = useState("Cotton");
  const [material] = useState("Cotton");
  const [texture] = useState("Smooth");
  const [color, setColor] = useState("Match Fabric");
  const [sleeve, setSleeve] = useState("Short Sleeve");
  const [neckline, setNeckline] = useState("Round Neck");
  /** Kaggle bake-time default is Production; local Next stays Standard. */
  const [generationMode, setGenerationMode] = useState<GenerationMode>(() => {
    const raw = (process.env.NEXT_PUBLIC_DEFAULT_GENERATION_MODE || "Standard").trim();
    if (raw === "Preview" || raw === "Standard" || raw === "Production") return raw;
    return "Standard";
  });

  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [durationMs, setDurationMs] = useState<number | undefined>();

  const promptSummary = useMemo(
    () =>
      `${gender} ${color} ${fabric} ${garmentType}, ${fit}, ${style} style, ${sleeve}, ${neckline}, ${season}, ${occasion}`,
    [gender, color, fabric, garmentType, fit, style, sleeve, neckline, season, occasion]
  );

  const resolvedResult = resolveMediaUrl(resultUrl) || resultUrl;

  /** Prefer backend-measured generation time over client poll clock when available. */
  const measuredDurationMs = useMemo(() => {
    const backendSec = metadata?.generation_time_s;
    if (typeof backendSec === "number" && backendSec > 0) {
      return Math.round(backendSec * 1000);
    }
    return durationMs;
  }, [metadata, durationMs]);

  const resultResolution = useMemo(() => {
    const w = metadata?.width;
    const h = metadata?.height;
    if (typeof w === "number" && typeof h === "number") return `${w}×${h}`;
    return undefined;
  }, [metadata]);

  const onFabricSelected = useCallback(
    (file: File | null) => {
      setFabricImage(file);
      setFriendlyError(null);
      if (fabricPreview) URL.revokeObjectURL(fabricPreview);
      if (file) {
        const url = URL.createObjectURL(file);
        setFabricPreview(url);
        toast.success("Image uploaded", "Fabric texture ready for generation.");
      } else {
        setFabricPreview(null);
      }
    },
    [fabricPreview, toast]
  );

  const handleGenerate = async () => {
    if (!fabricImage) return;

    setIsGenerating(true);
    setIsDone(false);
    setResultUrl(null);
    setMetadata(null);
    setFriendlyError(null);
    setShowCompare(false);
    setActiveHistoryId(null);
    progress.start();
    toast.info("Generation started", "FLUX Kontext is preparing your garment.");

    try {
      const { GenerationService } = await import("@/services/generationService");

      console.log("[FRONTEND QUALITY DEBUG]");
      console.log(`selected_generation_mode=${generationMode}`);

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
          color: color.toLowerCase() === "match fabric" ? "match_fabric" : color,
          sleeve,
          neckline,
          generationMode,
        },
        (pct, currentStep, stage) => {
          progress.update(pct, currentStep, stage);
        }
      );

      const elapsed = progress.complete();
      const url = response.resultUrl || null;
      const meta = (response.metadata as unknown as Record<string, unknown>) || null;
      const ts = new Date().toISOString();

      setResultUrl(url);
      setMetadata(meta);
      setGeneratedAt(ts);
      setDurationMs(elapsed);
      setIsDone(true);

      const thumb = resolveMediaUrl(url) || fabricPreview || "";
      const entry = addItem({
        module: "custom-garment",
        thumbnailUrl: thumb,
        resultUrl: url || undefined,
        title: `${color} ${garmentType}`,
        model: "FLUX Kontext",
        status: "completed",
        durationMs: elapsed,
        metadata: meta || undefined,
        promptSummary,
        createdAt: ts,
      });
      setActiveHistoryId(entry.id);
      toast.success("Generation completed", "Your garment concept is ready.");
    } catch (err) {
      console.error(err);
      progress.fail();
      const fe = toFriendlyError(err);
      setFriendlyError(fe);
      toast.error(fe.title, fe.message);
    } finally {
      setIsGenerating(false);
    }
  };

  const restoreHistory = (item: HistoryItem) => {
    setActiveHistoryId(item.id);
    setResultUrl(item.resultUrl || item.thumbnailUrl);
    setMetadata((item.metadata as Record<string, unknown>) || null);
    setGeneratedAt(item.createdAt);
    setDurationMs(item.durationMs);
    setIsDone(item.status === "completed");
    setIsGenerating(false);
    setFriendlyError(null);
    setShowCompare(false);
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
        await downloadFromUrl(resolvedResult, `fabricvision-garment-${Date.now()}.png`);
        toast.success("Download ready", "PNG saved to your device.");
      },
    },
    {
      id: "json",
      label: "Download Metadata JSON",
      icon: "json" as const,
      disabled: !isDone,
      onClick: async () => {
        await downloadJson(
          {
            generatedAt,
            model: "FLUX Kontext",
            prompt: promptSummary,
            parameters: {
              gender,
              season,
              occasion,
              garmentType,
              fit,
              style,
              fabric,
              color,
              sleeve,
              neckline,
            },
            metadata,
            resultUrl,
          },
          `fabricvision-garment-meta-${Date.now()}.json`
        );
        toast.success("JSON exported", "Metadata downloaded.");
      },
    },
    {
      id: "prompt",
      label: "Download Prompt",
      icon: "prompt" as const,
      disabled: !isDone,
      onClick: async () => {
        await downloadText(promptSummary, `fabricvision-prompt-${Date.now()}.txt`);
        toast.success("Download ready", "Prompt text exported.");
      },
    },
    {
      id: "zip",
      label: "Download ZIP",
      icon: "zip" as const,
      disabled: !resolvedResult,
      onClick: async () => {
        if (!resolvedResult) return;
        const imageBlob = await fetchAsBlob(resolvedResult);
        const metaBlob = new Blob(
          [JSON.stringify({ prompt: promptSummary, metadata, generatedAt }, null, 2)],
          { type: "application/json" }
        );
        const promptBlob = new Blob([promptSummary], { type: "text/plain" });
        await downloadZip(
          [
            { name: "garment.png", blob: imageBlob },
            { name: "metadata.json", blob: metaBlob },
            { name: "prompt.txt", blob: promptBlob },
          ],
          `fabricvision-garment-${Date.now()}.zip`
        );
        toast.success("Download ready", "ZIP package exported.");
      },
    },
  ];

  return (
    <div className="w-full flex flex-col xl:flex-row gap-4 sm:gap-6 min-h-[calc(100vh-140px)] xl:h-[calc(100vh-140px)] xl:overflow-hidden">
      {/* LEFT PANEL */}
      <div className="w-full xl:w-[340px] 2xl:w-[380px] flex flex-col bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden xl:h-full">
        <div className="p-5 sm:p-6 border-b border-gray-100 bg-white z-10">
          <h2 className="text-xl font-bold flex items-center">
            <Sparkles className="w-5 h-5 mr-2 text-[#1A1A1A]" />
            Custom Garment
          </h2>
          <p className="text-xs text-[#767676] mt-1">
            Upload → Customize → Generate → Download
          </p>
          {fluxWarmup.showWarmupBanner && (
            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-950">
              <p className="font-semibold">AI engine is warming up…</p>
              <p className="mt-1 text-amber-900/90">
                FLUX initialization: {fluxWarmup.progress}%
                {fluxWarmup.currentStep ? ` — ${fluxWarmup.currentStep}` : ""}
              </p>
              {typeof fluxWarmup.loadDurationS === "number" && (
                <p className="mt-1 text-amber-800/80">
                  Elapsed {Math.round(fluxWarmup.loadDurationS)}s (real API warmup)
                </p>
              )}
            </div>
          )}
          {fluxWarmup.ready && !fluxWarmup.failed && (
            <p className="mt-3 text-xs font-medium text-emerald-800">AI engine ready</p>
          )}
          {fluxWarmup.failed && (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-950">
              <p className="font-semibold">AI engine failed to initialize</p>
              <p className="mt-1">{fluxWarmup.error || "FLUX warmup failed"}</p>
              <p className="mt-1 text-red-900/80">
                You can retry with Generate Garment (loads on demand).
              </p>
            </div>
          )}
        </div>

        <div className="p-5 sm:p-6 flex-1 space-y-8 overflow-y-auto">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">
              1. Base Fabric
            </h3>
            <ImageDropzone
              label="Upload Fabric"
              onImageSelected={onFabricSelected}
              onValidationError={(msg) => {
                toast.error("Invalid Image", msg);
              }}
            />
          </div>

          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">
              2. Identity
            </h3>
            <div className="space-y-4">
              <SelectField label="Gender" value={gender} onChange={setGender} options={["Men", "Women", "Unisex"]} />
              <SelectField label="Season" value={season} onChange={setSeason} options={["Summer", "Winter", "Spring", "Autumn", "All Season"]} />
              <SelectField label="Occasion" value={occasion} onChange={setOccasion} options={["Casual", "Formal", "Party", "Sports", "Traditional", "Business"]} />
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">
              3. Garment Configuration
            </h3>
            <div className="space-y-4">
              <SelectField
                label="Garment Type"
                value={garmentType}
                onChange={setGarmentType}
                options={["Dress", "Shirt", "Trousers", "Jacket", "Kurti", "Lehenga", "Saree", "Top", "Skirt", "Jumpsuit", "Hoodie", "Blazer"]}
              />
              <SelectField label="Fit" value={fit} onChange={setFit} options={["Slim Fit", "Regular", "Oversized", "Relaxed", "Tailored"]} />
              <SelectField
                label="Style"
                value={style}
                onChange={setStyle}
                options={["Casual", "Formal", "Avant-Garde", "Minimalist", "Vintage", "Streetwear", "Bohemian"]}
              />
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">
              4. Physical Attributes
            </h3>
            <div className="space-y-4">
              <SelectField
                label="Fabric"
                value={fabric}
                onChange={setFabric}
                options={["Cotton", "Polyester", "Wool", "Silk", "Linen", "Denim", "Leather", "Fleece", "Nylon", "Spandex", "Velvet", "Corduroy", "Satin", "Chiffon", "Lace", "Rayon", "Viscose", "Cashmere"]}
              />
              <SelectField
                label="Color"
                value={color}
                onChange={setColor}
                options={[
                  "Match Fabric",
                  "Black",
                  "White",
                  "Red",
                  "Blue",
                  "Green",
                  "Yellow",
                  "Navy Blue",
                  "Royal Blue",
                  "Maroon",
                  "Beige",
                  "Olive Green",
                  "Pastel Pink",
                  "Lavender",
                  "Cream",
                ]}
              />
              <p className="text-[11px] text-[#767676] mt-1 leading-relaxed">
                Match Fabric keeps the uploaded textile colors. Choosing Black, Blue,
                Red, etc. recolors the garment while preserving fabric texture and pattern.
              </p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">
              5. Construction
            </h3>
            <div className="space-y-4">
              <SelectField
                label="Sleeve"
                value={sleeve}
                onChange={setSleeve}
                options={["Sleeveless", "Cap Sleeve", "Short Sleeve", "Half Sleeve", "Three Quarter Sleeve", "Full Sleeve", "Puff Sleeve", "Bell Sleeve", "Bishop Sleeve"]}
              />
              <SelectField
                label="Neckline"
                value={neckline}
                onChange={setNeckline}
                options={["Round Neck", "V Neck", "U Neck", "Collar Neck", "Boat Neck", "Square Neck", "Sweetheart Neck", "Halter Neck", "High Neck", "Mandarin Collar", "Off Shoulder", "Keyhole Neck"]}
              />
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A1A] mb-4">
              6. Quality Mode
            </h3>
            <SelectField
              label="Generation Mode"
              value={generationMode}
              onChange={(v) => setGenerationMode(v as GenerationMode)}
              options={["Preview", "Standard", "Production"]}
            />
            <p className="text-[11px] text-[#767676] mt-2 leading-relaxed">
              Preview: fast structure check. Standard: default balance. Production: max practical
              detail clarity on this GPU.
            </p>
          </div>
        </div>

        <div className="p-5 sm:p-6 border-t border-gray-100 bg-[#FDFCFB] space-y-3">
          <Button
            className="w-full py-4 text-sm uppercase tracking-wide"
            onClick={handleGenerate}
            disabled={
              !fabricImage ||
              isGenerating ||
              !fluxWarmup.generateEnabledByFlux
            }
            isLoading={isGenerating}
          >
            {fluxWarmup.warming
              ? "Waiting for AI engine…"
              : isGenerating
                ? "Generating..."
                : isDone
                  ? "Generate Again"
                  : "Generate Garment"}
          </Button>
        </div>
      </div>

      {/* CENTER */}
      <div className="w-full flex-1 min-h-[420px] bg-[#F7F5F0] rounded-2xl border border-gray-200 shadow-inner relative flex flex-col overflow-hidden xl:h-full">
        <AnimatePresence mode="wait">
          {!isGenerating && !isDone && !friendlyError && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex items-center justify-center p-6"
            >
              <EmptyState
                icon={Shirt}
                title="No garment generated yet."
                description="Upload a fabric image to begin. Configure your preferences, then generate a concept with FLUX Kontext."
                className="border-0 bg-transparent shadow-none max-w-md"
              />
            </motion.div>
          )}

          {(isGenerating || (progress.active && !isDone)) && (
            <motion.div
              key="progress"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex items-center justify-center p-6"
            >
              <div className="w-full max-w-md bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
                <p className="text-xs font-bold uppercase tracking-wider text-[#767676] mb-4">
                  Live Progress
                </p>
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

          {friendlyError && !isGenerating && (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex items-center justify-center p-6"
            >
              <FriendlyError error={friendlyError} onRetry={handleGenerate} className="max-w-lg" />
            </motion.div>
          )}

          {isDone && resolvedResult && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex-1 overflow-y-auto p-4 sm:p-6"
            >
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <h3 className="text-sm font-bold uppercase tracking-wider text-[#1A1A1A]">
                  Generated Result
                </h3>
                <div className="flex gap-2">
                  {fabricPreview && (
                    <Button
                      variant="secondary"
                      className="py-2 px-3 text-xs"
                      onClick={() => setShowCompare((v) => !v)}
                    >
                      <Columns2 className="w-3.5 h-3.5 mr-1.5" />
                      {showCompare ? "Result Only" : "Before / After"}
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    className="py-2 px-3 text-xs"
                    onClick={handleGenerate}
                    disabled={!fabricImage || isGenerating}
                  >
                    <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
                    Generate Again
                  </Button>
                </div>
              </div>

              <div className="max-w-xl mx-auto">
                {showCompare && fabricPreview ? (
                  <ImageComparison
                    beforeSrc={fabricPreview}
                    afterSrc={resolveMediaUrl(resultUrl) || resultUrl || ""}
                  />
                ) : (
                  <ResultCard
                    imageUrl={resultUrl || ""}
                    title={`${color} ${garmentType}`}
                    meta={{
                      model: `FLUX.1 Kontext · ${String(metadata?.generation_mode || generationMode)}`,
                      timestamp: generatedAt || undefined,
                      durationMs: measuredDurationMs,
                      promptSummary,
                      resolution: resultResolution,
                      extra: {
                        Steps:
                          typeof metadata?.num_inference_steps === "number"
                            ? String(metadata.num_inference_steps)
                            : undefined,
                        Guidance:
                          typeof metadata?.guidance_scale === "number"
                            ? String(metadata.guidance_scale)
                            : undefined,
                      },
                    }}
                  />
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* RIGHT */}
      <div className="w-full xl:w-[300px] 2xl:w-[320px] flex flex-col gap-4 xl:h-full">
        <div className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm overflow-y-auto flex flex-col min-h-[280px]">
          <div className="p-5 border-b border-gray-100">
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#1A1A1A]">AI Insights</h2>
          </div>

          <div className="p-5 flex-1 flex flex-col space-y-6">
            {(isGenerating || isDone || progress.failed) && (
              <LiveProgress
                compact
                stages={progress.stages}
                stageIndex={progress.stageIndex}
                stageLabel={progress.stageLabel}
                percent={isDone ? 100 : progress.percent}
                elapsedMs={progress.elapsedMs}
                failed={progress.failed}
              />
            )}

            <div className={isDone ? "opacity-100" : "opacity-40"}>
              <h3 className="text-xs font-bold uppercase tracking-wider text-[#767676] mb-3 border-b border-gray-100 pb-2">
                Generated Metadata
              </h3>
              {isDone ? (
                <div className="space-y-3">
                  <MetaRow label="Category" value={String(metadata?.category || garmentType)} />
                  <MetaRow label="Fabric" value={String(metadata?.fabric || fabric)} />
                  <MetaRow label="Style Affinity" value={String(metadata?.styleAffinity || style)} />
                  <MetaRow
                    label="Mode"
                    value={String(metadata?.generation_mode || generationMode)}
                  />
                  <MetaRow label="Model" value="FLUX.1-Kontext" />
                  <MetaRow
                    label="Fabric mode"
                    value={color === "Match Fabric" ? "Match Fabric" : String(color)}
                  />
                  <MetaRow
                    label="Resolution"
                    value={resultResolution || "—"}
                  />
                  <MetaRow
                    label="Steps"
                    value={
                      typeof metadata?.num_inference_steps === "number"
                        ? String(metadata.num_inference_steps)
                        : "—"
                    }
                  />
                  <MetaRow
                    label="Generation Time"
                    value={
                      typeof measuredDurationMs === "number"
                        ? formatElapsed(measuredDurationMs)
                        : "—"
                    }
                    accent
                  />
                  <MetaRow
                    label="Confidence"
                    value={
                      typeof metadata?.confidenceScore === "number"
                        ? `${((metadata.confidenceScore as number) * 100).toFixed(0)}%`
                        : "95%"
                    }
                  />
                </div>
              ) : isGenerating || progress.active ? (
                <p className="text-xs text-[#767676]">Metadata will appear when generation completes.</p>
              ) : (
                <p className="text-xs text-[#767676]">No metadata generated.</p>
              )}
            </div>

            {isDone && <DownloadCenter actions={downloadActions} />}
          </div>
        </div>

        <div className="hidden xl:block relative h-[240px]">
          <HistorySidebar
            items={history}
            open={historyOpen}
            onToggle={() => setHistoryOpen((o) => !o)}
            onSelect={restoreHistory}
            onRemove={removeItem}
            activeId={activeHistoryId}
            className="h-full"
          />
        </div>
      </div>

      {/* Mobile history */}
      <div className="xl:hidden w-full">
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
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <div>
      <label className="block text-xs font-semibold text-[#767676] mb-1">{label}</label>
      <select
        className="w-full bg-[#F7F5F0] border-none rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-[#1A1A1A] outline-none"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

function MetaRow({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex justify-between items-center text-sm gap-3">
      <span className="text-[#767676]">{label}</span>
      <span className={`font-semibold capitalize truncate ${accent ? "text-green-600" : "text-[#1A1A1A]"}`}>
        {value}
      </span>
    </div>
  );
}
