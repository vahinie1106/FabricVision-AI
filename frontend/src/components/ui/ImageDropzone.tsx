"use client";

import { useCallback, useRef, useState } from "react";
import { ImagePlus, RefreshCw, UploadCloud, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ACCEPTED_IMAGE_TYPES,
  formatBytes,
  MAX_IMAGE_SIZE_BYTES,
  validateImageFile,
} from "@/lib/imageValidation";
import { cn } from "@/lib/cn";

interface ImageDropzoneProps {
  label?: string;
  onImageSelected: (file: File | null) => void;
  onValidationError?: (message: string) => void;
  className?: string;
  compact?: boolean;
  /** Controlled preview URL (optional) — when provided, parent owns the preview */
  previewUrl?: string | null;
}

export function ImageDropzone({
  label = "Upload Image",
  onImageSelected,
  onValidationError,
  className,
  compact = false,
  previewUrl,
}: ImageDropzoneProps) {
  const [dragActive, setDragActive] = useState(false);
  const [internalPreview, setInternalPreview] = useState<string | null>(null);
  const [fileMeta, setFileMeta] = useState<{ name: string; size: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const preview = previewUrl !== undefined ? previewUrl : internalPreview;

  const clear = useCallback(() => {
    if (internalPreview) URL.revokeObjectURL(internalPreview);
    setInternalPreview(null);
    setFileMeta(null);
    setError(null);
    onImageSelected(null);
    if (inputRef.current) inputRef.current.value = "";
  }, [internalPreview, onImageSelected]);

  const processFile = useCallback(
    (file: File) => {
      const result = validateImageFile(file);
      if (!result.valid) {
        setError(result.message);
        onValidationError?.(result.message);
        return;
      }

      setError(null);
      if (internalPreview) URL.revokeObjectURL(internalPreview);
      const url = URL.createObjectURL(file);
      setInternalPreview(url);
      setFileMeta({ name: file.name, size: formatBytes(file.size) });
      onImageSelected(file);
    },
    [internalPreview, onImageSelected, onValidationError]
  );

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0]);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) processFile(e.target.files[0]);
  };

  const openPicker = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    inputRef.current?.click();
  };

  return (
    <div className={cn("w-full", className)}>
      <div
        className={cn(
          "relative w-full border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-colors overflow-hidden",
          compact ? "h-48" : "h-64",
          dragActive ? "border-[#1A1A1A] bg-[#F7F5F0]" : "border-gray-200 bg-[#FDFCFB] hover:bg-[#F7F5F0]",
          preview ? "cursor-default" : "cursor-pointer"
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !preview && openPicker()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (!preview && (e.key === "Enter" || e.key === " ")) openPicker();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_IMAGE_TYPES.join(",")}
          onChange={handleChange}
          className="hidden"
        />

        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key="preview"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 w-full h-full"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt="Upload preview" className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent opacity-0 hover:opacity-100 transition-opacity">
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between gap-2">
                  <div className="min-w-0 text-white">
                    {fileMeta && (
                      <>
                        <p className="text-xs font-medium truncate">{fileMeta.name}</p>
                        <p className="text-[10px] opacity-80">{fileMeta.size}</p>
                      </>
                    )}
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={openPicker}
                      className="p-2 bg-white/90 backdrop-blur-sm rounded-full text-[#1A1A1A] hover:bg-white shadow-sm transition-all"
                      title="Replace image"
                      aria-label="Replace image"
                    >
                      <RefreshCw size={16} />
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        clear();
                      }}
                      className="p-2 bg-white/90 backdrop-blur-sm rounded-full text-[#1A1A1A] hover:bg-white shadow-sm transition-all"
                      title="Remove image"
                      aria-label="Remove image"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>
              </div>
              {/* Always-visible controls for touch */}
              <div className="absolute top-3 right-3 flex gap-2 md:opacity-100">
                <button
                  type="button"
                  onClick={openPicker}
                  className="p-2 bg-white/90 backdrop-blur-sm rounded-full text-[#1A1A1A] hover:bg-white shadow-sm transition-all"
                  title="Replace image"
                  aria-label="Replace image"
                >
                  <ImagePlus size={16} />
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    clear();
                  }}
                  className="p-2 bg-white/90 backdrop-blur-sm rounded-full text-[#1A1A1A] hover:bg-white shadow-sm transition-all"
                  title="Remove image"
                  aria-label="Remove image"
                >
                  <X size={16} />
                </button>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center text-center p-6"
            >
              <div className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-4 text-[#1A1A1A]">
                <UploadCloud size={24} />
              </div>
              <p className="text-sm font-semibold text-[#1A1A1A] mb-1">{label}</p>
              <p className="text-xs text-[#767676]">Drag and drop or click to browse</p>
              <p className="text-[10px] text-gray-400 mt-2 uppercase tracking-wide">
                PNG, JPEG, WEBP · up to {Math.round(MAX_IMAGE_SIZE_BYTES / (1024 * 1024))}MB
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {error && (
        <motion.p
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 text-xs text-red-600"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
}
