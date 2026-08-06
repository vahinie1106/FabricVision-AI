"use client";

import { useState, useRef } from "react";
import { UploadCloud, X } from "lucide-react";
import Image from "next/image";

interface ImageDropzoneProps {
  label?: string;
  onImageSelected: (file: File | null) => void;
}

export function ImageDropzone({ label = "Upload Image", onImageSelected }: ImageDropzoneProps) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
    onImageSelected(file);
  };

  const removeImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    onImageSelected(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      className={`relative w-full h-64 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-colors cursor-pointer overflow-hidden
        ${dragActive ? "border-[#1A1A1A] bg-[#F7F5F0]" : "border-gray-200 bg-[#FDFCFB] hover:bg-[#F7F5F0]"}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
      />

      {preview ? (
        <div className="absolute inset-0 w-full h-full">
          <Image src={preview} alt="Upload preview" fill className="object-cover" />
          <button
            onClick={removeImage}
            className="absolute top-4 right-4 p-2 bg-white/80 backdrop-blur-sm rounded-full text-[#1A1A1A] hover:bg-white shadow-sm transition-all"
          >
            <X size={20} />
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center text-center p-6">
          <div className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center mb-4 text-[#1A1A1A]">
            <UploadCloud size={24} />
          </div>
          <p className="text-sm font-semibold text-[#1A1A1A] mb-1">{label}</p>
          <p className="text-xs text-[#767676]">Drag and drop or click to browse</p>
          <p className="text-[10px] text-gray-400 mt-2 uppercase tracking-wide">JPG, PNG up to 5MB</p>
        </div>
      )}
    </div>
  );
}
