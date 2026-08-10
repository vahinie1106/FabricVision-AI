export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"] as const;
export const ACCEPTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"] as const;
export const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB

export type ImageValidationResult =
  | { valid: true; file: File }
  | { valid: false; message: string };

export function validateImageFile(file: File, maxBytes = MAX_IMAGE_SIZE_BYTES): ImageValidationResult {
  const typeOk = ACCEPTED_IMAGE_TYPES.includes(file.type as (typeof ACCEPTED_IMAGE_TYPES)[number]);
  const name = file.name.toLowerCase();
  const extensionOk = ACCEPTED_IMAGE_EXTENSIONS.some((ext) => name.endsWith(ext));

  if (!typeOk && !extensionOk) {
    return {
      valid: false,
      message: "Invalid Image — Please upload a PNG, JPEG, or WEBP file.",
    };
  }

  if (file.size > maxBytes) {
    const maxMb = Math.round(maxBytes / (1024 * 1024));
    return {
      valid: false,
      message: `File too large — Maximum size is ${maxMb}MB.`,
    };
  }

  if (file.size === 0) {
    return {
      valid: false,
      message: "Invalid Image — The selected file appears to be empty.",
    };
  }

  return { valid: true, file };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
