const API_ORIGIN =
  process.env.NEXT_PUBLIC_API_ORIGIN ||
  (process.env.NEXT_PUBLIC_API_URL
    ? process.env.NEXT_PUBLIC_API_URL.replace(/\/api\/v1\/?$/, "")
    : "http://127.0.0.1:8000");

/**
 * Resolves backend-relative asset paths to absolute URLs.
 */
export function resolveMediaUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  if (url.startsWith("blob:") || url.startsWith("data:") || url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  const path = url.startsWith("/") ? url : `/${url}`;
  return `${API_ORIGIN}${path}`;
}

export { API_ORIGIN };
