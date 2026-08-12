import { resolveApiOrigin } from "@/lib/apiConfig";

/**
 * Resolves backend-relative asset paths to absolute or same-origin URLs.
 *
 * Backend already returns paths like `/outputs/...`. On Kaggle/gateway deployments
 * those must stay same-origin (optionally under `/proxy/8000`) so the Jupyter
 * proxy can reach them. Local Next.dev prepends http://127.0.0.1:8000.
 *
 * Idempotent: if ``url`` is already prefixed with the deployment base path /
 * API origin, it is returned unchanged. This matters because ResultCard,
 * downloads, and the studio pages may each call resolveMediaUrl.
 */
export function resolveMediaUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  if (
    url.startsWith("blob:") ||
    url.startsWith("data:") ||
    url.startsWith("http://") ||
    url.startsWith("https://")
  ) {
    return url;
  }
  const path = url.startsWith("/") ? url : `/${url}`;
  const origin = resolveApiOrigin();
  if (!origin) {
    return path;
  }
  // origin may itself be a path prefix (e.g. /k/<session>/proxy/proxy/8000)
  if (origin.startsWith("/")) {
    const base = origin.replace(/\/+$/, "");
    if (path === base || path.startsWith(`${base}/`)) {
      return path;
    }
    return `${base}${path}`;
  }
  return `${origin.replace(/\/+$/, "")}${path}`;
}

export { resolveApiOrigin as API_ORIGIN } from "@/lib/apiConfig";
