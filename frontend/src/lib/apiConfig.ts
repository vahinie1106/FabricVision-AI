/**
 * Environment-aware API + media URL resolution.
 *
 * LOCAL (Next on :3000 talking to FastAPI on :8000):
 *   http://127.0.0.1:8000/api/v1
 *
 * SAME-ORIGIN gateway (FastAPI proxies UI on :8000, e.g. Kaggle):
 *   {basePath}/api/v1   — basePath is "" locally, or "/proxy/8000" on Kaggle
 *
 * Override anytime with NEXT_PUBLIC_API_URL / NEXT_PUBLIC_API_ORIGIN.
 */

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

/** Next.js basePath baked at build time (Kaggle: /proxy/8000). */
export function getConfiguredBasePath(): string {
  const fromEnv = (process.env.NEXT_PUBLIC_BASE_PATH || "").trim();
  if (fromEnv) {
    return stripTrailingSlash(fromEnv.startsWith("/") ? fromEnv : `/${fromEnv}`);
  }
  return "";
}

/**
 * Runtime base path for public deployments behind Jupyter-style proxies.
 * Prefers build-time NEXT_PUBLIC_BASE_PATH; otherwise detects /proxy/<port>.
 */
export function getDeploymentBasePath(): string {
  const configured = getConfiguredBasePath();
  if (configured) return configured;
  if (typeof window !== "undefined") {
    const match = window.location.pathname.match(/^(\/proxy\/\d+)/);
    if (match) return match[1];
  }
  return "";
}

function isAbsoluteHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function isLocalNextDevHost(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname, port } = window.location;
  return (
    (hostname === "localhost" || hostname === "127.0.0.1") &&
    (port === "3000" || port === "")
  );
}

/**
 * Resolve the API root including `/api/v1` (no trailing slash).
 */
export function resolveApiBaseUrl(): string {
  const configured = (process.env.NEXT_PUBLIC_API_URL || "").trim();
  if (configured && isAbsoluteHttpUrl(configured)) {
    return stripTrailingSlash(configured);
  }

  // SSR / Node without window: keep local absolute API unless same-origin build.
  if (typeof window === "undefined") {
    if (
      process.env.NEXT_PUBLIC_USE_SAME_ORIGIN === "true" ||
      getConfiguredBasePath()
    ) {
      return stripTrailingSlash(`${getConfiguredBasePath()}/api/v1`);
    }
    if (configured) {
      const path = configured.replace(/^\.\//, "/");
      const normalized = path.startsWith("/") ? path : `/${path}`;
      return stripTrailingSlash(normalized);
    }
    return "http://127.0.0.1:8000/api/v1";
  }

  // Browser on local Next.dev → dedicated backend (unless same-origin forced).
  if (
    isLocalNextDevHost() &&
    process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true" &&
    !configured
  ) {
    return "http://127.0.0.1:8000/api/v1";
  }

  const basePath = getDeploymentBasePath();
  if (configured) {
    // Relative override, e.g. "/api/v1" or "./api/v1"
    const path = configured.replace(/^\.\//, "/");
    const normalized = path.startsWith("/") ? path : `/${path}`;
    return stripTrailingSlash(`${basePath}${normalized}`);
  }

  return stripTrailingSlash(`${basePath}/api/v1`);
}

/**
 * Origin used to resolve `/outputs/...` media paths.
 * Empty string ⇒ same-origin relative URLs (required for Kaggle proxy).
 */
export function resolveApiOrigin(): string {
  const configuredOrigin = (process.env.NEXT_PUBLIC_API_ORIGIN || "").trim();
  if (configuredOrigin) {
    return stripTrailingSlash(configuredOrigin);
  }

  const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "").trim();
  if (apiUrl && isAbsoluteHttpUrl(apiUrl)) {
    return stripTrailingSlash(apiUrl.replace(/\/api\/v1\/?$/, ""));
  }

  if (
    typeof window !== "undefined" &&
    isLocalNextDevHost() &&
    process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true" &&
    !apiUrl
  ) {
    return "http://127.0.0.1:8000";
  }

  // Same-origin / gateway: keep media paths relative (prepend basePath only).
  return getDeploymentBasePath();
}

/** Eager default for modules that read a constant; prefer resolveApiBaseUrl() in new code. */
export const API_BASE_URL = resolveApiBaseUrl();

export const API_ORIGIN = resolveApiOrigin();
