/**
 * Environment-aware API + media URL resolution.
 *
 * LOCAL (Next on :3000 talking to FastAPI on :8000):
 *   http://127.0.0.1:8000/api/v1
 *
 * SAME-ORIGIN gateway (FastAPI on :8000, including Kaggle Jupyter proxy):
 *   {detectedOrConfiguredBasePath}/api/v1
 *
 * On Kaggle the public prefix is dynamic. jupyter-server-proxy under a Kaggle
 * Jupyter tunnel typically looks like:
 *   /k/<session>/proxy/proxy/8000
 * (first /proxy/ = Kaggle Jupyter tunnel, second = port mapper).
 * Never bake a hard-coded host-root "/proxy/8000"-only path.
 */

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

/** Optional build-time basePath (set by scripts/run_kaggle.py from live Jupyter). */
export function getConfiguredBasePath(): string {
  const fromEnv = (process.env.NEXT_PUBLIC_BASE_PATH || "").trim();
  if (fromEnv) {
    return stripTrailingSlash(fromEnv.startsWith("/") ? fromEnv : `/${fromEnv}`);
  }
  return "";
}

/**
 * Detect the public gateway prefix from the browser location.
 * Prefer `/k/<session>/proxy/proxy/<port>` (intentional two-layer path), then
 * `/k/<session>/proxy/<port>`, then host-root `/proxy/<port>`.
 * Do NOT strip a legitimate `/proxy/proxy/<port>` segment.
 */
export function detectRuntimeBasePath(): string {
  if (typeof window === "undefined") return "";
  const path = window.location.pathname || "";

  const doubleProxy = path.match(/^(.*?\/proxy\/proxy\/\d+)(?:\/|$)/);
  if (doubleProxy) {
    return stripTrailingSlash(doubleProxy[1]);
  }

  const singleProxy = path.match(/^(.*?\/proxy\/\d+)(?:\/|$)/);
  if (singleProxy) {
    return stripTrailingSlash(singleProxy[1]);
  }
  return "";
}

/**
 * Runtime base path for public deployments behind Jupyter-style proxies.
 * On kaggle/jupyter-proxy hosts, prefer live URL detection over a stale build value.
 */
export function getDeploymentBasePath(): string {
  if (typeof window !== "undefined") {
    const host = window.location.hostname || "";
    const runtime = detectRuntimeBasePath();
    if (
      runtime &&
      (host.includes("kaggle.net") ||
        host.includes("jupyter-proxy") ||
        host.includes("googleapis.com") ||
        process.env.NEXT_PUBLIC_USE_SAME_ORIGIN === "true")
    ) {
      return runtime;
    }
    if (runtime) return runtime;
  }

  return getConfiguredBasePath();
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
    // Relative override such as "/api/v1" — if it already includes the full
    // public prefix, use as-is; otherwise prefix with the live base path.
    const path = configured.replace(/^\.\//, "/");
    const normalized = path.startsWith("/") ? path : `/${path}`;
    if (basePath && normalized.startsWith(`${basePath}/`)) {
      return stripTrailingSlash(normalized);
    }
    if (basePath && normalized === "/api/v1") {
      return stripTrailingSlash(`${basePath}/api/v1`);
    }
    // Configured absolute-from-root API under a dynamic public prefix.
    if (basePath && normalized.startsWith("/api/")) {
      return stripTrailingSlash(`${basePath}${normalized}`);
    }
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
    // If a stale "/proxy/8000" was baked but runtime path differs, prefer runtime.
    const runtime = getDeploymentBasePath();
    if (
      runtime &&
      configuredOrigin.replace(/\/+$/, "") === "/proxy/8000" &&
      runtime !== "/proxy/8000"
    ) {
      return runtime;
    }
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

  return getDeploymentBasePath();
}

/** Eager default for modules that read a constant; prefer resolveApiBaseUrl() in new code. */
export const API_BASE_URL = resolveApiBaseUrl();

export const API_ORIGIN = resolveApiOrigin();
