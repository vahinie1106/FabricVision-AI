/**
 * Environment-aware API + media URL resolution.
 *
 * LOCAL (Next on :3000 talking to FastAPI on :8000):
 *   http://127.0.0.1:8000/api/v1
 *
 * SAME-ORIGIN gateway (FastAPI on :8000, including Kaggle Jupyter proxy):
 *   {detectedOrConfiguredBasePath}/api/v1
 *
 * On Kaggle the public prefix is dynamic:
 *   /k/<session>/proxy/proxy/8000
 * Never bake host-root "/proxy/8000" alone.
 * Never call http://127.0.0.1:8000 from a jupyter-proxy browser session.
 */

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function isAbsoluteHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function isLoopbackAbsoluteUrl(value: string): boolean {
  try {
    const u = new URL(value);
    return u.hostname === "localhost" || u.hostname === "127.0.0.1";
  } catch {
    return false;
  }
}

function isProxyDeploymentHost(hostname: string): boolean {
  const host = (hostname || "").toLowerCase();
  return (
    host.includes("kaggle.net") ||
    host.includes("kaggleusercontent.com") ||
    host.includes("jupyter-proxy") ||
    host.includes("googleapis.com")
  );
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
 * Prefer `/k/<session>/proxy/proxy/<port>`, then `/k/<session>/proxy/<port>`,
 * then host-root `/proxy/<port>`.
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
      (isProxyDeploymentHost(host) ||
        process.env.NEXT_PUBLIC_USE_SAME_ORIGIN === "true")
    ) {
      return runtime;
    }
    if (runtime) return runtime;
  }

  return getConfiguredBasePath();
}

function isLocalNextDevHost(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname, port } = window.location;
  return (
    (hostname === "localhost" || hostname === "127.0.0.1") &&
    (port === "3000" || port === "")
  );
}

function sameOriginApiRoot(basePath: string): string {
  return stripTrailingSlash(`${basePath}/api/v1`);
}

/**
 * Resolve the API root including `/api/v1` (no trailing slash).
 */
export function resolveApiBaseUrl(): string {
  const rawConfigured = (process.env.NEXT_PUBLIC_API_URL || "").trim();
  const onProxyBrowser =
    typeof window !== "undefined" && isProxyDeploymentHost(window.location.hostname);

  // frontend/.env.local often has http://127.0.0.1:8000/api/v1 for local Next.dev.
  // That absolute loopback URL must NEVER win inside a public Kaggle proxy tab —
  // the browser would hang trying to reach the user's own machine.
  const configured =
    onProxyBrowser && rawConfigured && isAbsoluteHttpUrl(rawConfigured) && isLoopbackAbsoluteUrl(rawConfigured)
      ? ""
      : rawConfigured;

  if (configured && isAbsoluteHttpUrl(configured) && !isLoopbackAbsoluteUrl(configured)) {
    return stripTrailingSlash(configured);
  }

  // Absolute loopback is only valid for local Next.dev SSR/client against :8000.
  if (configured && isAbsoluteHttpUrl(configured) && isLoopbackAbsoluteUrl(configured)) {
    if (typeof window === "undefined") {
      // SSR during local next start — OK
      return stripTrailingSlash(configured);
    }
    if (isLocalNextDevHost() && process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true") {
      return stripTrailingSlash(configured);
    }
    // Same-origin / proxy builds: ignore loopback and continue.
  }

  // SSR / Node without window
  if (typeof window === "undefined") {
    if (
      process.env.NEXT_PUBLIC_USE_SAME_ORIGIN === "true" ||
      getConfiguredBasePath()
    ) {
      return sameOriginApiRoot(getConfiguredBasePath());
    }
    if (configured && !isAbsoluteHttpUrl(configured)) {
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
    !(configured && !isAbsoluteHttpUrl(configured))
  ) {
    return "http://127.0.0.1:8000/api/v1";
  }

  const basePath = getDeploymentBasePath();

  if (configured && !isAbsoluteHttpUrl(configured)) {
    const path = configured.replace(/^\.\//, "/");
    const normalized = path.startsWith("/") ? path : `/${path}`;

    if (basePath && normalized.startsWith(`${basePath}/`)) {
      return stripTrailingSlash(normalized);
    }
    // Stale baked session path from a previous Kaggle run — use live base.
    if (basePath && /\/proxy\/(?:proxy\/)?\d+\/api\/v1\/?$/.test(normalized)) {
      return sameOriginApiRoot(basePath);
    }
    if (normalized === "/api/v1" || normalized.startsWith("/api/")) {
      return stripTrailingSlash(`${basePath}${normalized}`);
    }
    return stripTrailingSlash(`${basePath}${normalized}`);
  }

  return sameOriginApiRoot(basePath);
}

/**
 * Origin used to resolve `/outputs/...` media paths.
 * Empty / path ⇒ same-origin relative URLs (required for Kaggle proxy).
 */
export function resolveApiOrigin(): string {
  const rawOrigin = (process.env.NEXT_PUBLIC_API_ORIGIN || "").trim();
  const onProxyBrowser =
    typeof window !== "undefined" && isProxyDeploymentHost(window.location.hostname);

  const configuredOrigin =
    onProxyBrowser && rawOrigin && isAbsoluteHttpUrl(rawOrigin) && isLoopbackAbsoluteUrl(rawOrigin)
      ? ""
      : rawOrigin;

  if (configuredOrigin) {
    const runtime = getDeploymentBasePath();
    if (
      runtime &&
      configuredOrigin.replace(/\/+$/, "") === "/proxy/8000" &&
      runtime !== "/proxy/8000"
    ) {
      return runtime;
    }
    if (isAbsoluteHttpUrl(configuredOrigin)) {
      if (isLoopbackAbsoluteUrl(configuredOrigin) && onProxyBrowser) {
        return runtime;
      }
      return stripTrailingSlash(configuredOrigin);
    }
    return stripTrailingSlash(configuredOrigin);
  }

  const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "").trim();
  if (apiUrl && isAbsoluteHttpUrl(apiUrl)) {
    if (!(onProxyBrowser && isLoopbackAbsoluteUrl(apiUrl))) {
      return stripTrailingSlash(apiUrl.replace(/\/api\/v1\/?$/, ""));
    }
  }

  if (
    typeof window !== "undefined" &&
    isLocalNextDevHost() &&
    process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true"
  ) {
    return "http://127.0.0.1:8000";
  }

  return getDeploymentBasePath();
}

/** Eager default for modules that read a constant; prefer resolveApiBaseUrl() in new code. */
export const API_BASE_URL = resolveApiBaseUrl();

export const API_ORIGIN = resolveApiOrigin();
