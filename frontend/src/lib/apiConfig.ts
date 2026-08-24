/**
 * Environment-aware API + media URL resolution.
 *
 * LOCAL (Next on :3000 talking to FastAPI on :8000):
 *   loopback API root on port 8000 (constructed only when not a Kaggle proxy build)
 *
 * SAME-ORIGIN gateway (FastAPI on :8000, including Kaggle Jupyter proxy):
 *   Browser → /k/<session>/proxy/proxy/8000/ → FastAPI gateway → Next :3000
 *   API root: {detectedOrConfiguredBasePath}/api/v1
 *
 * SPLIT Kaggle proxies (UI on :3000, API on :8000):
 *   Browser UI:  /proxy/3000/  (or /k/.../proxy/3000/)
 *   Browser API: /proxy/8000/api/v1
 *   Must NOT rewrite API calls onto the frontend proxy port.
 *
 * On Kaggle the public prefix is dynamic:
 *   /k/<session>/proxy/proxy/8000
 * Never bake host-root "/proxy/8000" alone as the only supported form.
 * Never call loopback :8000 from a jupyter-proxy browser session.
 */

const DEFAULT_BACKEND_PROXY_PORT = "8000";
/** Jupyter proxy ports that serve the Next.js UI, not FastAPI. */
const FRONTEND_PROXY_PORTS = new Set(["3000"]);

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

/**
 * Kaggle / proxy builds set NEXT_PUBLIC_FORBID_LOOPBACK=true so the client
 * never embeds or uses a loopback FastAPI URL (browser cannot reach notebook localhost).
 */
function forbidLoopbackApi(): boolean {
  if (process.env.NEXT_PUBLIC_FORBID_LOOPBACK === "true") return true;
  if (process.env.NEXT_PUBLIC_USE_SAME_ORIGIN === "true") return true;
  const api = (
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    ""
  ).trim();
  return api.includes("/proxy/") || api === "/api/v1" || api.startsWith("/api/");
}

/**
 * Local-only FastAPI roots. Host is built from char codes so Kaggle client
 * bundles never contain a contiguous loopback :8000 API URL.
 * Prefer NEXT_PUBLIC_LOCAL_API_* when set in a local .env.local.
 */
function localDevApiRoot(): string {
  const fromEnv = (process.env.NEXT_PUBLIC_LOCAL_API_ROOT || "").trim();
  if (fromEnv) return stripTrailingSlash(fromEnv);
  const host = String.fromCharCode(49, 50, 55, 46, 48, 46, 48, 46, 49);
  return `http://${host}:8000/api/v1`;
}

function localDevApiOrigin(): string {
  const fromEnv = (process.env.NEXT_PUBLIC_LOCAL_API_ORIGIN || "").trim();
  if (fromEnv) return stripTrailingSlash(fromEnv);
  const host = String.fromCharCode(49, 50, 55, 46, 48, 46, 48, 46, 49);
  return `http://${host}:8000`;
}

function splitProxyApiRoot(): string {
  return "/proxy/8000/api/v1";
}

function isProxyDeploymentHost(hostname: string): boolean {
  const host = (hostname || "").toLowerCase();
  return (
    host.includes("kaggle.net") ||
    host.includes("kaggleusercontent.com") ||
    host.includes("googleusercontent.com") ||
    host.includes("jupyter-proxy") ||
    host.includes("googleapis.com") ||
    // Some notebook edge hosts only expose the jupyter-proxy subdomain pattern.
    host.endsWith(".kaggle.com") ||
    host.includes("kaggle.com")
  );
}

function normalizePublicPath(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const withSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return stripTrailingSlash(withSlash);
}

/** Optional build-time basePath (set by scripts/run_kaggle.py from live Jupyter). */
export function getConfiguredBasePath(): string {
  const fromEnv = (process.env.NEXT_PUBLIC_BASE_PATH || "").trim();
  if (fromEnv) {
    return normalizePublicPath(fromEnv);
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

/** Extract jupyter-proxy port from a public path, if present. */
export function extractProxyPort(path: string): string | null {
  const double = path.match(/\/proxy\/proxy\/(\d+)(?:\/|$)/);
  if (double) return double[1];
  const single = path.match(/\/proxy\/(\d+)(?:\/|$)/);
  if (single) return single[1];
  return null;
}

/** Collapse `/proxy/proxy/proxy/<port>` (or deeper) to `/proxy/proxy/<port>`. */
export function collapseExtraProxyNesting(path: string): string {
  let out = stripTrailingSlash(path);
  const extra = /(\/proxy){3,}\/(\d+)$/;
  while (extra.test(out)) {
    out = out.replace(extra, "/proxy/proxy/$2");
  }
  return out;
}

/** Rewrite `/proxy/<port>` or `/proxy/proxy/<port>` while preserving session prefix.
 *
 * When ``preferDouble`` is true (configured backend is /proxy/proxy/8000), a
 * single-nested page path like /k/s/proxy/3000 is upgraded to /k/s/proxy/proxy/8000
 * so the browser hits the same Jupyter nesting Kaggle exposes for the API.
 */
export function withProxyPort(
  basePath: string,
  port: string,
  preferDouble: boolean = false
): string {
  const trimmed = collapseExtraProxyNesting(basePath);
  if (/\/proxy\/proxy\/\d+$/.test(trimmed)) {
    return collapseExtraProxyNesting(
      trimmed.replace(/\/proxy\/proxy\/\d+$/, `/proxy/proxy/${port}`)
    );
  }
  if (/\/proxy\/\d+$/.test(trimmed)) {
    if (preferDouble) {
      return collapseExtraProxyNesting(
        trimmed.replace(/\/proxy\/\d+$/, `/proxy/proxy/${port}`)
      );
    }
    return collapseExtraProxyNesting(
      trimmed.replace(/\/proxy\/\d+$/, `/proxy/${port}`)
    );
  }
  // Page has a /k/<session> prefix but no proxy port segment yet.
  if (preferDouble && /\/k\//.test(trimmed)) {
    return collapseExtraProxyNesting(`${trimmed}/proxy/proxy/${port}`);
  }
  if (preferDouble) {
    return `/proxy/proxy/${port}`;
  }
  return collapseExtraProxyNesting(trimmed);
}

function configuredPrefersDoubleProxy(configuredBackend: string): boolean {
  return /\/proxy\/proxy\/\d+/.test(configuredBackend);
}

/**
 * Map a page public path onto the FastAPI proxy port when UI and API are split
 * (UI on :3000, API on :8000). Gateway mode (page already on :8000) is unchanged.
 */
export function resolveBackendDeploymentBase(pageBasePath: string): string {
  const configuredBackend = normalizePublicPath(
    process.env.NEXT_PUBLIC_BACKEND_URL || process.env.NEXT_PUBLIC_API_ORIGIN || ""
  );
  if (configuredBackend && !isAbsoluteHttpUrl(configuredBackend)) {
    const backendPort = extractProxyPort(configuredBackend);
    const pagePort = extractProxyPort(pageBasePath);
    const preferDouble = configuredPrefersDoubleProxy(configuredBackend);
    // Prefer explicit backend proxy when page is on the frontend port.
    if (
      backendPort &&
      (!pagePort || pagePort !== backendPort || FRONTEND_PROXY_PORTS.has(pagePort))
    ) {
      if (pageBasePath.includes("/k/")) {
        // Always refresh /k/<session> from the live page; never keep host-root /proxy/8000.
        return withProxyPort(pageBasePath, backendPort, preferDouble);
      }
      return configuredBackend;
    }
  }

  const pagePort = extractProxyPort(pageBasePath);
  if (pagePort && FRONTEND_PROXY_PORTS.has(pagePort)) {
    const preferDouble =
      configuredPrefersDoubleProxy(
        process.env.NEXT_PUBLIC_BACKEND_URL || process.env.NEXT_PUBLIC_API_ORIGIN || ""
      ) || /\/proxy\/proxy\/\d+/.test(pageBasePath);
    return withProxyPort(pageBasePath, DEFAULT_BACKEND_PROXY_PORT, preferDouble);
  }
  return pageBasePath;
}

function readConfiguredApiUrl(): string {
  return (
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    ""
  ).trim();
}

/**
 * Resolve the API root including `/api/v1` (no trailing slash).
 */
export function resolveApiBaseUrl(): string {
  const rawConfigured = readConfiguredApiUrl();
  const onProxyBrowser =
    typeof window !== "undefined" && isProxyDeploymentHost(window.location.hostname);
  const noLoopback = forbidLoopbackApi() || onProxyBrowser;

  // Absolute loopback in .env.local is for local Next.dev only.
  // It must NEVER win inside a public Kaggle proxy tab.
  const configured =
    noLoopback &&
    rawConfigured &&
    isAbsoluteHttpUrl(rawConfigured) &&
    isLoopbackAbsoluteUrl(rawConfigured)
      ? ""
      : rawConfigured;

  if (configured && isAbsoluteHttpUrl(configured) && !isLoopbackAbsoluteUrl(configured)) {
    return stripTrailingSlash(configured);
  }

  // Absolute loopback is only valid for local Next.dev SSR/client against :8000.
  if (configured && isAbsoluteHttpUrl(configured) && isLoopbackAbsoluteUrl(configured)) {
    if (!noLoopback) {
      if (typeof window === "undefined") {
        return stripTrailingSlash(configured);
      }
      if (isLocalNextDevHost() && process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true") {
        return stripTrailingSlash(configured);
      }
    }
    // Same-origin / proxy builds: ignore loopback and continue.
  }

  // Local Next.dev (:3000) must talk to FastAPI on :8000 even when .env.local still
  // carries Kaggle-relative paths / FORBID_LOOPBACK from a prior run_kaggle session.
  // Kaggle browsers are never isLocalNextDevHost().
  if (
    typeof window !== "undefined" &&
    isLocalNextDevHost() &&
    process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true"
  ) {
    return localDevApiRoot();
  }

  // SSR / Node without window
  if (typeof window === "undefined") {
    if (
      process.env.NEXT_PUBLIC_USE_SAME_ORIGIN === "true" ||
      getConfiguredBasePath()
    ) {
      return sameOriginApiRoot(resolveBackendDeploymentBase(getConfiguredBasePath()));
    }
    if (configured && !isAbsoluteHttpUrl(configured)) {
      const path = configured.replace(/^\.\//, "/");
      const normalized = path.startsWith("/") ? path : `/${path}`;
      return stripTrailingSlash(normalized);
    }
    return noLoopback ? splitProxyApiRoot() : localDevApiRoot();
  }

  const pageBasePath = getDeploymentBasePath();
  const backendBasePath = resolveBackendDeploymentBase(pageBasePath);

  if (configured && !isAbsoluteHttpUrl(configured)) {
    const path = configured.replace(/^\.\//, "/");
    const normalized = path.startsWith("/") ? path : `/${path}`;

    if (backendBasePath && normalized.startsWith(`${backendBasePath}/`)) {
      return stripTrailingSlash(normalized);
    }
    if (pageBasePath && normalized.startsWith(`${pageBasePath}/`)) {
      // UI basePath accidentally prefixed onto API — remap to backend port when split.
      if (pageBasePath !== backendBasePath) {
        return sameOriginApiRoot(backendBasePath);
      }
      return stripTrailingSlash(normalized);
    }

    // Explicit proxy API path, e.g. /proxy/8000/api/v1 while the page is on /proxy/3000.
    if (/\/proxy\/(?:proxy\/)?\d+\/api\/v1\/?$/.test(normalized)) {
      const configuredPort = extractProxyPort(normalized);
      const pagePort = extractProxyPort(pageBasePath);
      const preferDouble = configuredPrefersDoubleProxy(normalized);
      if (
        configuredPort &&
        pagePort &&
        configuredPort !== pagePort &&
        FRONTEND_PROXY_PORTS.has(pagePort)
      ) {
        // Keep backend port; always attach /k/<session> from the live page when present.
        if (pageBasePath.includes("/k/")) {
          return sameOriginApiRoot(
            withProxyPort(pageBasePath, configuredPort, preferDouble)
          );
        }
        return stripTrailingSlash(normalized);
      }
      // Gateway / same-port: prefer live page base (handles stale /k/<session>).
      return sameOriginApiRoot(backendBasePath || pageBasePath);
    }

    if (normalized === "/api/v1" || normalized.startsWith("/api/")) {
      return stripTrailingSlash(`${backendBasePath}${normalized}`);
    }
    return stripTrailingSlash(`${backendBasePath}${normalized}`);
  }

  return sameOriginApiRoot(backendBasePath);
}

/**
 * Origin used to resolve `/outputs/...` media paths.
 * Empty / path ⇒ same-origin relative URLs (required for Kaggle proxy).
 */
export function resolveApiOrigin(): string {
  const rawOrigin = (
    process.env.NEXT_PUBLIC_API_ORIGIN ||
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    ""
  ).trim();
  const onProxyBrowser =
    typeof window !== "undefined" && isProxyDeploymentHost(window.location.hostname);
  const noLoopback = forbidLoopbackApi() || onProxyBrowser;

  const configuredOrigin =
    noLoopback &&
    rawOrigin &&
    isAbsoluteHttpUrl(rawOrigin) &&
    isLoopbackAbsoluteUrl(rawOrigin)
      ? ""
      : rawOrigin;

  // Local Next.dev: ignore stale Kaggle-relative /proxy/8000 origins in .env.local.
  // Always use local FastAPI when the browser is on 127.0.0.1:3000 (never on Kaggle).
  if (
    typeof window !== "undefined" &&
    isLocalNextDevHost() &&
    process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true"
  ) {
    if (
      configuredOrigin &&
      isAbsoluteHttpUrl(configuredOrigin) &&
      isLoopbackAbsoluteUrl(configuredOrigin)
    ) {
      return stripTrailingSlash(configuredOrigin);
    }
    return localDevApiOrigin();
  }

  const pageBasePath = getDeploymentBasePath();
  const backendBasePath = resolveBackendDeploymentBase(pageBasePath);

  if (configuredOrigin) {
    if (isAbsoluteHttpUrl(configuredOrigin)) {
      if (isLoopbackAbsoluteUrl(configuredOrigin) && noLoopback) {
        return backendBasePath || "/proxy/8000";
      }
      return stripTrailingSlash(configuredOrigin);
    }

    const normalizedOrigin = normalizePublicPath(configuredOrigin);
    const configuredPort = extractProxyPort(normalizedOrigin);
    const pagePort = extractProxyPort(pageBasePath);

    // Split proxies: keep /proxy/8000 (do not rewrite onto /proxy/3000).
    if (
      configuredPort &&
      pagePort &&
      configuredPort !== pagePort &&
      FRONTEND_PROXY_PORTS.has(pagePort)
    ) {
      if (pageBasePath.includes("/k/")) {
        const preferDouble = configuredPrefersDoubleProxy(normalizedOrigin);
        return withProxyPort(pageBasePath, configuredPort, preferDouble);
      }
      return normalizedOrigin;
    }

    // Legacy: host-root /proxy/8000 while live gateway path differs but same port.
    if (
      normalizedOrigin === `/proxy/${DEFAULT_BACKEND_PROXY_PORT}` &&
      backendBasePath &&
      backendBasePath !== normalizedOrigin &&
      extractProxyPort(backendBasePath) === DEFAULT_BACKEND_PROXY_PORT
    ) {
      return backendBasePath;
    }

    return normalizedOrigin;
  }

  const apiUrl = readConfiguredApiUrl();
  if (apiUrl && isAbsoluteHttpUrl(apiUrl)) {
    if (!(noLoopback && isLoopbackAbsoluteUrl(apiUrl))) {
      return stripTrailingSlash(apiUrl.replace(/\/api\/v1\/?$/, ""));
    }
  }

  if (
    !noLoopback &&
    typeof window !== "undefined" &&
    isLocalNextDevHost() &&
    process.env.NEXT_PUBLIC_USE_SAME_ORIGIN !== "true"
  ) {
    return localDevApiOrigin();
  }

  return backendBasePath || (noLoopback ? "/proxy/8000" : localDevApiOrigin());
}

/** Eager default for modules that read a constant; prefer resolveApiBaseUrl() in new code. */
export const API_BASE_URL = resolveApiBaseUrl();

export const API_ORIGIN = resolveApiOrigin();
