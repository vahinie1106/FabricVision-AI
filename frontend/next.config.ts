import type { NextConfig } from "next";
import path from "path";

/**
 * Pin Turbopack/Next to this frontend package.
 *
 * An orphan repo-root package-lock.json (empty packages, no package.json) caused
 * Next 16 to infer the parent as workspace root, breaking CSS/module resolution
 * under Kaggle and triggering the multiple-lockfiles warning.
 */
const frontendRoot = path.resolve(__dirname);

/**
 * Optional public path prefix for Jupyter/Kaggle reverse proxies.
 *
 * On Kaggle, scripts/run_kaggle.py sets this dynamically from the live Jupyter
 * server base_url. Split-proxy uses NEXT_PUBLIC_ASSET_PREFIX for PORT 3000
 * (example /k/<session>/proxy/proxy/3000) and leaves basePath empty so
 * jupyter-server-proxy can strip the prefix. Never use an :8000 prefix for assets.
 *
 * Leave unset for local development (Next on :3000 or gateway without prefix).
 */
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH || "").trim().replace(/\/+$/, "");
/** Split-proxy Kaggle: keep routes at `/` (jupyter strips the prefix) but load
 * `/_next` assets from the public PORT 3000 path so the browser does not hit
 * host-root `/_next` (blank page). Never use an :8000 prefix here. */
const assetPrefix = (
  process.env.NEXT_PUBLIC_ASSET_PREFIX ||
  process.env.NEXT_PUBLIC_BASE_PATH ||
  ""
)
  .trim()
  .replace(/\/+$/, "");

if (assetPrefix.includes("8000") && !(basePath && basePath.includes("8000"))) {
  throw new Error(
    `NEXT_PUBLIC_ASSET_PREFIX must be a PORT 3000 path, not 8000 (got ${assetPrefix})`
  );
}

const nextConfig: NextConfig = {
  // Absolute path required; keeps PostCSS/Tailwind scoped to frontend/.
  turbopack: {
    root: frontendRoot,
  },
  // jupyter-server-proxy + Next absolute 308s to localhost/0.0.0.0 break PORT 3000.
  skipTrailingSlashRedirect: true,
  trailingSlash: false,
  ...(basePath
    ? {
        basePath,
        assetPrefix: assetPrefix || basePath,
      }
    : assetPrefix
      ? { assetPrefix }
      : {}),
};

export default nextConfig;
