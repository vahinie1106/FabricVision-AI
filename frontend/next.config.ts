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
 * server base_url (for example /k/<session>/proxy/proxy/8000).
 * Do NOT hard-code host-root /proxy/8000 here.
 *
 * Leave unset for local development (Next on :3000 or gateway without prefix).
 */
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH || "").trim().replace(/\/+$/, "");

const nextConfig: NextConfig = {
  // Absolute path required; keeps PostCSS/Tailwind scoped to frontend/.
  turbopack: {
    root: frontendRoot,
  },
  ...(basePath
    ? {
        basePath,
        // Keep assets under the same public prefix so /_next/* stays on the proxy.
        assetPrefix: basePath,
      }
    : {}),
};

export default nextConfig;
