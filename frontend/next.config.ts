import type { NextConfig } from "next";

/**
 * Optional public path prefix for Jupyter/Kaggle reverse proxies.
 * Example: NEXT_PUBLIC_BASE_PATH=/proxy/8000
 *
 * Leave unset for local development (Next on :3000 or gateway without prefix).
 */
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH || "").trim().replace(/\/+$/, "");

const nextConfig: NextConfig = {
  ...(basePath
    ? {
        basePath,
        // Keep assets under the same public prefix so /_next/* stays on the proxy.
        assetPrefix: basePath,
      }
    : {}),
};

export default nextConfig;
