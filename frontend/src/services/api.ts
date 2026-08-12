import { resolveApiBaseUrl } from "@/lib/apiConfig";

export {
  API_BASE_URL,
  API_ORIGIN,
  detectRuntimeBasePath,
  getConfiguredBasePath,
  getDeploymentBasePath,
  resolveApiBaseUrl,
  resolveApiOrigin,
} from "@/lib/apiConfig";

export interface JobStatusResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  current_step: string;
  /** Authoritative backend lifecycle stage (prefer over UI inference). */
  stage?: string;
  result_url?: string;
  metadata?: any;
  error?: string;
  error_type?: string;
  failed_stage?: string;
}

async function readApiError(res: Response): Promise<string> {
  const fallback = `API Error: ${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
    if (Array.isArray(detail) && detail.length) {
      return detail
        .map((item: { msg?: string } | string) =>
          typeof item === "string" ? item : item?.msg || JSON.stringify(item)
        )
        .join("; ");
    }
    if (typeof body?.error === "string" && body.error.trim()) {
      return body.error;
    }
  } catch {
    // ignore non-JSON error bodies
  }
  return fallback;
}

export class ApiClient {
  static async get<T>(endpoint: string): Promise<T> {
    const base = resolveApiBaseUrl();
    let res: Response;
    try {
      res = await fetch(`${base}${endpoint}`, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      });
    } catch {
      throw new Error("Unable to connect to backend.");
    }

    if (!res.ok) {
      throw new Error(await readApiError(res));
    }
    return res.json();
  }

  static async post<T>(endpoint: string, data: unknown): Promise<T> {
    const base = resolveApiBaseUrl();
    let res: Response;
    try {
      res = await fetch(`${base}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(data),
      });
    } catch {
      throw new Error("Unable to connect to backend.");
    }

    if (!res.ok) {
      throw new Error(await readApiError(res));
    }
    return res.json();
  }

  static async postFormData<T>(endpoint: string, formData: FormData): Promise<T> {
    const base = resolveApiBaseUrl();
    let res: Response;
    try {
      res = await fetch(`${base}${endpoint}`, {
        method: "POST",
        body: formData,
        // Note: Do not set Content-Type header when sending FormData,
        // the browser automatically sets it along with the correct boundary.
      });
    } catch {
      throw new Error("Unable to connect to backend.");
    }

    if (!res.ok) {
      throw new Error(await readApiError(res));
    }
    return res.json();
  }

  /**
   * Polls a job status endpoint repeatedly until it is no longer queued or processing.
   * Long FLUX Kontext jobs on RTX 3050 can run for many minutes — transient network
   * blips must not surface as "Unable to connect to backend".
   */
  static async pollJobStatus(
    jobId: string,
    onProgress?: (status: JobStatusResponse) => void,
    options?: { maxPollMs?: number }
  ): Promise<JobStatusResponse> {
    const pollInterval = 2000;
    const maxConsecutiveErrors = 8;
    // FLUX Standard on RTX 3050 can exceed 10 minutes; allow long jobs but not forever.
    const maxPollMs = options?.maxPollMs ?? 45 * 60 * 1000;
    const startedAt = Date.now();

    return new Promise((resolve, reject) => {
      let consecutiveErrors = 0;

      const poll = async () => {
        try {
          if (Date.now() - startedAt > maxPollMs) {
            reject(
              Object.assign(
                new Error(
                  `Job timed out after ${Math.round(maxPollMs / 60000)} minutes while still processing.`
                ),
                { errorType: "TIMEOUT" }
              )
            );
            return;
          }

          const statusRes = await this.get<JobStatusResponse>(`/status/${jobId}`);
          consecutiveErrors = 0;

          if (onProgress) {
            onProgress(statusRes);
          }

          if (statusRes.status === "completed" || statusRes.status === "failed") {
            resolve(statusRes);
          } else {
            setTimeout(poll, pollInterval);
          }
        } catch (error) {
          consecutiveErrors += 1;
          if (consecutiveErrors >= maxConsecutiveErrors) {
            reject(error);
            return;
          }
          // Retry: long-running generation must stay alive across brief failures
          setTimeout(poll, pollInterval);
        }
      };

      poll();
    });
  }
}
