export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";

export interface JobStatusResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  current_step: string;
  result_url?: string;
  metadata?: any;
  error?: string;
}

export class ApiClient {
  static async get<T>(endpoint: string): Promise<T> {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      }
    });
    
    if (!res.ok) {
      throw new Error(`API Error: ${res.status} ${res.statusText}`);
    }
    return res.json();
  }

  static async post<T>(endpoint: string, data: any): Promise<T> {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(data)
    });
    
    if (!res.ok) {
      throw new Error(`API Error: ${res.status} ${res.statusText}`);
    }
    return res.json();
  }

  static async postFormData<T>(endpoint: string, formData: FormData): Promise<T> {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
      // Note: Do not set Content-Type header when sending FormData,
      // the browser automatically sets it along with the correct boundary.
    });
    
    if (!res.ok) {
      throw new Error(`API Error: ${res.status} ${res.statusText}`);
    }
    return res.json();
  }
  
  /**
   * Polls a job status endpoint repeatedly until it is no longer queued or processing.
   * By default, polls every 2 seconds.
   */
  static async pollJobStatus(jobId: string, onProgress?: (status: JobStatusResponse) => void): Promise<JobStatusResponse> {
    const pollInterval = 2000;
    
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const statusRes = await this.get<JobStatusResponse>(`/status/${jobId}`);
          
          if (onProgress) {
            onProgress(statusRes);
          }
          
          if (statusRes.status === "completed" || statusRes.status === "failed") {
            resolve(statusRes);
          } else {
            setTimeout(poll, pollInterval);
          }
        } catch (error) {
          reject(error);
        }
      };
      
      poll();
    });
  }
}
