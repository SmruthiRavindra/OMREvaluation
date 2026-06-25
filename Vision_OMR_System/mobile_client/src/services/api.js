/**
 * api.js
 * ------
 * Asynchronous Axios connection layer between the React Native app and the
 * Node.js data gateway (which proxies through to the FastAPI backend).
 *
 * All requests go through the data gateway at DATA_GATEWAY_URL, never
 * directly to the Python service, so routing and auth can be handled
 * centrally.
 */

import axios from 'axios';

// ── Base URL ──────────────────────────────────────────────────────────────
// Override via environment / build config:
//   Android emulator   → 10.0.2.2 (maps to host loopback)
//   iOS simulator      → 127.0.0.1
//   Physical device    → replace with your machine's LAN IP

const DATA_GATEWAY_URL =
  process.env.DATA_GATEWAY_URL ?? 'http://10.0.2.2:3000';

// ── Axios instance ────────────────────────────────────────────────────────
const apiClient = axios.create({
  baseURL: DATA_GATEWAY_URL,
  timeout: 30_000, // 30 s – OMR processing can take a few seconds
  headers: {
    Accept: 'application/json',
  },
});

// ── Request interceptor (add auth token if stored) ────────────────────────
apiClient.interceptors.request.use(
  config => {
    // TODO: inject Bearer token from secure storage when auth is implemented
    // const token = await SecureStore.getItemAsync('auth_token');
    // if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  error => Promise.reject(error),
);

// ── Response interceptor (normalise errors) ───────────────────────────────
apiClient.interceptors.response.use(
  response => response.data,
  error => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'Unknown API error';
    return Promise.reject(new Error(message));
  },
);

// ── API methods ───────────────────────────────────────────────────────────

/**
 * Upload a captured OMR sheet image for evaluation.
 *
 * @param {{ uri: string, width: number, height: number }} photo
 *   Object returned by CameraScanner's onCapture callback.
 *
 * @param {function} [onUploadProgress]
 *   Optional progress handler: (percentCompleted: number) => void
 *
 * @returns {Promise<EvaluationResponse>}
 *   The grading result from the backend.
 */
export async function evaluateSheet(photo, onUploadProgress) {
  const form = new FormData();
  form.append('file', {
    uri:  photo.uri,
    name: 'omr_sheet.jpg',
    type: 'image/jpeg',
  });

  return apiClient.post('/api/evaluate', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: progressEvent => {
      if (onUploadProgress && progressEvent.total) {
        const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onUploadProgress(pct);
      }
    },
  });
}

/**
 * Submit confirmed grading results to be persisted in the database.
 *
 * @param {object} payload  – Evaluation result + metadata (sheet id, student id…)
 * @returns {Promise<{ id: string, saved: boolean }>}
 */
export async function submitResults(payload) {
  return apiClient.post('/api/submit', payload);
}

/**
 * Fetch the grading history for a given student or exam session.
 *
 * @param {{ studentId?: string, sessionId?: string }} filters
 * @returns {Promise<Array>}
 */
export async function fetchHistory(filters = {}) {
  return apiClient.get('/api/history', { params: filters });
}

/**
 * Health-check – can be used to verify connectivity on app launch.
 *
 * @returns {Promise<{ status: string }>}
 */
export async function checkHealth() {
  return apiClient.get('/health');
}

export default apiClient;
