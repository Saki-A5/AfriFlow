import axios from 'axios';

// Prefer configuring this in `.env` as REACT_APP_API_URL for local/dev/prod parity.
// Example:
// - local:   REACT_APP_API_URL=http://localhost:5000
// - render:  REACT_APP_API_URL=https://afriflow-f5df.onrender.com
const API_BASE_URL =
  process.env.REACT_APP_API_URL || 'https://afriflow-1.onrender.com';

// Backend routes are mounted under `/api/*` (e.g. `/api/auth/login`).
// So the axios client should point at the backend root + `/api`.
const API_ROOT = `${API_BASE_URL.replace(/\/+$/, '')}/api`;

const api = axios.create({
  baseURL: API_ROOT,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
});

// ✅ Runs before EVERY request — always picks up the latest token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('afriflow_token');
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  res => res,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('afriflow_token');
      window.location.href = '/';
    }
    return Promise.reject(err);
  }
);

export default api;