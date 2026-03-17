import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '/api',
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