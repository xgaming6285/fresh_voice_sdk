import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Token management
let authToken = localStorage.getItem("token");

api.setAuthToken = (token) => {
  authToken = token;
  if (token) {
    api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common["Authorization"];
  }
};

// Set initial token if exists
if (authToken) {
  api.setAuthToken(authToken);
}

// Request interceptor for auth
api.interceptors.request.use(
  (config) => {
    // Token is already set in defaults, but double-check
    if (authToken && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${authToken}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem("token");
      authToken = null;
      delete api.defaults.headers.common["Authorization"];
      // Only redirect if not already on login page
      if (!window.location.pathname.includes("/login")) {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  }
);

// Lead APIs
export const leadAPI = {
  getAll: (params) => api.get("/api/crm/leads", { params }),
  getById: (id) => api.get(`/api/crm/leads/${id}`),
  create: (data) => api.post("/api/crm/leads", data),
  update: (id, data) => api.put(`/api/crm/leads/${id}`, data),
  delete: (id) => api.delete(`/api/crm/leads/${id}`),
  importCSV: (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/api/crm/leads/import", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
};

// Campaign APIs
export const campaignAPI = {
  getAll: (params) => api.get("/api/crm/campaigns", { params }),
  getById: (id) => api.get(`/api/crm/campaigns/${id}`),
  create: (data) => api.post("/api/crm/campaigns", data),
  update: (id, data) => api.put(`/api/crm/campaigns/${id}`, data),
  delete: (id) => api.delete(`/api/crm/campaigns/${id}`),
  addLeads: (id, leadIds, priority = 0) =>
    api.post(`/api/crm/campaigns/${id}/leads`, { lead_ids: leadIds, priority }),
  addFilteredLeads: (id, filter) =>
    api.post(`/api/crm/campaigns/${id}/leads/filter`, filter),
  getLeads: (id, params) =>
    api.get(`/api/crm/campaigns/${id}/leads`, { params }),
  start: (id) => api.post(`/api/crm/campaigns/${id}/start`),
  pause: (id) => api.post(`/api/crm/campaigns/${id}/pause`),
  stop: (id) => api.post(`/api/crm/campaigns/${id}/stop`),
};

// Session APIs
export const sessionAPI = {
  getAll: (params) => api.get("/api/crm/sessions", { params }),
  getById: (sessionId) => api.get(`/api/crm/sessions/${sessionId}`),
};

// Voice Agent APIs
export const voiceAgentAPI = {
  health: () => api.get("/health"),
  status: () => api.get("/health"), // Use health endpoint to check if voice agent is online
  config: () => api.get("/api/config"),
  activeSessions: () => api.get("/api/sessions"),
  recordings: () => api.get("/api/recordings"),
  transcripts: (sessionId) => api.get(`/api/transcripts/${sessionId}`),
  getTranscript: (sessionId, audioType) =>
    api.get(`/api/transcripts/${sessionId}/${audioType}`),
  retranscribe: (sessionId) =>
    api.post(`/api/transcripts/${sessionId}/retranscribe`),
  makeCall: (phoneNumber, callConfig = null, greetingFile = null) => {
    const payload = { phone_number: phoneNumber };
    if (callConfig) {
      payload.call_config = callConfig;
    }
    if (greetingFile) {
      payload.greeting_file = greetingFile;
    }
    return api.post("/api/make_call", payload);
  },
};

// Authentication APIs
export const authAPI = {
  login: (username, password) =>
    api.post("/api/auth/login", { username, password }),
  register: (userData) => api.post("/api/auth/register", userData),
  getMe: () => api.get("/api/auth/me"),
  verifyToken: () => api.post("/api/auth/verify-token"),
};

// User Management APIs (Admin only)
export const userAPI = {
  getAgents: () => api.get("/api/users/agents"),
  getAgent: (id) => api.get(`/api/users/agents/${id}`),
  createAgent: (data) => api.post("/api/users/agents", data),
  updateAgent: (id, data) => api.put(`/api/users/agents/${id}`, data),
  deleteAgent: (id) => api.delete(`/api/users/agents/${id}`),
};

export default api;
