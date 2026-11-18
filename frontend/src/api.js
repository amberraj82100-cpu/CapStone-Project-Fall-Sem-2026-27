// Address of the NeuroQuest Java backend. Override with VITE_API_URL in frontend/.env.local
// (e.g. VITE_API_URL=http://localhost:8081 when port 8080 is used by another program).
export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080';
