
const API_BASE_URL = "http://localhost:8000";

/**
 * Sends a prompt to the adaptive inference endpoint.
 * @param {string} prompt - User query
 * @returns {Promise<object>} - JSON response
 */
export const inferAdaptive = async (prompt) => {
    const response = await fetch(`${API_BASE_URL}/infer-adaptive`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
    });
    if (!response.ok) throw new Error("Network response was not ok");
    return response.json();
};

/**
 * Returns the SSE stream URL for the given prompt.
 * Note: SSE usually uses GET, but our endpoint is POST.
 * For true SSE with POST, we use fetch-event-source or similar libraries,
 * OR we can just use the standard fetch with reader if we want to handle the stream manually.
 * 
 * Our server endpoint /infer-stream is POST.
 * Standard EventSource does NOT support POST.
 * We will use fetch() with ReadableStream reader in the component.
 */
// ... existing exports
export const STREAM_ENDPOINT = `${API_BASE_URL}/infer-stream`;

/**
 * Fetches system health metrics from the backend.
 * @returns {Promise<object>} - JSON stats
 */
export const fetchSystemStats = async () => {
    const response = await fetch(`${API_BASE_URL}/system-stats`);
    if (!response.ok) throw new Error("Failed to fetch stats");
    return response.json();
};
