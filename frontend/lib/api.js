/**
 * API client for NetSec Tutor backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000/api";

/**
 * Generic fetch wrapper with error handling.
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  };

  // Handle FormData (for file uploads)
  if (options.body instanceof FormData) {
    delete config.headers["Content-Type"];
  }

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    // Handle streaming responses (NDJSON)
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/x-ndjson")) {
      return response; // Return the response for streaming
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error("Failed to connect to backend. Is the server running?");
    }
    throw error;
  }
}

/**
 * Fetch configuration from the backend.
 */
export async function fetchConfig() {
  return apiRequest("/config/");
}

/**
 * Stream chat responses from the tutor.
 * @param {Object} payload - { question, history, mode }
 * @param {Function} onEvent - Callback for each event: { type: "token"|"final"|"error", text?, answer?, sources?, error? }
 */
export async function streamChat(payload, onEvent) {
  const url = `${API_BASE}/chat/stream/`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line);
          onEvent(event);
        } catch (e) {
          console.error("Failed to parse event:", line, e);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * List all chat sessions.
 */
export async function listSessions() {
  return apiRequest("/sessions/");
}

/**
 * Create a new chat session.
 * @param {string} title - Optional session title
 */
export async function createSession(title) {
  return apiRequest("/sessions/create/", {
    method: "POST",
    body: JSON.stringify(title ? { title } : {}),
  });
}

/**
 * Get a specific session by ID.
 */
export async function getSession(sessionId) {
  return apiRequest(`/sessions/${sessionId}/`);
}

/**
 * Update a session (messages and/or title).
 */
export async function updateSession(sessionId, { messages, title }) {
  return apiRequest(`/sessions/${sessionId}/update/`, {
    method: "PATCH",
    body: JSON.stringify({ messages, title }),
  });
}

/**
 * Delete a session.
 */
export async function deleteSession(sessionId) {
  return apiRequest(`/sessions/${sessionId}/`, {
    method: "DELETE",
  });
}

/**
 * List knowledge base files.
 */
export async function listKnowledgeBase() {
  return apiRequest("/kb/files/");
}

/**
 * Upload files to the knowledge base.
 * @param {File[]} files - Array of File objects
 * @param {string} category - "textbooks" or "slides"
 */
export async function uploadKnowledgeBaseFile(files, category) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("category", category);

  return apiRequest("/kb/upload/", {
    method: "POST",
    body: formData,
  });
}

/**
 * Rebuild the knowledge base vector stores.
 * @param {string[]} categories - Array of category names to rebuild
 */
export async function rebuildKnowledgeBase(categories) {
  return apiRequest("/kb/rebuild/", {
    method: "POST",
    body: JSON.stringify({ categories }),
  });
}

/**
 * Clear the knowledge base.
 */
export async function clearKnowledgeBase() {
  return apiRequest("/kb/clear/", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

/**
 * Summarize a document based on a prompt.
 * @param {string} prompt - The summarization prompt
 */
export async function summarizePrompt(prompt) {
  return apiRequest("/summary/", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}

/**
 * Generate a quiz.
 * @param {Object} payload - Quiz generation parameters
 */
export async function generateQuiz(payload) {
  return apiRequest("/quiz/generate/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Grade a quiz.
 * @param {Object} quizData - The quiz data
 * @param {Object} userAnswers - User's answers
 */
export async function gradeQuiz(quizData, userAnswers) {
  return apiRequest("/quiz/grade/", {
    method: "POST",
    body: JSON.stringify({ quiz_data: quizData, user_answers: userAnswers }),
  });
}

/**
 * List quiz history/attempts.
 */
export async function listQuizHistory() {
  return apiRequest("/quiz/history/");
}

/**
 * Get a specific quiz attempt by ID.
 */
export async function getQuizAttempt(quizId) {
  return apiRequest(`/quiz/history/${quizId}/`);
}

/**
 * Delete a quiz attempt.
 */
export async function deleteQuizAttemptApi(quizId) {
  return apiRequest(`/quiz/history/${quizId}/`, {
    method: "DELETE",
  });
}

