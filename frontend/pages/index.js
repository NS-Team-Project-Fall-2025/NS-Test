import { useEffect, useRef, useState } from "react";
import {
  fetchConfig,
  streamChat,
  listSessions,
  createSession,
  updateSession,
  getSession
} from "@/lib/api";

export default function TutorPage() {
  const [config, setConfig] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("combined");
  const [isStreaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [isLoadingSessions, setLoadingSessions] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchConfig()
      .then((cfg) => setConfig(cfg))
      .catch(() => setConfig(null));
    refreshSessions();
  }, []);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  async function refreshSessions() {
    setLoadingSessions(true);
    try {
      const payload = await listSessions();
      setSessions(payload.sessions || []);
    } catch (err) {
      // ignore session load issues during first render
    } finally {
      setLoadingSessions(false);
    }
  }

  async function handleCreateSession() {
    try {
      const created = await createSession();
      const id = created.session_id || created.sessionId;
      if (id) {
        setActiveSession(id);
      }
      setMessages([]);
      setError(null);
      refreshSessions();
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSessionChange(id) {
    if (!id) {
      setActiveSession(null);
      setMessages([]);
      return;
    }
    try {
      const session = await getSession(id);
      setActiveSession(id);
      setMessages(session.messages || []);
    } catch (err) {
      setError(err.message);
    }
  }

  async function persistSession(sessionId, updatedMessages) {
    if (!sessionId) return;
    try {
      await updateSession(sessionId, {
        messages: updatedMessages,
        title: updatedMessages[0]?.content?.slice(0, 60) || undefined
      });
    } catch (err) {
      // surface persistence errors softly
      console.error("Failed to persist session", err);
    }
  }

  async function persistWithAutoSession(questionText, updatedMessages) {
    let sessionId = activeSession;
    if (!sessionId) {
      const trimmedTitle = (questionText || "").slice(0, 60) || "New session";
      try {
        const created = await createSession(trimmedTitle);
        sessionId = created.session_id || created.sessionId;
        if (!sessionId) {
          throw new Error("Failed to create a session.");
        }
        setActiveSession(sessionId);
        refreshSessions();
      } catch (err) {
        console.error("Failed to auto-create session", err);
        setError(err.message || "Unable to auto-create a session.");
        return;
      }
    }
    await persistSession(sessionId, updatedMessages);
  }

  async function submitMessage(question) {
    if (!question || isStreaming) return;
    setError(null);

    const baseHistory = messages.map((msg) => ({
      role: msg.role,
      content: msg.content
    }));
    const submissionHistory = [...baseHistory, { role: "user", content: question }];

    const userMessage = { role: "user", content: question };
    const assistantMessage = { role: "assistant", content: "", sources: [], streaming: true };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setStreaming(true);

    try {
      await streamChat(
        {
          question,
          history: submissionHistory,
          mode
        },
        (event) => {
          if (event.type === "token") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === "assistant") {
                last.content = (last.content || "") + event.text;
              }
              return next;
            });
          } else if (event.type === "final") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === "assistant") {
                last.content = event.answer || last.content;
                last.sources = event.sources || [];
                last.streaming = false;
              }
              return next;
            });
            const persistedMessages = [...submissionHistory, { role: "assistant", content: event.answer }];
            persistWithAutoSession(question, persistedMessages);
          } else if (event.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === "assistant") {
                last.content = `[Error] ${event.error || event.text || "Unable to complete request."}`;
                last.streaming = false;
              }
              return next;
            });
            setError(event.error || event.text || "Unable to complete request.");
          }
        }
      );
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === "assistant") {
          last.content = `[Error] ${err.message}`;
          last.streaming = false;
        }
        return next;
      });
      setError(err.message);
    } finally {
      setStreaming(false);
    }
  }

  async function handleFormSubmit(event) {
    event.preventDefault();
    const question = input.trim();
    if (!question) return;
    setInput("");
    await submitMessage(question);
  }

  function handleInputKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      const question = input.trim();
      if (!question || isStreaming) return;
      setInput("");
      submitMessage(question);
    }
  }

  return (
    <div className="tutor-layout">
      <div className="tutor-toolbar">
        <div className="tutor-toolbar-left">
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="combined">Combined knowledge</option>
            <option value="textbooks">Textbooks only</option>
            <option value="slides">Lecture slides</option>
          </select>
          {config && (
            <div className="tutor-pill">
              <span>{config.ollama?.model}</span>
              <span>·</span>
              <span>{mode}</span>
            </div>
          )}
        </div>
        <div className="tutor-toolbar-right">
          {messages.length > 0 && (
            <button
              className="button secondary"
              type="button"
              onClick={handleCreateSession}
              disabled={isStreaming}
            >
              New session
            </button>
          )}
          <select
            value={activeSession || ""}
            onChange={(e) => handleSessionChange(e.target.value)}
            disabled={isLoadingSessions}
          >
            <option value="">No saved session</option>
            {sessions.map((session) => (
              <option key={session.session_id} value={session.session_id}>
                {session.title || session.session_id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="chat-bubble alert">
          {error}
        </div>
      )}

      <div className="tutor-chat-area">
        <div className="chat-window">
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}-${message.content.slice(0, 5)}`}
              className={`chat-bubble ${message.role} ${message.streaming ? "streaming" : ""}`}
            >
              <h4>{message.role === "user" ? "You" : "NetSec Tutor"}</h4>
              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
                {message.content}
                {message.streaming && (
                  <span className="typing-indicator">
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                  </span>
                )}
              </div>
              {message.sources && message.sources.length > 0 && !message.streaming && (
                <div className="chat-sources">
                  <div className="chat-sources-header">Sources ({message.sources.length}):</div>
                  {message.sources.map((source, idx) => {
                    // Ensure source is an object with proper structure
                    const sourceObj = typeof source === 'object' && source !== null ? source : {};
                    return (
                      <div key={idx} className="chat-source">
                        <strong>{sourceObj.filename || sourceObj.source || "Unknown source"}</strong>
                        {sourceObj.page_number ? <span> · page {sourceObj.page_number}</span> : null}
                        {sourceObj.content_preview ? (
                          <div className="list-muted" style={{ marginTop: "0.35rem" }}>
                            {sourceObj.content_preview}
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <form onSubmit={handleFormSubmit} className="chat-input-row">
          <textarea
            className="textarea-small"
            placeholder="Ask about Network Security materials, type your question here..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleInputKeyDown}
            disabled={isStreaming}
          />
          <button className="button" type="submit" disabled={isStreaming || !input.trim()}>
            {isStreaming ? "Streaming..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
}
