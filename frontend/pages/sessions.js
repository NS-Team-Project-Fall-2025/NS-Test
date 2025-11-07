import { useEffect, useState } from "react";
import { listSessions, getSession, deleteSession } from "@/lib/api";

export default function SessionsPage() {
  const [sessions, setSessions] = useState([]);
  const [selected, setSelected] = useState(null);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
    try {
      const payload = await listSessions();
      setSessions(payload.sessions || []);
    } catch (err) {
      setMessage({ text: err.message, tone: "danger" });
    }
  }

  async function loadSession(id) {
    try {
      const session = await getSession(id);
      if (session.error) {
        setMessage({ text: session.error, tone: "danger" });
        return;
      }
      setSelected(session);
    } catch (err) {
      setMessage({ text: err.message, tone: "danger" });
    }
  }

  async function handleDelete(id) {
    try {
      await deleteSession(id);
      setMessage({ text: "Session deleted.", tone: "success" });
      setSelected(null);
      refresh();
    } catch (err) {
      setMessage({ text: err.message, tone: "danger" });
    }
  }

  return (
    <div className="card">
      <h2>Saved Sessions</h2>
      <p className="muted">Review or clean up persisted tutor conversations.</p>

      {message && (
        <div
          className="chat-bubble"
          style={{
            borderColor: message.tone === "danger" ? "rgba(248,113,113,0.4)" : "rgba(52,211,153,0.35)",
            color: message.tone === "danger" ? "#fecaca" : "#dcfce7"
          }}
        >
          {message.text}
        </div>
      )}

      <div className="grid two" style={{ marginTop: "1.5rem" }}>
        <div className="card" style={{ margin: 0 }}>
          <h3>Sessions</h3>
          {sessions.length === 0 && <p className="muted">No sessions saved yet.</p>}
          <div className="file-list">
            {sessions.map((session) => (
              <div key={session.session_id} className="file-item">
                <div>
                  <strong>{session.title || session.session_id}</strong>
                  <div className="list-muted">Messages: {session.message_count}</div>
                  <div className="list-muted">
                    Updated: {session.updated_at ? new Date(session.updated_at).toLocaleString() : "—"}
                  </div>
                </div>
                <div className="inline-form" style={{ marginBottom: 0 }}>
                  <button className="button secondary" type="button" onClick={() => loadSession(session.session_id)}>
                    View
                  </button>
                  <button className="button secondary" type="button" onClick={() => handleDelete(session.session_id)}>
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card" style={{ margin: 0 }}>
          <h3>Details</h3>
          {!selected && <p className="muted">Select a session to inspect its content.</p>}
          {selected && (
            <div>
              <div className="pill" style={{ marginBottom: "1rem" }}>
                {selected.title || selected.session_id}
              </div>
              <div className="chat-window" style={{ maxHeight: "60vh" }}>
                {(selected.messages || []).map((msg, idx) => (
                  <div key={idx} className={`chat-bubble ${msg.role}`}>
                    <h4>{msg.role === "user" ? "User" : "NetSec Tutor"}</h4>
                    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{msg.content}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
