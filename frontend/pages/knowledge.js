import { useEffect, useRef, useState } from "react";
import {
  listKnowledgeBase,
  uploadKnowledgeBaseFile,
  rebuildKnowledgeBase,
  clearKnowledgeBase,
  summarizePrompt
} from "@/lib/api";

const CATEGORY_LABELS = {
  textbooks: "Textbooks",
  slides: "Lecture Slides"
};

export default function KnowledgeBasePage() {
  const [files, setFiles] = useState({});
  const [category, setCategory] = useState("textbooks");
  const [message, setMessage] = useState(null);
  const [isBusy, setBusy] = useState(false);
  const [summaryPrompt, setSummaryPrompt] = useState("");
  const [summaryResult, setSummaryResult] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isRebuilding, setRebuilding] = useState(false);
  const [rebuildAnimation, setRebuildAnimation] = useState({
    running: false,
    frames: [],
    pointer: 0,
    message: "",
    status: "idle"
  });
  const fileInputRef = useRef(null);

  useEffect(() => {
    refreshFiles();
  }, []);

  useEffect(() => {
    setSelectedFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [category]);

  useEffect(() => {
    if (!rebuildAnimation.running || rebuildAnimation.frames.length === 0) {
      return;
    }
    const id = setInterval(() => {
      setRebuildAnimation((prev) => {
        if (!prev.running) {
          return prev;
        }
        const nextPointer = prev.pointer + 1;
        if (nextPointer >= prev.frames.length) {
          return {
            ...prev,
            running: false,
            pointer: prev.frames.length - 1,
            message: "Vector stores refreshed.",
            status: "success"
          };
        }
        return {
          ...prev,
          pointer: nextPointer,
          message: prev.frames[nextPointer]
        };
      });
    }, 900);
    return () => clearInterval(id);
  }, [rebuildAnimation.running, rebuildAnimation.frames.length]);

  async function refreshFiles() {
    try {
      const data = await listKnowledgeBase();
      setFiles(data.files || {});
    } catch (err) {
      setMessage({ text: err.message, tone: "danger" });
    }
  }

  function handleFileSelection(event) {
    const chosen = Array.from(event.target.files || []);
    setSelectedFiles(chosen);
  }

  function startRebuildAnimation(message) {
    setRebuildAnimation({
      running: false,
      frames: [],
      pointer: 0,
      message,
      status: "running"
    });
  }

  function animateRebuildResult(details, fallbackMessage) {
    const frames = [];
    Object.entries(details || {}).forEach(([cat, names]) => {
      (names || []).forEach((name) => {
        const label = CATEGORY_LABELS[cat] || cat;
        frames.push(`Embedding "${name}" into ${label}`);
      });
    });
    if (frames.length) {
      setRebuildAnimation({
        running: true,
        frames,
        pointer: 0,
        message: frames[0],
        status: "running"
      });
    } else {
      setRebuildAnimation({
        running: false,
        frames: [],
        pointer: 0,
        message: fallbackMessage,
        status: "success"
      });
    }
  }

  async function handleIngest() {
    if (!selectedFiles.length) {
      setMessage({ text: "Select at least one file before ingesting.", tone: "danger" });
      return;
    }
    const label = CATEGORY_LABELS[category] || category;
    setBusy(true);
    setMessage(null);
    setRebuilding(true);
    startRebuildAnimation(
      `Uploading ${selectedFiles.length === 1 ? "1 file" : `${selectedFiles.length} files`} to ${label}...`
    );
    try {
      const uploadPayload = await uploadKnowledgeBaseFile(selectedFiles, category);
      const uploadedNames = uploadPayload.filenames || selectedFiles.map((file) => file.name);
      const uploadedSummary = uploadedNames.join(", ");
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      await refreshFiles();
      const embeddingMessage =
        uploadedNames.length === 1
          ? `Embedding "${uploadedNames[0]}" into ${label}...`
          : `Embedding ${uploadedNames.length} files into ${label}...`;
      startRebuildAnimation(embeddingMessage);
      const rebuildResult = await rebuildKnowledgeBase([category]);
      const rebuildErrors = rebuildResult.errors || {};
      if (Object.keys(rebuildErrors).length) {
        const errorMsg = Object.entries(rebuildErrors)
          .map(([cat, err]) => `${CATEGORY_LABELS[cat] || cat}: ${err}`)
          .join("; ");
        setRebuildAnimation({
          running: false,
          frames: [],
          pointer: 0,
          message: "Embedding failed.",
          status: "error"
        });
        setMessage({
          text: `Uploaded ${uploadedSummary}, but embedding failed: ${errorMsg}`,
          tone: "danger"
        });
      } else {
        animateRebuildResult(
          rebuildResult.details,
          `Knowledge base for ${label} refreshed.`
        );
        setMessage({
          text: `Ingested ${uploadedSummary} into ${label}.`,
          tone: "success"
        });
      }
      await refreshFiles();
    } catch (err) {
      setMessage({ text: err.message, tone: "danger" });
      setRebuildAnimation({
        running: false,
        frames: [],
        pointer: 0,
        message: "Ingestion failed.",
        status: "error"
      });
    } finally {
      setBusy(false);
      setRebuilding(false);
    }
  }

  async function handleClear() {
    if (isBusy) return;
    const confirmed = window.confirm(
      "This will remove all uploaded knowledge base files and their embeddings. Continue?"
    );
    if (!confirmed) return;
    setBusy(true);
    setMessage(null);
    setRebuilding(true);
    startRebuildAnimation("Clearing knowledge base...");
    try {
      const result = await clearKnowledgeBase();
      const cleared = result.cleared || {};
      const errors = result.errors || {};
      const clearedSummary = Object.entries(cleared).map(([cat, stats]) => {
        const label = CATEGORY_LABELS[cat] || cat;
        const filesRemoved = stats?.files_removed ?? 0;
        const vectorsRemoved = stats?.vector_items_removed ?? 0;
        return `${label} (${filesRemoved} files, ${vectorsRemoved} vector entries)`;
      });
      if (Object.keys(errors).length) {
        const errorMsg = Object.entries(errors)
          .map(([cat, err]) => `${CATEGORY_LABELS[cat] || cat}: ${err}`)
          .join("; ");
        setRebuildAnimation({
          running: false,
          frames: [],
          pointer: 0,
          message: "Failed to clear one or more categories.",
          status: "error"
        });
        setMessage({
          text: clearedSummary.length
            ? `Cleared ${clearedSummary.join("; ")}, but issues occurred: ${errorMsg}`
            : `Clear operation encountered issues: ${errorMsg}`,
          tone: "danger"
        });
      } else {
        setMessage({
          text: clearedSummary.length
            ? `Cleared: ${clearedSummary.join("; ")}`
            : "Knowledge base already empty.",
          tone: "success"
        });
        setRebuildAnimation({
          running: false,
          frames: [],
          pointer: 0,
          message: "Knowledge base cleared.",
          status: "success"
        });
      }
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      await refreshFiles();
    } catch (err) {
      setMessage({ text: err.message, tone: "danger" });
      setRebuildAnimation({
        running: false,
        frames: [],
        pointer: 0,
        message: "Failed to clear knowledge base.",
        status: "error"
      });
    } finally {
      setBusy(false);
      setRebuilding(false);
    }
  }

  async function handleSummarize(ev) {
    ev.preventDefault();
    if (!summaryPrompt.trim()) return;
    setBusy(true);
    setSummaryResult(null);
    setMessage(null);
    try {
      const payload = await summarizePrompt(summaryPrompt.trim());
      setSummaryResult(payload);
      if (payload.error) {
        setMessage({ text: payload.error, tone: "danger" });
      }
    } catch (err) {
      setMessage({ text: err.message, tone: "danger" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Knowledge Base</h2>
      <p className="muted" style={{ marginBottom: "1.5rem" }}>
        Upload textbooks or lecture slides and ingest them to refresh the embeddings automatically.
      </p>

      {message && (
        <div
          className="chat-bubble"
          style={{
            borderColor: message.tone === "danger" ? "rgba(248,113,113,0.5)" : "rgba(52,211,153,0.35)",
            color: message.tone === "danger" ? "#fecaca" : "#dcfce7"
          }}
        >
          {message.text}
        </div>
      )}

      <div className="grid two" style={{ marginBottom: "2rem" }}>
        {Object.entries(CATEGORY_LABELS).map(([key, label]) => {
          const list = files[key] || [];
          return (
            <div key={key} className="card" style={{ marginBottom: 0 }}>
              <div className="inline-form" style={{ marginBottom: "0.75rem" }}>
                <span className="pill-badge">{label}</span>
                <span className="pill">
                  {list.length} {list.length === 1 ? "file" : "files"}
                </span>
              </div>
              <div className={`file-list${list.length > 3 ? " scrollable" : ""}`}>
                {list.length === 0 && <span className="muted">No files uploaded yet.</span>}
                {list.map((item) => (
                  <div key={item.path} className="file-item">
                    <div>
                      <strong>{item.filename}</strong>
                      <div className="list-muted">
                        {(Number(item.size_bytes) / 1024).toFixed(1)} KB · {CATEGORY_LABELS[key]}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div className="card" style={{ marginBottom: "2rem" }}>
        <h3>Upload Material</h3>
        <div className="inline-form upload-toolbar">
          <select
            className="upload-category-select"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            disabled={isBusy}
          >
            {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
          <label className="button secondary" style={{ cursor: "pointer" }}>
            <input
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.txt"
              style={{ display: "none" }}
              onChange={handleFileSelection}
              ref={fileInputRef}
              disabled={isBusy}
            />
            Select files
          </label>
          <button className="button" type="button" onClick={handleIngest} disabled={isBusy || selectedFiles.length === 0}>
            {isBusy ? "Working..." : "Ingest"}
          </button>
          <button className="button secondary" type="button" onClick={refreshFiles} disabled={isBusy}>
            Refresh list
          </button>
          <button
            className="button secondary"
            type="button"
            onClick={handleClear}
            disabled={isBusy}
            style={{
              borderColor: "rgba(248,113,113,0.6)",
              color: "#fecaca",
              backgroundColor: "rgba(248,113,113,0.12)"
            }}
          >
            Clear knowledge base
          </button>
        </div>
        {selectedFiles.length > 0 ? (
          <ul className="selected-file-list">
            {selectedFiles.map((file, idx) => (
              <li key={`${file.name}-${idx}`}>{file.name}</li>
            ))}
          </ul>
        ) : (
          <p className="list-muted">No files selected yet.</p>
        )}
        <p className="list-muted">Accepted formats: PDF, DOCX, TXT · Max 25 MB recommended.</p>
      </div>

      {(isRebuilding || rebuildAnimation.message) && (
        <div
          className={`vector-animation${
            rebuildAnimation.status === "error" ? " error" : ""
          }`}
        >
          {rebuildAnimation.status === "error" ? (
            <span className="warning-icon" aria-hidden="true">⚠</span>
          ) : rebuildAnimation.status === "success" ? (
            <span className="checkmark" aria-hidden="true">✔</span>
          ) : (
            <span className="spinner" aria-hidden="true" />
          )}
          <span>{rebuildAnimation.message}</span>
        </div>
      )}

      <div className="card">
        <h3>Quick Summaries</h3>
        <form onSubmit={handleSummarize}>
          <textarea
            className="textarea-small"
            placeholder="E.g. Summarize page 12 from Network Security Essentials."
            value={summaryPrompt}
            onChange={(e) => setSummaryPrompt(e.target.value)}
            disabled={isBusy}
          />
          <button className="button" type="submit" disabled={isBusy || !summaryPrompt.trim()}>
            {isBusy ? "Working..." : "Summarize"}
          </button>
        </form>
        {summaryResult && !summaryResult.error && (
          <div className="chat-bubble" style={{ marginTop: "1rem" }}>
            <h4>Summary</h4>
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{summaryResult.summary}</div>
            {summaryResult.citation && (
              <div className="chat-source" style={{ marginTop: "1rem" }}>
                <strong>{summaryResult.citation.filename}</strong>
                {summaryResult.citation.page_number ? (
                  <span> · page {summaryResult.citation.page_number}</span>
                ) : null}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
