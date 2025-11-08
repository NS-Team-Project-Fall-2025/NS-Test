import { useEffect, useState } from "react";
import {
  generateQuiz,
  gradeQuiz,
  listQuizHistory,
  getQuizAttempt,
  deleteQuizAttemptApi
} from "@/lib/api";

const SOURCE_LABELS = {
  textbooks: "Textbooks",
  slides: "Lecture Slides"
};

const initialConfig = {
  num_mcq: 2,
  num_true_false: 2,
  num_open_ended: 1,
  sourceCategory: "both",
  mode: "random",
  difficulty: "medium",
  topics: ""
};

export default function QuizPage() {
  const [config, setConfig] = useState(initialConfig);
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});
  const [grading, setGrading] = useState(false);
  const [isGenerating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    refreshHistory();
  }, []);

  const formatDifficulty = (value) => value ? value.charAt(0).toUpperCase() + value.slice(1) : "";

  function updateConfig(field, value) {
    setConfig((prev) => ({
      ...prev,
      [field]: value
    }));
  }

  async function refreshHistory() {
    try {
      const payload = await listQuizHistory();
      setHistory(payload.attempts || []);
    } catch (err) {
      setMessage({
        text: err.message,
        tone: "danger"
      });
    }
  }

  async function handleGenerate(event) {
    event.preventDefault();
    if (isGenerating) return;
    setGenerating(true);
    setMessage(null);
    setQuiz(null);
    setResult(null);
    setAnswers({});
    try {
      const sourceCategories = config.sourceCategory === "both" 
        ? ["textbooks", "slides"]
        : [config.sourceCategory];
      
      const payload = await generateQuiz({
        num_mcq: Number(config.num_mcq),
        num_true_false: Number(config.num_true_false),
        num_open_ended: Number(config.num_open_ended),
        mode: config.mode,
        difficulty: config.difficulty,
        source_categories: sourceCategories,
        topics: config.topics ? config.topics.split(",").map(topic => topic.trim()) : undefined
      });
      setQuiz(payload);
    } catch (err) {
      setMessage({
        text: err.message,
        tone: "danger"
      });
    } finally {
      setGenerating(false);
    }
  }

  function updateAnswer(qid, value) {
    setAnswers((prev) => ({
      ...prev,
      [qid]: value
    }));
  }

  async function handleGrade(event) {
    event.preventDefault();
    if (!quiz) return;
    setGrading(true);
    setMessage(null);
    try {
      const payload = await gradeQuiz(quiz, answers);
      setResult(payload.results);
      setQuiz(payload.attempt?.quiz_data || quiz);
      refreshHistory();
    } catch (err) {
      setMessage({
        text: err.message,
        tone: "danger"
      });
    } finally {
      setGrading(false);
    }
  }

  async function handleHistorySelect(id) {
    try {
      const attempt = await getQuizAttempt(id);
      if (attempt.error) {
        setMessage({
          text: attempt.error,
          tone: "danger"
        });
        return;
      }
      setQuiz(attempt.quiz_data);
      setAnswers(attempt.user_answers || {});
      setResult(attempt.results || null);
      setMessage({
        text: `Loaded quiz attempt "${attempt.title || id}"`,
        tone: "success"
      });
    } catch (err) {
      setMessage({
        text: err.message,
        tone: "danger"
      });
    }
  }

  async function handleHistoryDelete(id) {
    try {
      await deleteQuizAttemptApi(id);
      refreshHistory();
      setMessage({
        text: "Quiz attempt deleted.",
        tone: "success"
      });
    } catch (err) {
      setMessage({
        text: err.message,
        tone: "danger"
      });
    }
  }

  return (
    <div className="card">
      <h2>Quiz Lab</h2>
      <p className="muted" style={{ marginBottom: "1.5rem" }}>
        Generate quizzes grounded in your knowledge base. Grade responses to reveal precise feedback with citations.
      </p>

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

      <form className="grid two" onSubmit={handleGenerate} style={{ marginBottom: "2rem" }}>
        <div className="card" style={{ margin: 0 }}>
          <label>Multiple-choice</label>
          <input
            className="input"
            type="number"
            min="0"
            value={config.num_mcq}
            onChange={(e) => updateConfig("num_mcq", e.target.value)}
            disabled={isGenerating}
          />
          <label>True / False</label>
          <input
            className="input"
            type="number"
            min="0"
            value={config.num_true_false}
            onChange={(e) => updateConfig("num_true_false", e.target.value)}
            disabled={isGenerating}
          />
          <label>Open-ended</label>
          <input
            className="input"
            type="number"
            min="0"
            value={config.num_open_ended}
            onChange={(e) => updateConfig("num_open_ended", e.target.value)}
            disabled={isGenerating}
          />
          <label>Knowledge Source</label>
          <div className="radio-group">
            {[
              { value: "both", label: "Textbooks + Lecture Slides" },
              { value: "textbooks", label: "Textbooks only" },
              { value: "slides", label: "Lecture Slides only" }
            ].map((option) => (
              <label key={option.value}>
                <input
                  type="radio"
                  name="sourceCategory"
                  value={option.value}
                  checked={config.sourceCategory === option.value}
                  onChange={(e) => updateConfig("sourceCategory", e.target.value)}
                  disabled={isGenerating}
                />
                {" "}{option.label}
              </label>
            ))}
          </div>
        </div>

        <div className="card" style={{ margin: 0 }}>
          <label>Mode</label>
          <select
            value={config.mode}
            onChange={(e) => updateConfig("mode", e.target.value)}
            disabled={isGenerating}
          >
            <option value="random">Random</option>
            <option value="custom">Custom topics</option>
          </select>
          <label>Difficulty</label>
          <select
            value={config.difficulty}
            onChange={(e) => updateConfig("difficulty", e.target.value)}
            disabled={isGenerating}
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
          <label>Topics (comma separated)</label>
          <textarea
            className="textarea-small"
            placeholder="E.g. symmetric encryption, TLS handshake"
            value={config.topics}
            onChange={(e) => updateConfig("topics", e.target.value)}
            disabled={config.mode !== "custom" || isGenerating}
          />
          <button className="button" type="submit" disabled={isGenerating}>
            {isGenerating ? "Generating..." : "Generate quiz"}
          </button>
        </div>
      </form>

      {isGenerating && (
        <div className="loading-indicator">
          <span className="spinner" aria-hidden="true" />
          <span>Generating quiz... this may take a few seconds.</span>
        </div>
      )}

      {quiz && (
        <form onSubmit={handleGrade}>
          <div className="card" style={{ marginBottom: "2rem" }}>
            <h3>{quiz.title}</h3>
            <div className="inline-form" style={{ gap: "0.5rem", marginBottom: "0.75rem" }}>
              {quiz.difficulty && (
                <span className="pill-badge">
                  Difficulty: {formatDifficulty(quiz.difficulty)}
                </span>
              )}
              {quiz.source_categories?.length ? (
                <span className="pill-badge">
                  Sources: {quiz.source_categories.map(cat => SOURCE_LABELS[cat] || cat).join(", ")}
                </span>
              ) : null}
            </div>
            {quiz.topics?.length ? (
              <div className="pill" style={{ marginBottom: "1rem" }}>
                Topics: {quiz.topics.join(", ")}
              </div>
            ) : null}
            {(quiz.questions || []).map((question) => (
              <div key={question.id} className="quiz-question">
                <h3>
                  {question.id}. {question.question}
                </h3>
                {question.type === "multiple_choice" && (
                  <div className="radio-group">
                    {question.choices.map((choice, index) => {
                      const [label, text] = choice.split(".", 2);
                      const optionValue = (label || "").trim() || String.fromCharCode(65 + index);
                      return (
                        <label key={optionValue}>
                          <input
                            type="radio"
                            name={question.id}
                            value={optionValue.trim()}
                            checked={answers[question.id] === optionValue.trim()}
                            onChange={(e) => updateAnswer(question.id, e.target.value)}
                          />
                          {" "}{text ? `${optionValue}. ${text.trim()}` : choice}
                        </label>
                      );
                    })}
                  </div>
                )}
                {question.type === "true_false" && (
                  <div className="radio-group">
                    {["True", "False"].map((option) => (
                      <label key={option}>
                        <input
                          type="radio"
                          name={question.id}
                          value={option}
                          checked={answers[question.id] === option}
                          onChange={(e) => updateAnswer(question.id, e.target.value)}
                        />
                        {" "}{option}
                      </label>
                    ))}
                  </div>
                )}
                {question.type === "open_ended" && (
                  <textarea
                    className="textarea-small"
                    value={answers[question.id] || ""}
                    onChange={(e) => updateAnswer(question.id, e.target.value)}
                  />
                )}
              </div>
            ))}
            <button className="button" type="submit" disabled={grading || isGenerating}>
              {grading ? "Grading..." : "Grade quiz"}
            </button>
          </div>
        </form>
      )}

      {result && (
        <div className="card">
          <h3>
            Score: {result.earned_score.toFixed(1)} / {result.max_score} ({result.percentage.toFixed(1)}%)
          </h3>
          {(result.questions || []).map((item, idx) => (
            <div key={idx} className="quiz-question">
              <div className="inline-form" style={{ marginBottom: "0.5rem" }}>
                <span className="pill-badge">
                  {item.is_correct ? "Correct" : "Needs review"}
                </span>
              </div>
              <strong>{item.question?.question}</strong>
              <div className="list-muted">
                Your answer: {item.user_answer_display || "—"}
              </div>
              <div className="list-muted">
                Expected: {item.correct_answer_display || "—"}
              </div>
              {item.feedback && (
                <div style={{ marginTop: "0.75rem" }}>
                  {item.feedback}
                </div>
              )}
              {(item.citations || []).length > 0 && (
                <div className="chat-sources" style={{ marginTop: "0.75rem" }}>
                  {item.citations.map((citation, idx2) => (
                    <div key={idx2} className="chat-source">
                      <strong>{citation.filename || "Source"}</strong>
                      {citation.page_number ? <span> · page {citation.page_number}</span> : null}
                      {citation.snippet ? (
                        <div className="list-muted" style={{ marginTop: "0.25rem" }}>
                          {citation.snippet}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <h3>History</h3>
        {history.length === 0 && <p className="muted">No graded quizzes stored yet.</p>}
        {history.map((attempt) => (
          <div key={attempt.quiz_id} className="file-item">
            <div>
              <strong>{attempt.title || attempt.quiz_id}</strong>
              <div className="list-muted">
                {attempt.topics?.length ? `Topics: ${attempt.topics.join(", ")}` : "Random topics"}
              </div>
              {attempt.difficulty && (
                <div className="list-muted">
                  Difficulty: {formatDifficulty(attempt.difficulty)}
                </div>
              )}
              {attempt.source_categories?.length ? (
                <div className="list-muted">
                  Sources: {attempt.source_categories.map(cat => SOURCE_LABELS[cat] || cat).join(", ")}
                </div>
              ) : null}
              <div className="list-muted">
                Score: {typeof attempt.percentage === "number" ? `${attempt.percentage.toFixed(1)}%` : "—"}
              </div>
            </div>
            <div className="inline-form" style={{ marginBottom: 0 }}>
              <button className="button secondary" type="button" onClick={() => handleHistorySelect(attempt.quiz_id)}>
                View
              </button>
              <button className="button secondary" type="button" onClick={() => handleHistoryDelete(attempt.quiz_id)}>
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

