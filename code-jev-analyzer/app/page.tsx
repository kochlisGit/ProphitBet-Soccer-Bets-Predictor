"use client";

import React, { useState, useEffect, useRef } from "react";

interface AnswerItem {
  type: string;
  probability?: number;
  score?: number;
}

interface SummaryItem {
  question: string;
  probability: number;
  flagged: boolean | "review";
  description: string;
}

interface EvaluationData {
  answers?: Record<string, AnswerItem>;
  summary?: SummaryItem[];
  gatewayNotice?: string;
  error?: string;
}

interface HistoryItem {
  id: string;
  time: string;
  filename: string;
  riskScore: number;
  flaggedCount: number;
  reviewCount: number;
}

const SAMPLE_FILES = [
  {
    name: "vulnerable_api.ts",
    code: `import { NextRequest } from "next/server";
import { db } from "./db";

const STRIPE_SECRET = "sk_live_98127398127391823719823";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const userId = body.userId;
  
  // Unsanitized SQL query
  const user = await db.query("SELECT * FROM users WHERE id = " + userId);
  
  // Unhandled external webhook
  const response = await fetch("https://api.payments.com/charge", {
    method: "POST",
    body: JSON.stringify({ userId, amount: body.amount })
  });
  
  return Response.json({ success: true, user });
}`,
  },
  {
    name: "borderline_parser.ts",
    code: `export function parseWebhookPayload(rawPayload: any) {
  // Input received without strict schema validation
  const event = JSON.parse(rawPayload);
  const eventType = event.type;
  
  if (eventType === "user.signup") {
    return { id: event.data.id, email: event.data.email };
  }
  
  return null;
}`,
  },
  {
    name: "clean_service.ts",
    code: `import { z } from "zod";

const UserSchema = z.object({
  id: z.string().uuid(),
  role: z.enum(["admin", "user"]),
});

export async function verifyUser(input: unknown) {
  try {
    const validated = UserSchema.parse(input);
    return { success: true, data: validated };
  } catch (error) {
    console.error("Validation error:", error);
    return { success: false, error: "Invalid user data" };
  }
}`,
  },
];

export default function JevWorkbench() {
  const [selectedFile, setSelectedFile] = useState(SAMPLE_FILES[0].name);
  const [code, setCode] = useState(SAMPLE_FILES[0].code);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState<EvaluationData | null>(null);
  const [continuousLoop, setContinuousLoop] = useState(false);
  const [loopInterval, setLoopInterval] = useState(4); // seconds
  const [appliedChanges, setAppliedChanges] = useState<string[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [evalCount, setEvalCount] = useState(0);
  const [statusMessage, setStatusMessage] = useState("Ready");

  const loopTimerRef = useRef<NodeJS.Timeout | null>(null);
  const codeRef = useRef(code);
  codeRef.current = code;

  // Run evaluation
  const runEvaluation = async (currentCode = code, filename = selectedFile) => {
    setIsEvaluating(true);
    setStatusMessage("Jev evaluating code...");
    try {
      const res = await fetch("/api/analyze-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: currentCode, filename }),
      });
      const data: EvaluationData = await res.json();
      if (!res.ok || !data.answers) throw new Error(data.error || 'Evaluation unavailable');
      setEvalResult(data);
      setEvalCount((prev) => prev + 1);

      const riskScore = data.answers?.riskLevel?.score ?? 0;
      const flaggedCount = data.summary?.filter((s) => s.flagged === true).length ?? 0;
      const reviewCount = data.summary?.filter((s) => s.flagged === "review").length ?? 0;

      const newHistoryItem: HistoryItem = {
        id: Math.random().toString(36).substring(7),
        time: new Date().toLocaleTimeString(),
        filename,
        riskScore,
        flaggedCount,
        reviewCount,
      };

      setHistory((prev) => [newHistoryItem, ...prev.slice(0, 9)]);
      setStatusMessage("Evaluation complete");
    } catch (err: any) {
      console.error(err);
      setEvalResult(null);
      setContinuousLoop(false);
      setStatusMessage(err?.message || "Evaluation unavailable — not validated");
    } finally {
      setIsEvaluating(false);
    }
  };

  // Regex rewriting is not a safe automated remediation engine.
  const runAutoFix = async () => {
    setStatusMessage("Automatic fixes disabled; review findings and test changes manually.");
  };

  // Continuous evaluation loop effect
  useEffect(() => {
    if (continuousLoop) {
      runEvaluation(codeRef.current, selectedFile);
      loopTimerRef.current = setInterval(() => {
        runEvaluation(codeRef.current, selectedFile);
      }, loopInterval * 1000);
    } else {
      if (loopTimerRef.current) {
        clearInterval(loopTimerRef.current);
        loopTimerRef.current = null;
      }
    }

    return () => {
      if (loopTimerRef.current) {
        clearInterval(loopTimerRef.current);
      }
    };
  }, [continuousLoop, loopInterval, selectedFile]);

  // Initial evaluation on mount
  useEffect(() => {
    runEvaluation(SAMPLE_FILES[0].code, SAMPLE_FILES[0].name);
  }, []);

  const handleSelectSample = (sample: (typeof SAMPLE_FILES)[0]) => {
    setSelectedFile(sample.name);
    setCode(sample.code);
    setAppliedChanges([]);
    runEvaluation(sample.code, sample.name);
  };

  const riskScore = evalResult?.answers?.riskLevel?.score ?? 0;
  const flaggedCount = evalResult?.summary?.filter((s) => s.flagged === true).length ?? 0;
  const reviewCount = evalResult?.summary?.filter((s) => s.flagged === "review").length ?? 0;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="logo-badge">⚡</div>
          <div className="brand-text">
            <h1>TypeSafe AI Jev — Code Quality & Security Workbench</h1>
            <p>Server-side continuous code analysis & automated remediation engine</p>
          </div>
        </div>

        <div className="header-status">
          <button
            className={`loop-toggle-btn ${continuousLoop ? "active" : ""}`}
            onClick={() => setContinuousLoop(!continuousLoop)}
          >
            <span className={`pulse-dot ${continuousLoop ? "pulse" : ""}`} />
            {continuousLoop ? `Continuous Loop Active (${loopInterval}s)` : "Start Continuous Loop"}
          </button>
        </div>
      </header>

      {/* Gateway Notice / Status */}
      {evalResult?.gatewayNotice && (
        <div className="notice-banner">
          <span>ℹ️ {evalResult.gatewayNotice}</span>
          <span style={{ opacity: 0.8, fontSize: "11px" }}>Vercel AI Gateway $5/mo Tier</span>
        </div>
      )}

      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <span className="label">Total Evaluated</span>
          <div className="value">
            {evalCount} <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>runs</span>
          </div>
        </div>
        <div className="stat-card">
          <span className="label">Overall Risk Level</span>
          <div className="value" style={{ color: riskScore >= 3 ? "#f87171" : riskScore > 0 ? "#fbbf24" : "#34d399" }}>
            {!evalResult?.answers ? "Not evaluated" : riskScore === 0 ? "Trivial (0)" : riskScore === 1 ? "Minor (1)" : riskScore === 2 ? "Notable (2)" : "Severe (3)"}
          </div>
        </div>
        <div className="stat-card">
          <span className="label">Critical Issues Flagged</span>
          <div className="value" style={{ color: flaggedCount > 0 ? "#f87171" : "var(--text-main)" }}>
            {flaggedCount}
          </div>
        </div>
        <div className="stat-card">
          <span className="label">Needs Human Review</span>
          <div className="value" style={{ color: reviewCount > 0 ? "#fbbf24" : "var(--text-main)" }}>
            {reviewCount}
          </div>
        </div>
      </div>

      {/* Workbench Panels */}
      <div className="workbench-grid">
        {/* Left Panel: Code Editor */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>📝 Source Code Editor</span>
            </div>
            <div className="file-selector">
              {SAMPLE_FILES.map((sample) => (
                <button
                  key={sample.name}
                  className={`file-tab ${selectedFile === sample.name ? "active" : ""}`}
                  onClick={() => handleSelectSample(sample)}
                >
                  {sample.name}
                </button>
              ))}
            </div>
          </div>

          <div className="editor-wrapper">
            <textarea
              className="code-textarea"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              spellCheck={false}
              placeholder="Paste or edit code here..."
            />
          </div>

          <div className="panel-footer">
            <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
              {statusMessage}
            </span>
            <div style={{ display: "flex", gap: "10px" }}>
              <button
                className="btn btn-primary"
                onClick={() => runEvaluation(code, selectedFile)}
                disabled={isEvaluating}
              >
                {isEvaluating ? "Evaluating..." : "🔍 Run Jev Evaluation"}
              </button>
              {flaggedCount > 0 && (
                <button className="btn btn-success" onClick={runAutoFix}>
                  Review remediation requirements
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel: Evaluation Telemetry */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>📊 Real-Time Jev Evaluation Telemetry</span>
            </div>
            {isEvaluating && (
              <span style={{ fontSize: "11px", color: "#6366f1" }}>Analyzing...</span>
            )}
          </div>

          <div className="eval-content">
            {/* Risk Level Banner */}
            <div className={`risk-level-banner risk-score-${riskScore}`}>
              <div>
                <strong>Risk Rubric: </strong>
                {!evalResult?.answers ? "Not evaluated — no correctness claim" : riskScore === 0
                  ? "Trivial / Clean code"
                  : riskScore === 1
                  ? "Minor concern"
                  : riskScore === 2
                  ? "Notable risk (could break features)"
                  : "Severe risk (could compromise prod or leak data)"}
              </div>
              <span className="badge" style={{ background: "rgba(0,0,0,0.2)" }}>
                Score: {evalResult?.answers ? `${riskScore}/3` : "Unavailable"}
              </span>
            </div>

            {/* Questions Checklist */}
            <div className="questions-list">
              {[
                { key: "hardcodedSecret", label: "Hardcoded Credential / Secret" },
                { key: "sqlInjectionRisk", label: "SQL Injection Risk" },
                { key: "unhandledError", label: "Unhandled Error Path" },
                { key: "missingInputValidation", label: "Missing Input Validation" },
                { key: "leagueIsolationRisk", label: "League / Country Isolation Risk" },
                { key: "unverifiedDataRisk", label: "Unverified Football Data Risk" },
              ].map(({ key, label }) => {
                const answer = evalResult?.answers?.[key];
                const prob = answer?.probability ?? 0;
                const isFlagged = prob >= 0.8;
                const isReview = prob >= 0.4 && prob < 0.8;

                let badgeClass = "badge-success";
                let badgeText = "Clean";
                let barColor = "#10b981";

                if (!answer) {
                  badgeClass = "badge-warning"; badgeText = "Not evaluated"; barColor = "#71717a";
                } else if (isFlagged) {
                  badgeClass = "badge-danger";
                  badgeText = "Flagged (>=80%)";
                  barColor = "#ef4444";
                } else if (isReview) {
                  badgeClass = "badge-warning";
                  badgeText = "Review (40-80%)";
                  barColor = "#f59e0b";
                }

                return (
                  <div key={key} className="question-card">
                    <div className="question-header">
                      <span className="question-name">{label}</span>
                      <span className={`badge ${badgeClass}`}>{badgeText}</span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-bar"
                        style={{
                          width: `${Math.round(prob * 100)}%`,
                          background: barColor,
                        }}
                      />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)" }}>
                      <span>Model Probability</span>
                      <strong style={{ color: "var(--text-main)" }}>{answer ? `${(prob * 100).toFixed(1)}%` : "Unavailable"}</strong>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Auto Fix Changes Box */}
            {appliedChanges.length > 0 && (
              <div className="changes-box">
                <h4>✅ Auto-Remediations Applied:</h4>
                <ul>
                  {appliedChanges.map((change, idx) => (
                    <li key={idx}>{change}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Continuous Loop Evaluation History Feed */}
      <div className="activity-feed">
        <div className="feed-header">
          <h3>⏱️ Continuous Evaluation Activity Log</h3>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
            Updated in real-time
          </span>
        </div>
        <table className="feed-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Target File</th>
              <th>Risk Score</th>
              <th>Flagged Issues</th>
              <th>Review Status</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ textAlign: "center", color: "var(--text-muted)" }}>
                  No evaluations logged yet.
                </td>
              </tr>
            ) : (
              history.map((item) => (
                <tr key={item.id}>
                  <td style={{ fontFamily: "monospace" }}>{item.time}</td>
                  <td><code>{item.filename}</code></td>
                  <td>
                    <span
                      style={{
                        color:
                          item.riskScore >= 3
                            ? "#f87171"
                            : item.riskScore > 0
                            ? "#fbbf24"
                            : "#34d399",
                      }}
                    >
                      Score {item.riskScore}/3
                    </span>
                  </td>
                  <td>
                    {item.flaggedCount > 0 ? (
                      <span className="badge badge-danger">{item.flaggedCount} Flagged</span>
                    ) : (
                      <span className="badge badge-success">0</span>
                    )}
                  </td>
                  <td>
                    {item.reviewCount > 0 ? (
                      <span className="badge badge-warning">{item.reviewCount} Need Review</span>
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>None</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
