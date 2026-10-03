"use client";

import { useEffect, useRef, useState } from "react";
import { apiFetch } from "@/lib/api";

export type TaskProgressData = {
  job_id: string;
  status: string;
  current: number;
  total: number;
  percent: number;
  phase: string;
  item?: string | null;
  result?: Record<string, unknown> | null;
  error?: string | null;
};

type Props = {
  jobId: string;
  token: string;
  label: string;
  statusPath?: string;
  onFinished?: (progress: TaskProgressData) => void;
};

export default function TaskProgress({
  jobId,
  token,
  label,
  statusPath,
  onFinished,
}: Props) {
  const [progress, setProgress] = useState<TaskProgressData>({
    job_id: jobId,
    status: "PENDING",
    current: 0,
    total: 0,
    percent: 0,
    phase: "Queued",
  });
  const onFinishedRef = useRef(onFinished);
  onFinishedRef.current = onFinished;

  useEffect(() => {
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const next = await apiFetch<TaskProgressData>(statusPath || `/admin/jobs/${jobId}`, { token });
        if (cancelled) return;
        setProgress(next);
        if (["SUCCESS", "FAILURE", "REVOKED"].includes(next.status)) {
          onFinishedRef.current?.(next);
          return;
        }
      } catch (error: any) {
        if (cancelled) return;
        setProgress((old) => ({ ...old, phase: `Progress check failed: ${error.message}` }));
      }
      timeout = setTimeout(poll, 1200);
    };

    poll();
    return () => {
      cancelled = true;
      if (timeout) clearTimeout(timeout);
    };
  }, [jobId, token, statusPath]);

  const done = progress.status === "SUCCESS";
  const failed = ["FAILURE", "REVOKED"].includes(progress.status);
  const width = done ? 100 : Math.max(0, Math.min(100, progress.percent || 0));
  const resultText = progress.result
    ? Object.entries(progress.result).map(([key, value]) => `${key.replaceAll("_", " ")}: ${String(value)}`).join(" · ")
    : null;

  return (
    <div className={`mt-4 rounded-xl border p-4 ${failed ? "border-red-500/30 bg-red-950/20" : done ? "border-emerald-500/30 bg-emerald-950/20" : "border-blue-500/30 bg-blue-950/20"}`}>
      <div className="mb-2 flex items-center justify-between gap-4 text-sm">
        <span className="font-medium text-white">{label}</span>
        <span className={failed ? "text-red-400" : done ? "text-emerald-400" : "text-blue-300"}>
          {done ? "Complete" : failed ? "Failed" : progress.total > 0 ? `${width}%` : "Starting…"}
        </span>
      </div>
      <div
        className="h-2.5 overflow-hidden rounded-full bg-gray-800"
        role="progressbar"
        aria-label={`${label} progress`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={width}
      >
        <div
          className={`h-full rounded-full transition-[width] duration-500 ${failed ? "bg-red-500" : done ? "bg-emerald-500" : "bg-blue-500"} ${progress.total === 0 && !done && !failed ? "w-1/3 animate-pulse" : ""}`}
          style={progress.total > 0 || done || failed ? { width: `${width}%` } : undefined}
        />
      </div>
      <div className="mt-2 flex flex-wrap justify-between gap-x-4 gap-y-1 text-xs text-gray-400">
        <span>{failed ? progress.error || progress.phase : progress.phase}</span>
        {progress.total > 0 && <span>{Math.min(progress.current, progress.total)} / {progress.total}</span>}
      </div>
      {progress.item && !done && !failed && <p className="mt-1 truncate text-xs text-gray-500">{progress.item}</p>}
      {done && resultText && <p className="mt-2 text-xs text-emerald-300">{resultText}</p>}
    </div>
  );
}
