import type { JobSnapshot } from "../types/job";

export interface JobState {
  snapshot: JobSnapshot | null;
  uploadState: "idle" | "uploading" | "error";
  error: string | null;
}

export const initialJobState: JobState = {
  snapshot: null,
  uploadState: "idle",
  error: null,
};

export type JobAction =
  | { type: "upload-start" }
  | { type: "snapshot"; snapshot: JobSnapshot }
  | { type: "error"; message: string }
  | { type: "reset" };

export function jobReducer(state: JobState, action: JobAction): JobState {
  switch (action.type) {
    case "upload-start":
      return { ...state, uploadState: "uploading", error: null };
    case "snapshot":
      return { ...state, snapshot: action.snapshot, uploadState: "idle", error: null };
    case "error":
      return { ...state, uploadState: "error", error: action.message };
    case "reset":
      return initialJobState;
    default:
      return state;
  }
}

export function shouldPollJob(snapshot: JobSnapshot): boolean {
  return !["complete", "blocked", "failed", "stopped"].includes(snapshot.status);
}
