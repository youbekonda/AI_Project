/** CampusAsk API client. Point at Railway/Render/Fly via NEXT_PUBLIC_API_URL. */

export type AcademicLevel = "all" | "undergraduate" | "graduate";

export type SourceCard = {
  /** 1-based number matching the [n] markers in the answer text. */
  ref?: number;
  page?: number | null;
  section?: string | null;
  chunk_ids?: string[];
  title?: string;
  location?: string;
  url?: string;
  excerpt?: string;
  level?: string;
  year?: string;
  score?: number;
  chunk_id?: string;
};

export type AskResponse = {
  status: "answered" | "abstained" | "conversational" | "error" | string;
  answer: string;
  sources: SourceCard[];
  cited_ids: string[];
  invalid_citation_ids: string[];
  max_score: number | null;
  error: string | null;
  intent: string | null;
  suggested_academic_level?: "undergraduate" | "graduate" | null;
  /** Optional alias some API builds may return for the synced level dropdown. */
  academic_level?: "undergraduate" | "graduate" | "all" | null;
};

export type IdentityResponse = {
  ok: boolean;
  full_name: string | null;
  first_name: string | null;
  masked_id: string | null;
  greeting: string | null;
  errors: Record<string, string>;
};

export type SampleQuestion = {
  id: string;
  label: string;
  question: string;
};

export type SampleGroup = {
  topic: string;
  questions: SampleQuestion[];
};

export const SAMPLE_GROUPS: SampleGroup[] = [
  {
    topic: "Academic standing",
    questions: [
      {
        id: "probation",
        label: "Academic probation GPA",
        question: "What GPA places an undergraduate student on academic probation?",
      },
      {
        id: "probation-exit",
        label: "Getting off probation",
        question: "How does an undergraduate get removed from academic probation?",
      },
    ],
  },
  {
    topic: "Enrollment & credit hours",
    questions: [
      {
        id: "credits",
        label: "Full-time credit hours",
        question: "How many credit hours are required for full-time graduate enrollment?",
      },
      {
        id: "max-load",
        label: "Maximum course load",
        question: "What is the maximum number of credit hours an undergraduate can take in a semester?",
      },
    ],
  },
  {
    topic: "Graduate study",
    questions: [
      {
        id: "thesis",
        label: "Thesis vs non-thesis",
        question: "What is the difference between a thesis and non-thesis master's option?",
      },
      {
        id: "transfer",
        label: "Transfer credit",
        question: "How many transfer credit hours can count toward a master's degree?",
      },
    ],
  },
];

export const SAMPLE_QUESTIONS: SampleQuestion[] = SAMPLE_GROUPS.flatMap((g) => g.questions);

export const CAPTION =
  "Official UCM policy answers, grounded in the catalogs.";

function apiBase(): string {
  const raw = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").trim();
  return raw.replace(/\/$/, "");
}

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`API returned non-JSON (${res.status})`);
  }
  if (!res.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return data as T;
}

export async function validateIdentity(
  fullName: string,
  ucmId: string,
): Promise<IdentityResponse> {
  const res = await fetch(`${apiBase()}/v1/identity/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ full_name: fullName, ucm_id: ucmId }),
  });
  return parseJson<IdentityResponse>(res);
}

export async function askQuestion(input: {
  question: string;
  academicLevel: AcademicLevel;
  firstName?: string | null;
}): Promise<AskResponse> {
  const res = await fetch(`${apiBase()}/v1/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: input.question,
      academic_level: input.academicLevel,
      first_name: input.firstName || undefined,
    }),
  });
  return parseJson<AskResponse>(res);
}

export type StreamStatus = { stage: "searching" | "writing" | string; passages?: number };

export type StreamHandlers = {
  onStatus?: (status: StreamStatus) => void;
  /** A chunk of answer text (may contain [n] citation markers). */
  onDelta?: (text: string) => void;
};

/** Thrown when the API predates /v1/ask/stream; callers fall back to askQuestion. */
export class StreamUnavailableError extends Error {}

/**
 * Ask with Server-Sent Events: status after retrieval, answer text as it is
 * written, then the final AskResponse (authoritative — replaces streamed text).
 */
export async function askQuestionStream(
  input: { question: string; academicLevel: AcademicLevel; firstName?: string | null },
  handlers: StreamHandlers = {},
): Promise<AskResponse> {
  const res = await fetch(`${apiBase()}/v1/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      question: input.question,
      academic_level: input.academicLevel,
      first_name: input.firstName || undefined,
    }),
  });
  if (res.status === 404 || res.status === 405) throw new StreamUnavailableError();
  if (!res.ok || !res.body) return parseJson<AskResponse>(res);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let final: AskResponse | null = null;

  const dispatch = (block: string) => {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) return;
    const payload = JSON.parse(data);
    if (event === "status") handlers.onStatus?.(payload as StreamStatus);
    else if (event === "delta") handlers.onDelta?.((payload as { text: string }).text);
    else if (event === "done") final = payload as AskResponse;
    else if (event === "error") throw new Error((payload as { message?: string }).message || "CampusAsk ask failed.");
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      dispatch(buffer.slice(0, split));
      buffer = buffer.slice(split + 2);
      split = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim()) dispatch(buffer);
  if (!final) throw new Error("The answer stream ended before the answer was complete.");
  return final;
}

export async function checkHealth(): Promise<{
  status: string;
  index_loaded?: boolean;
  llm_configured?: boolean;
}> {
  const res = await fetch(`${apiBase()}/health`, { cache: "no-store" });
  return parseJson(res);
}
