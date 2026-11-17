"use client";

import Image from "next/image";
import Link from "next/link";
import {
  FormEvent,
  KeyboardEvent as ReactKeyboardEvent,
  ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  AcademicLevel,
  AskResponse,
  SAMPLE_GROUPS,
  SourceCard,
  StreamUnavailableError,
  askQuestion,
  askQuestionStream,
  validateIdentity,
} from "@/lib/api";
import {
  StudentSession,
  clearSession,
  loadSession,
  markGreeted,
  saveSession,
  setSessionAcademicLevel,
} from "@/lib/session";
import SiteHeader from "@/components/SiteHeader";
import styles from "./ask.module.css";

type TurnStatus =
  | "pending"
  | "streaming"
  | "answered"
  | "abstained"
  | "error"
  | "conversational";

type Turn = {
  id: string;
  question: string;
  level: AcademicLevel;
  status: TurnStatus;
  answer?: string;
  sources?: SourceCard[];
  /** Passages found by retrieval, reported before the answer is written. */
  passagesFound?: number;
};

const LEVELS: { value: AcademicLevel; label: string; scope: string }[] = [
  { value: "all", label: "All catalogs", scope: "Undergraduate and Graduate Catalogs" },
  { value: "undergraduate", label: "Undergraduate", scope: "Undergraduate Catalog" },
  { value: "graduate", label: "Graduate", scope: "Graduate Catalog" },
];

/** Closest passages shown when the catalogs don't answer a question. */
const ABSTAIN_PREVIEW = 3;

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function scopeLabel(level: AcademicLevel) {
  return LEVELS.find((l) => l.value === level)?.scope ?? "catalogs";
}

function normalizeStatus(status: string): TurnStatus {
  if (status === "answered" || status === "abstained" || status === "conversational") {
    return status;
  }
  return "error";
}

function shortTitle(title?: string) {
  if (!title) return "Catalog source";
  return title.replace(/^UCM\s+/i, "").replace(/(\d{4})-(\d{4})/, "$1–$2");
}

function locationLabel(src: SourceCard) {
  if (typeof src.page === "number") return `Page ${src.page}`;
  if (src.section) return src.section;
  return src.location || "";
}

/* ---------- Answer text: paragraphs, lists, **bold**, and [n] citation markers ---------- */

function renderInline(
  text: string,
  onCite: (ref: number) => void,
  refs: Set<number>,
  keyBase: string,
): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
  return parts.map((part, i) => {
    const key = `${keyBase}-${i}`;
    const bold = part.match(/^\*\*([^*]+)\*\*$/);
    if (bold) return <strong key={key}>{bold[1]}</strong>;
    const cite = part.match(/^\[(\d+)\]$/);
    if (cite) {
      const ref = Number(cite[1]);
      if (!refs.has(ref)) return null;
      return (
        <button
          key={key}
          type="button"
          className={styles.cite}
          onClick={() => onCite(ref)}
          aria-label={`Show source ${ref}`}
        >
          {ref}
        </button>
      );
    }
    return part;
  });
}

function AnswerText({
  text,
  sources,
  onCite,
}: {
  text: string;
  sources: SourceCard[];
  onCite: (ref: number) => void;
}) {
  const refs = new Set(sources.map((s) => s.ref).filter((r): r is number => typeof r === "number"));
  const blocks: ReactNode[] = [];
  const lines = text.replace(/\r/g, "").split("\n");
  let list: { ordered: boolean; items: string[] } | null = null;
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) {
      const k = `p${blocks.length}`;
      blocks.push(<p key={k}>{renderInline(para.join(" "), onCite, refs, k)}</p>);
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      const k = `l${blocks.length}`;
      const items = list.items.map((item, i) => (
        <li key={`${k}-${i}`}>{renderInline(item, onCite, refs, `${k}-${i}`)}</li>
      ));
      blocks.push(list.ordered ? <ol key={k}>{items}</ol> : <ul key={k}>{items}</ul>);
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    // "### Title" or a line that is only bold ("**Title**" / "**Title:**") acts as a sub-heading.
    const heading = line.match(/^#{1,4}\s+(.*)$/) || line.match(/^\*\*([^*]+?):?\*\*:?$/);
    if (!line) {
      flushPara();
      flushList();
    } else if (heading) {
      flushPara();
      flushList();
      const k = `h${blocks.length}`;
      blocks.push(<h4 key={k}>{renderInline(heading[1], onCite, refs, k)}</h4>);
    } else if (bullet || numbered) {
      flushPara();
      const ordered = Boolean(numbered);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push((bullet || numbered)![1]);
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return <div className={styles.answerBody}>{blocks}</div>;
}

/* ---------- Sources ---------- */

function SourceRow({
  src,
  index,
  turnId,
  open,
  flash,
  onToggle,
}: {
  src: SourceCard;
  index: number;
  turnId: string;
  open: boolean;
  flash: boolean;
  onToggle: () => void;
}) {
  const ref = src.ref ?? index + 1;
  const panelId = `${turnId}-src-${ref}-panel`;
  return (
    <li id={`${turnId}-src-${ref}`} className={`${styles.source} ${flash ? styles.sourceFlash : ""}`}>
      <button
        type="button"
        className={styles.sourceHead}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={onToggle}
      >
        <span className={styles.sourceRef}>{ref}</span>
        <span className={styles.sourceTitle}>{shortTitle(src.title)}</span>
        <span className={styles.sourceLoc}>{locationLabel(src)}</span>
        <span className={styles.chevron} aria-hidden="true" />
      </button>
      {open ? (
        <div id={panelId} className={styles.sourceBody}>
          {src.excerpt ? <blockquote className={styles.excerpt}>{src.excerpt}</blockquote> : null}
          {src.url ? (
            <a className={styles.sourceLink} href={src.url} target="_blank" rel="noopener noreferrer">
              {typeof src.page === "number" ? `Open the catalog at page ${src.page}` : "Open source"}
              <span aria-hidden="true"> ↗</span>
            </a>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

function SourceList({
  turnId,
  sources,
  openRefs,
  flashRef,
  onToggle,
}: {
  turnId: string;
  sources: SourceCard[];
  openRefs: Set<number>;
  flashRef: number | null;
  onToggle: (ref: number) => void;
}) {
  return (
    <ol className={styles.sourceList}>
      {sources.map((src, i) => {
        const ref = src.ref ?? i + 1;
        return (
          <SourceRow
            key={`${turnId}-${ref}`}
            src={src}
            index={i}
            turnId={turnId}
            open={openRefs.has(ref)}
            flash={flashRef === ref}
            onToggle={() => onToggle(ref)}
          />
        );
      })}
    </ol>
  );
}

/* ---------- One question and its answer ---------- */

function TurnView({ turn, onRetry }: { turn: Turn; onRetry: (turn: Turn) => void }) {
  const [openRefs, setOpenRefs] = useState<Set<number>>(new Set());
  const [flashRef, setFlashRef] = useState<number | null>(null);
  const sources = turn.sources ?? [];

  function toggle(ref: number) {
    setOpenRefs((prev) => {
      const next = new Set(prev);
      if (next.has(ref)) next.delete(ref);
      else next.add(ref);
      return next;
    });
  }

  function cite(ref: number) {
    setOpenRefs((prev) => new Set(prev).add(ref));
    setFlashRef(ref);
    window.setTimeout(() => setFlashRef(null), 1400);
    requestAnimationFrame(() => {
      document
        .getElementById(`${turn.id}-src-${ref}`)
        ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  if (turn.status === "conversational") {
    return (
      <article className={styles.turn}>
        <h2 className={styles.question}>{turn.question}</h2>
        <p className={styles.chatReply}>{turn.answer}</p>
      </article>
    );
  }

  return (
    <article
      className={styles.turn}
      aria-busy={turn.status === "pending" || turn.status === "streaming"}
    >
      <h2 className={styles.question}>{turn.question}</h2>

      {turn.status === "pending" ? (
        <div className={styles.pending} role="status">
          <p className={styles.pendingText}>
            <span className={styles.dots} aria-hidden="true" />
            {typeof turn.passagesFound === "number"
              ? `Found ${turn.passagesFound} catalog passages. Writing the answer…`
              : `Searching the ${scopeLabel(turn.level)}…`}
          </p>
          <span className={styles.skel} />
          <span className={styles.skel} />
          <span className={`${styles.skel} ${styles.skelShort}`} />
        </div>
      ) : null}

      {turn.status === "streaming" ? (
        <>
          <p className={styles.pendingText} role="status">
            <span className={styles.dots} aria-hidden="true" />
            Writing from {turn.passagesFound ?? "the"} catalog passages…
          </p>
          {/* Citation numbers become clickable once sources arrive with the final answer. */}
          <div className={styles.streaming} aria-live="off">
            <AnswerText text={turn.answer || ""} sources={[]} onCite={cite} />
            <span className={styles.caret} aria-hidden="true" />
          </div>
        </>
      ) : null}

      {turn.status === "answered" ? (
        <>
          <p className={`${styles.badge} ${styles.badgeOk}`}>
            Answered from {sources.length} catalog {sources.length === 1 ? "page" : "pages"}
          </p>
          <AnswerText text={turn.answer || ""} sources={sources} onCite={cite} />
          {sources.length ? (
            <section className={styles.sources} aria-label="Sources">
              <h3 className={styles.sourcesHeading}>Sources</h3>
              <SourceList
                turnId={turn.id}
                sources={sources}
                openRefs={openRefs}
                flashRef={flashRef}
                onToggle={toggle}
              />
            </section>
          ) : null}
        </>
      ) : null}

      {turn.status === "abstained" ? (
        <>
          <p className={`${styles.badge} ${styles.badgeWarn}`}>Not covered in the catalogs</p>
          <div className={styles.answerBody}>
            <p>{turn.answer}</p>
          </div>
          {sources.length ? (
            <details className={styles.closest}>
              <summary>
                Closest passages we checked ({Math.min(sources.length, ABSTAIN_PREVIEW)})
              </summary>
              <SourceList
                turnId={turn.id}
                sources={sources.slice(0, ABSTAIN_PREVIEW)}
                openRefs={openRefs}
                flashRef={null}
                onToggle={toggle}
              />
            </details>
          ) : null}
        </>
      ) : null}

      {turn.status === "error" ? (
        <>
          <p className={`${styles.badge} ${styles.badgeErr}`}>No answer this time</p>
          <div className={styles.answerBody}>
            <p>{turn.answer}</p>
          </div>
          <button type="button" className={styles.retryBtn} onClick={() => onRetry(turn)}>
            Try again
          </button>
        </>
      ) : null}
    </article>
  );
}

/* ---------- Page ---------- */

export default function AskClient() {
  const [ready, setReady] = useState(false);
  const [session, setSession] = useState<StudentSession | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [level, setLevel] = useState<AcademicLevel>("all");
  const [draft, setDraft] = useState("");
  const [fullName, setFullName] = useState("");
  const [ucmId, setUcmId] = useState("");
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const lastTurnId = turns[turns.length - 1]?.id;

  useEffect(() => {
    const existing = loadSession();
    if (existing?.academicLevel) setLevel(existing.academicLevel);
    setSession(existing && !existing.greeted ? markGreeted(existing) : existing);
    setReady(true);
  }, []);

  useEffect(() => {
    if (!navOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setNavOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [navOpen]);

  // Bring each new question to the top of the reading area.
  useEffect(() => {
    if (!lastTurnId) return;
    document.getElementById(`turn-${lastTurnId}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [lastTurnId]);

  // Grow the question box with its content, up to the CSS max-height.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [draft]);

  function changeLevel(next: AcademicLevel) {
    setLevel(next);
    if (session) setSession(setSessionAcademicLevel(session, next));
  }

  async function onCheckIn(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setFormErrors({});
    try {
      const result = await validateIdentity(fullName, ucmId);
      if (!result.ok) {
        setFormErrors(result.errors || {});
        return;
      }
      const next: StudentSession = {
        fullName: result.full_name || fullName.trim(),
        firstName: result.first_name || "there",
        maskedId: result.masked_id || "••••",
        greeted: true,
      };
      saveSession(next);
      setSession(next);
      setTurns([]);
    } catch (err) {
      setFormErrors({
        form: err instanceof Error ? err.message : "Could not reach CampusAsk. Is the API running?",
      });
    } finally {
      setBusy(false);
    }
  }

  async function ask(question: string) {
    const cleaned = question.trim();
    if (!cleaned || !session || busy) return;
    const turn: Turn = { id: uid(), question: cleaned, level, status: "pending" };
    setBusy(true);
    setDraft("");
    setNavOpen(false);
    setTurns((prev) => [...prev, turn]);
    const patch = (fn: (t: Turn) => Turn) =>
      setTurns((prev) => prev.map((t) => (t.id === turn.id ? fn(t) : t)));
    const input = { question: cleaned, academicLevel: level, firstName: session.firstName };
    try {
      let result: AskResponse;
      try {
        result = await askQuestionStream(input, {
          onStatus: (s) => {
            if (typeof s.passages === "number") patch((t) => ({ ...t, passagesFound: s.passages }));
          },
          onDelta: (text) =>
            patch((t) => ({ ...t, status: "streaming", answer: (t.answer || "") + text })),
        });
      } catch (err) {
        if (!(err instanceof StreamUnavailableError)) throw err;
        result = await askQuestion(input); // API without the streaming endpoint
      }
      // Follow the level the API inferred ("I'm a master's student…").
      const suggested = result.suggested_academic_level || result.academic_level || null;
      if ((suggested === "graduate" || suggested === "undergraduate") && suggested !== level) {
        changeLevel(suggested);
      }
      const status = normalizeStatus(result.status);
      // The final answer is authoritative: it replaces the streamed text.
      patch((t) => ({
        ...t,
        status,
        answer: result.answer,
        sources: status === "conversational" ? [] : result.sources || [],
      }));
    } catch (err) {
      patch((t) => ({
        ...t,
        status: "error",
        answer:
          err instanceof Error
            ? `CampusAsk couldn't be reached (${err.message}).`
            : "CampusAsk couldn't be reached.",
      }));
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  function retry(turn: Turn) {
    setTurns((prev) => prev.filter((t) => t.id !== turn.id));
    void ask(turn.question);
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void ask(draft);
  }

  function onComposerKey(e: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void ask(draft);
    }
  }

  function onNewConversation() {
    setTurns([]);
    setNavOpen(false);
    inputRef.current?.focus();
  }

  function onSwitchStudent() {
    clearSession();
    setSession(null);
    setTurns([]);
    setFullName("");
    setUcmId("");
    setFormErrors({});
    setNavOpen(false);
  }

  return (
    <div className={`${styles.shell} ${navOpen ? styles.shellNavOpen : ""}`}>
      <SiteHeader
        active="ask"
        fluid
        navStart={
          <button
            type="button"
            className={styles.menuBtn}
            aria-expanded={navOpen}
            aria-controls="campusask-nav"
            onClick={() => setNavOpen((open) => !open)}
          >
            <span className={styles.menuIcon} aria-hidden="true" />
            Menu
          </button>
        }
      />
      <div className={styles.body}>
      <button
        type="button"
        className={styles.backdrop}
        aria-label="Close menu"
        tabIndex={navOpen ? 0 : -1}
        onClick={() => setNavOpen(false)}
      />

      <aside id="campusask-nav" className={`${styles.sidebar} ${navOpen ? styles.sidebarOpen : ""}`}>
        <div className={styles.sideHeader}>
          <p className={styles.sideTitle}>Your session</p>
          <button
            type="button"
            className={styles.closeNavBtn}
            onClick={() => setNavOpen(false)}
            aria-label="Close menu"
          >
            ✕
          </button>
        </div>

        {session ? (
          <div className={styles.chip}>
            <p className={styles.chipLabel}>Signed in</p>
            <p className={styles.chipName}>{session.fullName}</p>
            <p className={styles.chipId}>UCM ID {session.maskedId}</p>
          </div>
        ) : null}

        <div className={styles.sideNote}>
          <p className={styles.chipLabel}>How answers work</p>
          <p>
            CampusAsk searches the 2026–27 UCM catalogs and answers only from what it finds. Each
            answer lists the pages it used. If the catalogs don&apos;t cover a question, it says so.
          </p>
        </div>

        <div className={styles.sideActions}>
          <button
            type="button"
            className={styles.ghostBtn}
            onClick={onNewConversation}
            disabled={!session || turns.length === 0}
          >
            New conversation
          </button>
          {session ? (
            <button type="button" className={styles.ghostBtn} onClick={onSwitchStudent}>
              Switch student
            </button>
          ) : null}
          <Link className={styles.textLink} href="/" onClick={() => setNavOpen(false)}>
            ← Back to home
          </Link>
        </div>
      </aside>

      <section className={styles.main}>
        <div className={styles.thread} ref={threadRef}>
          <div className={styles.column}>
            {!ready ? <p className={styles.hint}>Loading…</p> : null}

            {ready && !session ? (
              <div className={styles.identityPanel}>
                <p className={styles.eyebrow}>Academic Catalog Desk</p>
                <h1 className={styles.identityTitle}>Student check-in</h1>
                <p className={styles.identityCopy}>
                  Enter your name and UCM ID to start. Your ID stays in this browser session and is
                  never sent to the language model.
                </p>
                <form className={styles.form} onSubmit={onCheckIn}>
                  <div>
                    <label className={styles.inputLabel} htmlFor="full-name">
                      Full name
                    </label>
                    <input
                      id="full-name"
                      className={styles.input}
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      placeholder="e.g. Alex Rivera"
                      maxLength={80}
                      autoComplete="name"
                      required
                    />
                    {formErrors.full_name ? <p className={styles.error}>{formErrors.full_name}</p> : null}
                  </div>
                  <div>
                    <label className={styles.inputLabel} htmlFor="ucm-id">
                      UCM ID number
                    </label>
                    <input
                      id="ucm-id"
                      className={styles.input}
                      value={ucmId}
                      onChange={(e) => setUcmId(e.target.value)}
                      placeholder="e.g. 700782694"
                      maxLength={12}
                      inputMode="numeric"
                      autoComplete="off"
                      required
                    />
                    {formErrors.ucm_id ? <p className={styles.error}>{formErrors.ucm_id}</p> : null}
                  </div>
                  {formErrors.form ? <p className={styles.error}>{formErrors.form}</p> : null}
                  <button className={styles.primaryBtn} type="submit" disabled={busy}>
                    {busy ? "Checking…" : "Continue to CampusAsk"}
                  </button>
                </form>
              </div>
            ) : null}

            {session && turns.length === 0 ? (
              <div className={styles.welcome}>
                <div className={styles.welcomeHead}>
                  <Image
                    className={styles.welcomeMo}
                    src="/ucm/mo-the-mule.webp"
                    alt="Mo the Mule reading in the library"
                    width={480}
                    height={319}
                    priority
                  />
                  <div>
                    <p className={styles.eyebrow}>Hi {session.firstName}</p>
                    <h1 className={styles.welcomeTitle}>Ask a Question</h1>
                    <p className={styles.welcomeCopy}>
                      Ask about academic policy in the 2026–27 Undergraduate and Graduate Catalogs.
                      Every answer links to the catalog page it came from.
                    </p>
                  </div>
                </div>
                <div className={styles.suggestGrid}>
                  {SAMPLE_GROUPS.map((group) => (
                    <section key={group.topic} className={styles.suggestGroup}>
                      <h2 className={styles.suggestTopic}>{group.topic}</h2>
                      {group.questions.map((q) => (
                        <button
                          key={q.id}
                          type="button"
                          className={styles.suggestBtn}
                          disabled={busy}
                          onClick={() => void ask(q.question)}
                        >
                          {q.question}
                        </button>
                      ))}
                    </section>
                  ))}
                </div>
              </div>
            ) : null}

            {turns.map((turn) => (
              <div key={turn.id} id={`turn-${turn.id}`} className={styles.turnWrap}>
                <TurnView turn={turn} onRetry={retry} />
              </div>
            ))}
          </div>
        </div>

        {session ? (
          <form className={styles.composer} onSubmit={onSubmit}>
            <div className={styles.column}>
              <fieldset className={styles.scope}>
                <legend className={styles.scopeLegend}>Search in</legend>
                <div className={styles.segmented}>
                  {LEVELS.map((opt) => (
                    <label
                      key={opt.value}
                      className={`${styles.segment} ${level === opt.value ? styles.segmentOn : ""}`}
                    >
                      <input
                        type="radio"
                        name="academic-level"
                        id={`level-${opt.value}`}
                        value={opt.value}
                        checked={level === opt.value}
                        onChange={() => changeLevel(opt.value)}
                      />
                      {opt.label}
                    </label>
                  ))}
                </div>
              </fieldset>
              <div className={styles.inputRow}>
                <textarea
                  id="question"
                  ref={inputRef}
                  rows={1}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={onComposerKey}
                  placeholder="Ask a policy question…"
                  aria-label="Your question"
                  maxLength={2000}
                />
                <button className={styles.sendBtn} type="submit" disabled={busy || !draft.trim()}>
                  {busy ? "Searching…" : "Ask"}
                </button>
              </div>
              <p className={styles.footnote}>
                Answers come only from the 2026–27 UCM catalogs. Check the cited page before you act
                on it.<span className={styles.keyHint}> Press Enter to ask, Shift+Enter for a new line.</span>
              </p>
            </div>
          </form>
        ) : null}
      </section>
      </div>
    </div>
  );
}
