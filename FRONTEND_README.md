# DommerAI — Frontend Integration Guide

This covers everything the Hejdansk frontend needs to know to call DommerAI
correctly: the three submission modes, the two-call mock flow, the error
positioning contract, and what changed in this stabilization pass.

Auth: every request needs header `X-API-Key: <key>`. Base URL is your
Render deployment URL.

---

## The three submission modes

DommerAI has one endpoint, `POST /evaluate`, with a `submission_mode` field
that changes how a request is handled. If omitted, it defaults to `"single"`.

### 1. `single` — one question, one answer, official grade

Used for PD2 (both delprøver combined into one `answer` field by whoever
builds the payload — DommerAI doesn't split PD2 into two calls) and for any
one-off single-task grading.

```json
POST /evaluate
{
  "eval_id": "some-unique-id",
  "exam_type": "PD2",
  "question": "...",
  "question_description": "...",
  "answer": "..."
}
```

Response (immediate): `{"eval_id": "...", "status": "pending", ...}` — actual
grading runs in the background. Poll `GET /evaluation/{eval_id}` until
`status` moves to `"scored"` or `"failed"`.

Scored response has: `rubrik` (pragmatisk/diskursiv/lingvistisk, each
Top/Midt/Bund/"Under niveau"), `overall` (the -3/0/2/4/7/10/12 scale),
`pass_fail`, `feedback`, `examiner_summary`, `errors[]` (see below),
`word_count`.

### 2. `mock` — PD3's two-part exam, sent as two separate calls

PD3's real exam has two delprøver (Del 1: an e-mail reply; Del 2: an essay,
either diagram analysis or a viewpoint piece) that get graded together as
one official result. Because PD3's UI locks Part 1 before Part 2 unlocks,
DommerAI expects these as **two separate API calls sharing a `mock_id`**,
not one combined call.

**Del 1:**
```json
POST /evaluate
{
  "eval_id": "mock-abc123-del1",
  "exam_type": "PD3",
  "submission_mode": "mock",
  "mock_id": "mock-abc123",
  "delprove_part": "del1",
  "question": "...",
  "answer": "..."
}
```
Response: `{"status": "awaiting_other_part", ...}` — **no grading happens
yet, no LLM calls are spent.** This is deliberate: an abandoned mock (student
does Del 1 and never comes back) never gets graded and never costs anything.

**Del 2** (send whenever the student finishes it — could be seconds or days
later, that's fine):
```json
POST /evaluate
{
  "eval_id": "mock-abc123-del2",
  "exam_type": "PD3",
  "submission_mode": "mock",
  "mock_id": "mock-abc123",
  "delprove_part": "del2",
  "question": "...",
  "answer": "..."
}
```
Response: `{"status": "pending", ...}` — **this triggers background grading
of both halves.** Poll `GET /evaluation/{mock_id}` (the shared `mock_id`,
**not** either half's own `eval_id`) until `status` is `"scored"` or
`"failed"`.

**PD2 does not use this mode.** By design, PD2's two parts are combined
into one `answer` string and sent as `single` mode. If that ever changes,
this doc needs updating — right now, `mock` mode is PD3-only.

#### Combined result shape

```json
{
  "eval_id": "mock-abc123",
  "status": "scored",
  "overall": 7,
  "pass_fail": "PASSED",
  "rubrik": { "pragmatisk": "Midt", "diskursiv": "Midt", "lingvistisk": "Bund" },
  "feedback": "Del 1 (e-mail): ... Del 2 (skriftlig fremstilling): ...",
  "examiner_summary": "Del 1 (e-mail): ... Del 2 (skriftlig fremstilling): ...",
  "errors": [ "see 'The errors contract' below" ],
  "del1_word_count": 260,
  "del2_word_count": 355,
  "del1": { "Del 1's complete standalone result -- its own rubrik, feedback, errors, writing_statistics, everything" },
  "del2": { "Del 2's complete standalone result, same shape" }
}
```

`rubrik`/`overall`/`pass_fail` at the top level reflect **Del 2**, since
Del 2 is explicitly the decisive part per the official grading guide — Del 1
only tips the grade in specific boundary cases (documented in the combination
logic, not something the frontend needs to reason about).

#### `status: "failed"` on a mock — read this carefully

If one half's grading genuinely breaks (an LLM/infrastructure failure, not
"the student didn't answer"), the **whole combined result comes back as
`status: "failed"`**, with `overall`/`pass_fail`/`rubrik` all `null` and a
real explanation in `error`. This is deliberate: an infrastructure hiccup on
our side must never silently produce a plausible-looking but wrong grade for
a student. If you see this, it means grading needs a retry or manual
investigation — do not show whatever grade might be partially present as
if it were final; there isn't one.

### 3. `practice` — standalone drills, pass/fail only, no grade

For casual practice exercises (e.g. the Writing Correction tool), not tied
to an exam mock:

```json
POST /evaluate
{
  "eval_id": "practice-xyz",
  "exam_type": "PD2",
  "submission_mode": "practice",
  "question": "...",
  "question_description": "...",
  "answer": "..."
}
```

Scored response: `rubrik` and `overall` are always `null` here — this
isn't a bug, practice drills aren't on the official exam scale. `pass_fail`
is based on two things: no unresolved high-severity grammar error, AND no
task requirement marked "missing" in `task_coverage` (e.g. an answer wildly
under the stated word count still fails, even if what little text exists is
grammatically clean).

---

## The errors[] contract — same shape everywhere, one thing to check

Every mode returns errors in the same `InlineError` shape:

```json
{
  "original": "...", "correction": "...", "type": "...",
  "explanation": "...", "severity": "low | medium | high",
  "part": "del1 | del2 | null",
  "line": 3, "column_start": 1, "column_end": 45,
  "start_char": 120, "end_char": 165,
  "line_text": "...", "grammar_rule_title": "...", "affects_score": true
}
```

**The one thing to check: `part`.** For `single`/`practice` results, `part`
is always `null` — apply every error's position against the single `answer`
field, as you'd expect.

For `mock` results, the top-level `errors[]` array contains both halves'
errors merged into one list, each tagged with `part: "del1"` or `"del2"`.
Positions are relative to that part's own text, not the combined `answer`
field. If you're rendering two separate textareas (email + essay), filter
this array by `part` and apply each group's positions against the matching
textarea. `del1.errors` and `del2.errors` also contain the same per-part
lists individually, if that's more convenient than filtering the merged
array.

---

## Timestamps

All timestamps (`created_at`, `updated_at`, `completed_at`) are computed in
real Danish local time (Europe/Copenhagen), not UTC — the ISO string carries
the correct offset (+02:00 in summer, +01:00 in winter) already baked in.
No conversion needed on your end.

---

## What changed in this stabilization pass

If you'd already integrated against an earlier version of this API, here's
what's different:

- **`mock`/`practice` modes are new** — previously only `single` existed.
- **`del1`/`del2` fields are new** on mock results — full standalone
  results for each part.
- **`del1_word_count`/`del2_word_count` are new** — no need to dig into
  `del1.word_count`/`del2.word_count` anymore for a quick word count.
- **`errors[].part` is new** — lets you use one merged errors array for
  mock results instead of always reading `del1.errors`/`del2.errors`
  separately.
- **`status: "failed"` on a mock now means something specific**: a genuine
  grading failure on one half, not "the student got a low/failing grade."
  Previously this case was silently mis-handled as if the student hadn't
  answered that part, producing an incorrect grade — now it's surfaced
  honestly instead.
- **Provider/model names in `model_metadata`** now say `"together"` (or
  whatever's actually configured) instead of hardcoded `"groq"` text
  leaking into logs/metadata regardless of the real provider.
