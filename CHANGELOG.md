# DommerAI — Stabilization Pass Changelog

Summary of every change made across this session, in rough chronological
order. Read `FRONTEND_README.md` for the API contract Adnan needs; this file
is for you/anyone reviewing the diff before pushing to production.

## Schema / Supabase

- **New `evaluation_results` table** replaces the old `evaluations` table.
  Consolidates 9 secondary columns (webhook_url, writing_statistics,
  knowledge_used, retrieval_metadata, model_metadata, dimension_reasons,
  task_coverage, strengths, improvements) into one `metadata` jsonb column.
  Drops `candidate_id` and the redundant `started_at`/`submitted_at` pair
  (both always identical — replaced with one `created_at`).
  Run `evaluation_results_table.sql` for a fresh setup, or the two ALTER
  scripts if the table already exists: `mock_progress_table.sql` (separate,
  unrelated table — holds one submitted mock half while waiting for its
  pair) and `evaluation_results_add_word_counts.sql`.
- All timestamps now computed in real Europe/Copenhagen local time before
  being sent to Supabase, not UTC. Requires `tzdata` (added to
  `requirements.txt`) since minimal Docker images don't always ship a
  system tz database.

## New submission modes

- **`mock` mode**: PD3's two-part exam (Del 1 email + Del 2 essay), sent as
  two separate `/evaluate` calls sharing a `mock_id`. Grading — and LLM
  spend — only happens once both halves have arrived; an abandoned
  single-part mock is never graded. Deterministic combination logic
  (`Scorer.combine_mock_grades`) implements the official PD3/PD2 grading
  guides' rules: essay-only docks one grade, email-only caps at 0/-3, and a
  specific boundary-nudge rule for grades landing exactly on 0 or 2.
- **`practice` mode**: standalone drills (Writing Correction tool). No
  official grade scale — `rubrik`/`overall` are always null. `pass_fail`
  checks both grammar-error severity AND task completion (word count,
  sentence count) — a short-but-clean answer now correctly fails if it's
  well under the stated requirement.

## Bug fixes, in order found

1. **`exam_type`/`question`/`answer` were never set directly on
   `WebhookPayload`** — they only ever reached the database through an
   implicit "submission" merge that mock mode's design (no pending-row
   save) didn't populate. Caused `NOT NULL` constraint violations that
   silently dropped entire mock result rows. Fixed by making these
   first-class fields, set explicitly everywhere a payload is built.
2. **Same class of bug, `MockProgressStore` and `EvaluationResultStore`
   both had an in-memory-only read before merging** — a process restart
   between two saves would silently lose earlier data. Fixed by checking
   Supabase before assuming an ID missing from the cache is genuinely new.
3. **Silent-failure gap: a syntactically valid but near-empty LLM response**
   (e.g. `{"overall": 0}` with no rubric/feedback) would flow through
   per-field defaults and come back looking like a normal "scored" result.
   Added a completeness check that retries instead of accepting it.
4. **Truncated responses retried with the identical `max_tokens`**, failing
   the same way every time. Now escalates the budget (×1.5 per attempt) on
   a detected truncation before retrying.
5. **Grade guardrail was too strict**: `pragmatisk == Bund` + one missing
   task item auto-failed the grade (capped at 0), contradicting the
   official guide's explicit wording that content issues alone should
   never push a grade below the pass line. Loosened to cap at 2 instead.
6. **Grade-10+ guardrail gap**: a single Top rubric dimension (often
   pragmatisk, the easiest one) could sustain a 10 even with the other two
   dimensions at Midt. Added a rule requiring at least 2 of 3 Top
   dimensions for any grade ≥10.
7. **The most serious one: a genuine LLM/infrastructure grading failure on
   one mock half was indistinguishable from "student didn't answer,"**
   silently applying the essay-only/email-only docking penalty to a
   real, present submission. Now returns `status: "failed"` on the whole
   combined result instead of a wrong-but-plausible grade.
8. Hardcoded `"Groq"` text in error/log messages regardless of actual
   configured provider — fixed to use the real provider name dynamically.

## Renames — backward compatible

Every `GROQ_*`-named environment variable now has an agnostic primary name
(`LLM_GRADING_MODEL`, `LLM_INTERN_MODEL`, `LLM_GRADING_TEMPERATURE`, etc.),
with the old `GROQ_*` name still working as a fallback — **no Render env var
changes are required**, but new deployments should use the new names.
Internal identifiers (`_call_groq` → `_call_llm`, etc.) renamed to match.

## Grading consistency

`LLM_GRADING_TEMPERATURE` default lowered `0.05` → `0.025`,
`LLM_INTERN_TEMPERATURE` `0.1` → `0.05`. Directly motivated by observing the
same exact submission land on grades 4 and 7 (a full grade-step apart) on
back-to-back runs — the model's own rubric-dimension labeling varied enough
between runs to swing the final grade. Lower temperature should shrink this,
not eliminate it entirely; worth re-testing against ground truth after
deploying.

## New top-level fields

- `del1_word_count` / `del2_word_count` on mock results — no need to dig
  into `del1.word_count`/`del2.word_count` for a quick number.
- `errors[].part` (`"del1"` / `"del2"` / `null`) — lets a frontend use one
  merged errors array for mock results instead of two separate reads.

## Known open items — not fixed, flagged for awareness

- COR/DanskGrammatik Hub lookups failed once during testing
  (`retrieval_metadata.knowledge_sources.grammarhub_cor.status: "failed"`)
  and recovered on a later run without any code change — likely transient,
  but undiagnosed. If it recurs, check whether the separate COR Supabase
  project (`SUPABASE_URL_COR`) is reachable.
- The all-3-Bund-dimensions hard cap (max grade 2) was flagged as
  potentially too strict against one official ground-truth case, same as
  the fixed pragmatisk-Bund rule — left as-is per explicit decision not to
  touch it yet.
- `mock` mode's combined `feedback`/`examiner_summary` text hardcodes
  "Del 1 (e-mail)" / "Del 2 (skriftlig fremstilling)" labels, correct for
  PD3 specifically. If PD2 ever moves onto `mock` mode, these labels would
  be wrong (PD2's Del 1 is a review/complaint/invitation, not an email).
