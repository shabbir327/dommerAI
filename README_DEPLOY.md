# DommerAI v2.1 deployment

DommerAI intentionally connects to **two different Supabase projects**.

## 1. DommerAI Supabase project

This project contains DKF/EKE knowledge and the `evaluations` table. Configure:

```text
DOMMER_SUPABASE_URL=https://<dommerai-project-ref>.supabase.co
DOMMER_SUPABASE_SERVICE_ROLE_KEY=<dommerai-service-role-key>
```

Candidate submissions are written to the `evaluations` table in this project. The default table settings are:

```text
EVALUATIONS_TABLE=evaluations
EVALUATION_ID_COLUMN=eval_id
EVALUATION_STATUS_COLUMN=status
EVALUATION_RESULT_COLUMN=result_json
EVALUATION_UPDATED_COLUMN=updated_at
PERSIST_EVALUATIONS=true
```

## 2. DommerGrammar / dansk-grammar-hub Supabase project

This project contains COR / ordregister.dk resources in the `language` schema. Configure:

```text
GRAMMAR_SUPABASE_URL=https://<grammar-project-ref>.supabase.co
GRAMMAR_SUPABASE_SERVICE_ROLE_KEY=<grammar-service-role-key>
```

## Existing settings

```text
DOMMER_API_KEY=<existing-api-key>
GROQ_API_KEY=<groq-key>
WEBHOOK_URL=<existing-webhook-url>
```

Optional:

```text
LEXICAL_TIMEOUT_SECONDS=12
LEXICAL_MAX_CONCURRENCY=8
KNOWLEDGE_CACHE_TTL_SECONDS=300
```

For backward compatibility only, `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are accepted as fallbacks for the **DommerAI** project. They are never used for Grammar Hub. Prefer the explicit `DOMMER_*` names.

Render start command remains:

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The public evaluation contract is unchanged: `POST /evaluate`, the request payload, webhook payload, polling endpoints, and API key header remain the same.

## v2.1.2 evaluation persistence fix

`POST /evaluate` now writes the candidate submission directly into the existing flat
columns of the DommerAI `public.evaluations` table. It no longer assumes that a
`result_json` column must exist.

Recommended request field:

```json
"candidate_id": "candidate-pd2-001"
```

When `candidate_id` is omitted, `eval_id` is used as a safe fallback. Render logs now
show either `Evaluation persisted` or an explicit `Supabase evaluation persistence FAILED`
message, so database failures are no longer hidden behind successful webhook delivery.
