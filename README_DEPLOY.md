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
SUPABASE_URL_COR=https://<grammar-project-ref>.supabase.co
SUPABASE_KEY_COR=<grammar-service-role-key>
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


## Candidate submission persistence

Each `POST /evaluate` immediately upserts a pending row to the original DommerAI Supabase `evaluations` table. The `result_json.submission` object stores `exam_type`, `question`, `question_description`, and the candidate `answer`. When scoring finishes, the result is merged into the same row, so the original candidate submission is preserved together with the evaluation.
