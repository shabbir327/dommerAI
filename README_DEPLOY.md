# DommerAI v2 deployment

## What stays unchanged for Adnan

- Base URL remains `https://dommerai.onrender.com`
- `POST /evaluate` remains available
- `X-API-Key` remains required
- Existing request fields remain valid
- HTTP response remains `202` with `eval_id` and `status: pending`
- Webhook delivery remains active
- `GET /evaluation/{eval_id}` and `GET /evaluations` remain available

The acknowledgement may still include the existing optional fields `webhook_url_used` and `webhook_source`, exactly as the current production code does.

## New endpoints

- `GET /grammar-hub/health`
- `GET /lexicon/lookup?word=husene`
- `GET /lexicon/lemma/{cor_lemma_id}`
- `POST /lexicon/analyze`

All new endpoints use the same `X-API-Key`.

## GitHub update

Copy the full folder structure into the repository root. Keep the Render start command:

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Required Render environment variables

Existing variables remain unchanged. Confirm these also exist:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
DOMMER_API_KEY
GROQ_API_KEY
WEBHOOK_URL
```

Optional lexical settings:

```text
LEXICAL_TIMEOUT_SECONDS=12
LEXICAL_MAX_CONCURRENCY=8
```

## Smoke tests

```bash
curl "https://dommerai.onrender.com/health"
```

```bash
curl "https://dommerai.onrender.com/grammar-hub/health" \
  -H "X-API-Key: test-key-123"
```

```bash
curl "https://dommerai.onrender.com/lexicon/lookup?word=husene" \
  -H "X-API-Key: test-key-123"
```

Existing integration test:

```bash
curl -X POST "https://dommerai.onrender.com/evaluate" \
  -H "accept: application/json" \
  -H "X-API-Key: test-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "eval_id": "pd2-regression-test-001",
    "exam_type": "PD2",
    "question": "Du har købt en jakke online, men den er ankommet med en skade. Skriv en e-mail til butikken.",
    "question_description": "Forklar problemet, fortæl hvornår du modtog jakken, og sig hvad du ønsker.",
    "answer": "Kære kundeservice. Jeg modtog jakken i mandags, men den var ødelagt. Jeg ønsker en ny jakke."
  }'
```
