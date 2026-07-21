# DommerAI v1.0

Production-ready FastAPI service for asynchronous PD2/PD3 writing evaluation.

## Required Render environment variables

- `DOMMER_API_KEY`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `DATABASE_URL`
- `WEBHOOK_URL` (optional when every request supplies `webhook_url`)

## Render start command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Deployment order

1. Run `supabase_schema.sql` in Supabase SQL Editor.
2. Add the environment variables in Render.
3. Replace repository files with this package.
4. Deploy with a cleared build cache.
5. Test `/health`, then `/evaluate`, then `/evaluations/{eval_id}`.
