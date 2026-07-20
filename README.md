# Dommer API

Render start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Required Render environment variables:

- `GROQ_API_KEY`
- `DOMMER_API_KEY`
- Optional: `GROQ_MODEL`

Endpoints:

- `GET /health`
- `POST /evaluate`
- `GET /evaluations/{eval_id}`
- `GET /docs`
