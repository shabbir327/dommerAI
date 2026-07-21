create table if not exists public.evaluations (
    eval_id text primary key,
    candidate_id text,
    exam_type text not null check (exam_type in ('PD2', 'PD3')),
    status text not null check (status in ('pending', 'processing', 'scored', 'failed')),

    question text not null,
    question_description text,
    answer text not null,

    submitted_at timestamptz not null,
    started_at timestamptz,
    completed_at timestamptz,

    metadata jsonb not null default '{}'::jsonb,
    webhook_url text,

    rubric jsonb,
    overall integer,
    pass_fail text,
    feedback_da text,
    errors jsonb not null default '[]'::jsonb,
    word_count integer,

    model_name text,
    prompt_version text not null default 'v1',
    error text,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists evaluations_candidate_id_idx
    on public.evaluations (candidate_id);

create index if not exists evaluations_status_idx
    on public.evaluations (status);

create index if not exists evaluations_submitted_at_idx
    on public.evaluations (submitted_at desc);
