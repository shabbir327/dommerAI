"""Async PostgreSQL persistence for DommerAI v1.0."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings


def _build_async_url(raw_url: str) -> str:
    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
    raise RuntimeError("DATABASE_URL must be a PostgreSQL connection string.")


engine: AsyncEngine = create_async_engine(
    _build_async_url(settings.database_url),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    pool_recycle=300,
    connect_args={"ssl": "require"},
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def check_database() -> None:
    async with engine.connect() as connection:
        await connection.execute(text("select 1"))


async def close_database() -> None:
    await engine.dispose()


async def evaluation_exists(eval_id: str) -> bool:
    query = text(
        """
        select exists (
            select 1 from public.evaluations where eval_id = :eval_id
        )
        """
    )
    async with SessionLocal() as session:
        result = await session.execute(query, {"eval_id": eval_id})
        return bool(result.scalar_one())


async def create_evaluation(
    *,
    eval_id: str,
    candidate_id: str | None,
    exam_type: str,
    question: str,
    question_description: str | None,
    answer: str,
    submitted_at: datetime,
    metadata: dict[str, Any],
    webhook_url: str | None,
) -> None:
    query = text(
        """
        insert into public.evaluations (
            eval_id, candidate_id, exam_type, status,
            question, question_description, answer,
            submitted_at, metadata, webhook_url
        ) values (
            :eval_id, :candidate_id, :exam_type, 'pending',
            :question, :question_description, :answer,
            :submitted_at, cast(:metadata as jsonb), :webhook_url
        )
        """
    )
    params = {
        "eval_id": eval_id,
        "candidate_id": candidate_id,
        "exam_type": exam_type,
        "question": question,
        "question_description": question_description,
        "answer": answer,
        "submitted_at": submitted_at,
        "metadata": json.dumps(metadata),
        "webhook_url": webhook_url,
    }
    async with SessionLocal() as session:
        await session.execute(query, params)
        await session.commit()


async def mark_processing(eval_id: str, started_at: datetime) -> None:
    query = text(
        """
        update public.evaluations
        set status = 'processing', started_at = :started_at, updated_at = now()
        where eval_id = :eval_id
        """
    )
    async with SessionLocal() as session:
        await session.execute(query, {"eval_id": eval_id, "started_at": started_at})
        await session.commit()


async def complete_evaluation(
    *,
    eval_id: str,
    status: str,
    completed_at: datetime,
    rubric: dict[str, Any] | None,
    overall: int | None,
    pass_fail: str | None,
    feedback_da: str | None,
    errors: list[dict[str, Any]],
    word_count: int | None,
    model_name: str | None,
    prompt_version: str,
    error: str | None,
) -> None:
    query = text(
        """
        update public.evaluations
        set status = :status,
            completed_at = :completed_at,
            rubric = cast(:rubric as jsonb),
            overall = :overall,
            pass_fail = :pass_fail,
            feedback_da = :feedback_da,
            errors = cast(:errors as jsonb),
            word_count = :word_count,
            model_name = :model_name,
            prompt_version = :prompt_version,
            error = :error,
            updated_at = now()
        where eval_id = :eval_id
        """
    )
    params = {
        "eval_id": eval_id,
        "status": status,
        "completed_at": completed_at,
        "rubric": json.dumps(rubric) if rubric is not None else None,
        "overall": overall,
        "pass_fail": pass_fail,
        "feedback_da": feedback_da,
        "errors": json.dumps(errors),
        "word_count": word_count,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "error": error,
    }
    async with SessionLocal() as session:
        await session.execute(query, params)
        await session.commit()


async def get_evaluation_record(eval_id: str) -> dict[str, Any] | None:
    query = text(
        """
        select * from public.evaluations where eval_id = :eval_id
        """
    )
    async with SessionLocal() as session:
        result = await session.execute(query, {"eval_id": eval_id})
        row = result.mappings().first()
        return dict(row) if row is not None else None
