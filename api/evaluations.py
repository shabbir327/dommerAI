"""Public evaluation and result-polling routes."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query

from api.security import require_api_key
from app_state import state
from config import DEFAULT_WEBHOOK_URL
from models import (
    AckResponse,
    EvaluationListResponse,
    EvaluationRequest,
    SubmissionStatus,
    WebhookPayload,
)
from services.evaluation_service import (
    grade_and_combine_mock,
    score_store_and_notify,
    store_mock_part,
)

router = APIRouter()


EVALUATE_EXAMPLES = {
    "pd2_single_mock": {
        "summary": "PD2 — single mock (question + answer, official grade)",
        "description": (
            "Official PD2 ground truth, grade 07 (bedømmelsesvejledning "
            "April 2025). A complete PD2 writing submission graded on the "
            "official -3..12 scale. This is the default mode "
            "(submission_mode can be omitted)."
        ),
        "value": {
            "eval_id": "pd2-official-07",
            "exam_type": "PD2",
            "question": "DEL 1 (B: En klage): Situation: Du bor i en lejlighed i boligforeningen Lysbo. Du har en nabo, som larmer meget. Du vil skrive en klage over din nabo til boligforeningen. Opgave: Skriv klagen til boligforeningen Lysbo. Du skal fortælle: hvem din nabo er, og hvor din nabo bor; hvordan og hvor tit din nabo larmer; hvad du har gjort for at løse problemet; hvad du håber, boligforeningen vil gøre. Du skal begynde og afslutte klagen på en passende måde.\n\nDEL 2 (En e-mail): Situation: Du har fået en e-mail fra din ven Emma. I e-mailen skriver hun bl.a.: '...I din sidste e-mail skrev du, at du har fået et nyt arbejde. Tillykke med det! Men du skrev også, at arbejdet somme tider er hårdt og stressende. Det lyder ikke så godt. Skriv og fortæl om dit nye arbejde, og hvorfor det somme tider er hårdt og stressende...' Opgave: Skriv et svar til Emma og fortæl om dit nye arbejde, og hvorfor det somme tider er hårdt og stressende. Du skal skrive minimum 100 ord.",
            "question_description": "PD2 skriftlig del, to delprøver. Bedømmes holistisk efter 7-trins-skalaen (-3 til 12) på pragmatisk, diskursiv og lingvistisk færdighed, B1-niveau. Delprøve 2 er afgørende for den endelige karakter.",
            "answer": "DEL 1 (KLAGE):\nTil Boligforeningen Lysbo\nNavn: (fornavn+efternavn)\n(Adresse)\nMin nabo hedder Susanne, og hun bor i 1. st. th.. Hun har to børn og de larmer meget især i aften, jeg kan ikke sove.\nJeg snak med hun og skrev besked, men hun har ikke gjorde noget.\nJeg vil gerne at boligforeningen skal stope larmet og vi skal bo sammen stille og rolig.\nMvh\n(fornavn + efternan + e-mail)\n\nDEL 2 (E-MAIL):\nKære Emma\nTak for din e-mail. ♥\nSom du ved, har jeg fået nyt arbejde på Apollo restaurant. Jeg er heldigvis, fordi jeg startede arbejde efter lang tid, men det er somme tider hårdt og stressende for mig, når restauranten er travlt og når gæstne ikke snakke Engelsk. Restauranten ligger tæt på i havet, og om sommeren er meget travlt.\nRestauranten holder også fest og brullup, så jeg er altid træt når jeg kom hjem.\nDet betyder jeg har hårdt somme tider og når gæstne snakker hurtig dansk jeg bliver nervøs og stressende, men min chef er sød, han hjælper mig, og han laver altid sjovt, så vi glemmte vores stressende.\nJeg håber at høre nogen fra dig.\nVi ses snart\nKh (fornavn)",
        },
    },
    "pd3_mock_del1": {
        "summary": "PD3 mock — step 1 of 2 (Del 1, the e-mail)",
        "description": (
            "Official PD3 ground truth, grade 07 (bedømmelsesvejledning "
            "August 2024). First half of a full PD3 mock. Response comes "
            "back 'awaiting_other_part' — no grade yet, no LLM calls spent. "
            "Send the Del 2 example next with the SAME mock_id to trigger "
            "grading."
        ),
        "value": {
            "eval_id": "pd3-official-07-del1",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-official-07",
            "delprove_part": "del1",
            "question": "Situation: Du har fået en mail fra din danske ven Mia. Hun har været udstationeret i New York i tre år for et dansk firma, og nu skal hun snart hjem til Odense sammen med sin familie. Mailen (uddrag): '...Tak for snakken i sidste uge, det var hyggeligt at tale med dig på Skype. Jeg er meget spændt på at høre om din køreprøve. Hvordan gik det? Som du ved, er vores tid i New York snart slut, og vi skal hjem til Odense om tre uger. Jeg skal arbejde sammen med nogle kolleger, jeg ikke kender. Hvad synes du, jeg skal gøre for at få en god start på arbejdet? Børnene er jo blevet teenagere, og de er meget kede af at sige farvel til deres venner her i New York. Jeg er spændt på, om de kan holde kontakten. Tror du, det er muligt? Jens har fået nyt arbejde i Danmark, som han glæder sig til at komme i gang med efter 3 år som hjemmegående. Det nye job ligger dog 30 km fra Odense, så vi overvejer at købe en ekstra bil, men han kan også tage toget. Hvad ville du gøre?...' Opgave: Skriv et svar til din ven. Tak for mailen. Kom ind på de understregede dele i mailen. Foreslå, at I mødes, når Mia kommer hjem.",
            "answer": "Hej Mia\nTak for mailen min kære ven. Jeg er glad for at høre fra dig og jeg håber, at du har det godt.\nI forhold til køreprøven gik det godt. Jeg levede nogen enkelte fejle, men jeg bestod heldigvis. Min familie overraskede mig med en fest. Det var meget hyggeligt og jeg ønskede du var hjemme og kunne fejre det sammen.\nMht. dine nye kolleger skal det nok gå godt. Du er en meget flink person og meget god til samarbejde. Jeg synes ikke, du skal gøre så meget. Du skal ikke præster noget du ikke ejer eller du ikke er. Du skal være meget åben, hjælpsom og høflig og det vil være nok for at få et godt indtryk.\nNår man er ung, er man meget nemt at få mange venner alle mulige steder, men det er svært at holde kontakten hvis man ikke ser og møder dem tit. Jeg tror, i starten bliver det ok og de kommer til at snakke hver dag og savne hinanden, men efter de har mødt nye mennesker, bliver det svaret at holde kontakten med de der er langt væk.\nTillykke med jobbet. Jeg er glad for ham, jeg ved godt hvor meget han savnede at være en del af arbejdsmarked. Jeg forslår, at tage toget. Det er meget tilgængelig for ham, for jeres hus ligger ved siden af banegården og det er nemlig også en miljøvenlig transportform.\nJeg glæder mig at se jer! Hvad synes du om at vi mødes hos mig, når I kommer hjemme?\nHils familien\nKh NN",
        },
    },
    "pd3_mock_del2": {
        "summary": "PD3 mock — step 2 of 2 (Del 2, the essay)",
        "description": (
            "Official PD3 ground truth, grade 07 — same case as the Del 1 "
            "example, same mock_id. Sending this triggers background "
            "grading of BOTH halves and the deterministic combination — "
            "poll GET /evaluation/pd3-official-07 for the final result once "
            "status moves past 'pending'."
        ),
        "value": {
            "eval_id": "pd3-official-07-del2",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-official-07",
            "delprove_part": "del2",
            "question": "DEL 2 (Skriftlig fremstilling om et alment og samfundsmæssigt emne, punkt B): Beskriv kort en attraktiv arbejdsplads fra egen erfaring (punkt 1). Kommentér, hvilken faktor du synes har størst betydning for en attraktiv arbejdsplads (punkt 2). Udtryk og argumenter for din holdning: er lederens personlige eller faglige kvalifikationer vigtigst for en attraktiv arbejdsplads (punkt 3)?",
            "answer": "Jeg er ung og har ikke haft så mange jobs, men et job som jeg var meget glad for var som assistent medhjælper i et hjemmepleje. Jeg elskede at arbejde der, da jeg hjalp gamle mennesker med hvad de havde brug for, jeg havde gode kolleger og en imponerede løn.\nDer er meget debat om hvad gøre ansatte tilfredse på en arbejdsplads og hvad der kan have betydning for, hvor attraktiv en arbejdsplads er for medarbejderne. Der er mange der siger, at en god løn betyder meget for jobsøgerne. Jeg er enig i det. Vi lever i et samfund hvor livs standard stiger, priserne stiger og man bliver nød til at tænke på hvor meget man tjener, derfor er jobbet med svag løn fravalget og det er mere attraktivt at arbejde for en højere løn.\nDerudover er personalegoder afgørende for medarbejdernes tilfredshed og attraktivitet. Mange virksomheder tilbyder deres medarbejdere personalegoder, som efteruddannelse eller kurser. Det er en god måde at fast holde ansatte, fordi de således bliver klogere, mere effektive på jobbet og i hverdagen.\nDer er forskellige faktorer, der kan have betydning for, hvor attraktiv en arbejdsplads er for medarbejderne. En af faktorerne er chefen. Man vil hele tiden have en dygtig og god leder, derfor synes jeg, at lederens både personlige og faglige kvalifikationer er vigtige for, om en arbejdsplads er attraktiv eller ej. Lederen er personen der ansætter og fyrer folk i en virksomhed, derfor at man er kvalificeret nok til det. Hvis man ikke er dygtig faglig, ved man ikke, hvis en person er kompetent nok til jobstilingen og man kan risikere at ansætte en forkert person og skabe en dårlig stemning i virksomheden. På den anden side vigtig at chefen er flink, åben og hjælpsom med sine medarbejdere. Når man får hjælp og støtte fra lederen, føler man en del af fællesskabet og jobbet bliver således mere attraktiv.\nAfslutningsvis er lederens både personlige og faglige kvalifikationer vigtige for, om en arbejdsplads er attraktiv og man er tilfreds med sit job, da ledere spiller en vigtig rolle over for deres ansatte og de er ansvarlige for at skabe et godt arbejdsmiljø.",
        },
    },
    "practice_writing": {
        "summary": "Practice drill (Writing Correction tool — pass/fail, no grade)",
        "description": (
            "The actual prompt from Hejdansk's Writing Correction tool "
            "(Week 2, Day 5 exercise — 'My Daily Routine'), not an official "
            "exam case since practice drills aren't part of the ministry "
            "grading guides. Response omits 'rubrik' and 'overall' "
            "entirely — pass/fail is based on whether any high-severity "
            "error remains unresolved. The 'errors' list (exact line/char "
            "positions for inline highlighting) is identical in shape to "
            "the exam modes."
        ),
        "value": {
            "eval_id": "practice-my-daily-routine-001",
            "exam_type": "PD2",
            "submission_mode": "practice",
            "question": "Describe your daily routine in 8 sentences using present tense. Include adjectives to describe things and people.",
            "question_description": "Week 2, Day 5. Minimum 50 words. Use adjectives from Chapter 3 and basic verbs from Chapter 4. Remember adjective agreement.",
            "answer": "Jeg vågner kl. 7. Jeg spiser en stor morgenmad. Jeg cykler til arbejde hver dag.",
        },
    },
}


@router.post(
    "/evaluate",
    response_model=AckResponse,
    status_code=202,
    tags=["Scoring"],
    summary="Submit a writing evaluation",
)
async def evaluate(
    request: Annotated[EvaluationRequest, Body(openapi_examples=EVALUATE_EXAMPLES)],
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
) -> AckResponse:
    if state.scorer is None or state.result_store is None:
        raise HTTPException(status_code=503, detail="Scorer is not ready.")

    request_webhook = str(request.webhook_url) if request.webhook_url else None
    effective_webhook = request_webhook or DEFAULT_WEBHOOK_URL or None
    if request_webhook:
        webhook_source = "request"
    elif DEFAULT_WEBHOOK_URL:
        webhook_source = "environment"
    else:
        webhook_source = "none"

    if request.submission_mode == "mock":
        # Deliberately skips the pending-row save below: an abandoned
        # single-part mock should leave no permanently-"pending" row and
        # cost no LLM calls. store_mock_part is synchronous (no LLM calls),
        # so we know synchronously whether to ack "still waiting" or kick
        # off the real grading in the background.
        if state.mock_progress is None:
            raise HTTPException(status_code=503, detail="Mock progress store is not ready.")
        ready = await store_mock_part(request)
        if not ready:
            return AckResponse(
                eval_id=request.eval_id,
                status="awaiting_other_part",
                webhook_url_used=effective_webhook,
                webhook_source=webhook_source,
            )
        background_tasks.add_task(grade_and_combine_mock, request.mock_id, effective_webhook)
        return AckResponse(
            eval_id=request.mock_id,
            status="pending",
            webhook_url_used=effective_webhook,
            webhook_source=webhook_source,
        )

    # "submission" carries the original request fields through to
    # EvaluationResultStore._build_database_row, which reads exam_type /
    # question / answer / webhook_url from exactly this key.
    state.result_store.save({
        "eval_id": request.eval_id,
        "status": "pending",
        "submission": request.model_dump(mode="json", exclude_none=True),
    })
    background_tasks.add_task(score_store_and_notify, request, effective_webhook)

    return AckResponse(
        eval_id=request.eval_id,
        status="pending",
        webhook_url_used=effective_webhook,
        webhook_source=webhook_source,
    )


@router.get(
    "/evaluation/{eval_id}",
    response_model=WebhookPayload,
    response_model_exclude_none=True,
    tags=["Results"],
    summary="Get one evaluation by eval_id",
    description=(
        "For submission_mode='single' or 'practice', use the eval_id you sent. "
        "For submission_mode='mock', poll using mock_id instead — the combined "
        "result is stored under that id, not under either half's own eval_id."
    ),
)
async def get_evaluation(
    eval_id: str,
    _: str = Depends(require_api_key),
) -> WebhookPayload:
    if state.result_store is None:
        raise HTTPException(status_code=503, detail="Result store is not ready.")
    result = state.result_store.get(eval_id.strip())
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    return WebhookPayload.model_validate(result)


@router.get(
    "/evaluations",
    response_model=EvaluationListResponse,
    tags=["Results"],
    summary="List recent evaluations",
)
async def list_evaluations(
    limit: int = Query(default=20, ge=1, le=100),
    status: SubmissionStatus | None = Query(default=None),
    _: str = Depends(require_api_key),
) -> EvaluationListResponse:
    if state.result_store is None:
        raise HTTPException(status_code=503, detail="Result store is not ready.")
    items = state.result_store.list(limit=limit, status=status)
    return EvaluationListResponse(count=len(items), items=items)
