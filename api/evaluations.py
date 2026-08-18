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
            "Official PD2 ground truth, grade 04 (bedømmelsesvejledning "
            "April 2025). A complete PD2 writing submission graded on the "
            "official -3..12 scale. This is the default mode "
            "(submission_mode can be omitted)."
        ),
        "value": {
            "eval_id": "pd2-official-04",
            "exam_type": "PD2",
            "question": "DEL 1 (B: En klage): Situation: Du bor i en lejlighed i boligforeningen Lysbo. Du har en nabo, som larmer meget. Du vil skrive en klage over din nabo til boligforeningen. Opgave: Skriv klagen til boligforeningen Lysbo. Du skal fortælle: hvem din nabo er, og hvor din nabo bor; hvordan og hvor tit din nabo larmer; hvad du har gjort for at løse problemet; hvad du håber, boligforeningen vil gøre. Du skal begynde og afslutte klagen på en passende måde.\n\nDEL 2 (En e-mail): Situation: Du har fået en e-mail fra din ven Emma. I e-mailen skriver hun bl.a.: '...I din sidste e-mail skrev du, at du har fået et nyt arbejde. Tillykke med det! Men du skrev også, at arbejdet somme tider er hårdt og stressende. Det lyder ikke så godt. Skriv og fortæl om dit nye arbejde, og hvorfor det somme tider er hårdt og stressende...' Opgave: Skriv et svar til Emma og fortæl om dit nye arbejde, og hvorfor det somme tider er hårdt og stressende. Du skal skrive minimum 100 ord.",
            "question_description": "PD2 skriftlig del, to delprøver. Bedømmes holistisk efter 7-trins-skalaen (-3 til 12) på pragmatisk, diskursiv og lingvistisk færdighed, B1-niveau. Delprøve 2 er afgørende for den endelige karakter.",
            "answer": "DEL 1 (KLAGE):\nBoligforeningen Lysbo\nHej,\nJeg hedder (fornavn + efternavn).\nJeg bor i en lejlighed i boligforeningen Lysbo\nJeg har problemer med en nabo.\nHans navn er Anders Jensen. Han er meget larmer.\nHan gør hjemme omkring klokken 23 hver natten og lyder music og dansede.\nMin job er meget hård og jeg behøver min afslapning.\nJeg snakker om det med Anders, men han ikke lyder mig.\nJeg håber, du har hjælp til mig.\nMange tak.\nHilsen\n(fornavn + efternavn).\n\nDEL 2 (E-MAIL):\nHej Emma\nHvordan har du det?\nJeg håber, at er du det godt.\nTak for din e-mail.\nJeg skriv du om min ny arbejde.\nJeg skriv nu om det mere.\nJeg arbejdet som special pædagog til mennesker med problemer.\nDet er drenge fra 12 år til 18 år.\nDrenge har problem med alkohol, aggression og tyveri.\nOm wekkendet jeg arbejdet kl. 8-22.\nOm hverdag jeg arbejdet kl. 8-17.\nMin job er meget hårdt og stressende.\nJeg kommer hjemme meget træt.\nJeg har ikke tid til min familie.\nDet er også stressende til mig og min børne.\nJeg behøver tid med min mænd og min børn.\nNu jeg behøver måske ferie i vores sommerhus i Rødvig.\nLyder music, laver honninkage, spiser god mad og vin, og spiller music med min mænd og børn.\ntak for din tid til mig.\nHav en god dag.\nHilsen\n(fornavn)",
        },
    },
    "pd3_mock_del1": {
        "summary": "PD3 mock — step 1 of 2 (Del 1, the e-mail)",
        "description": (
            "Official PD3 ground truth, grade 04 (bedømmelsesvejledning "
            "August 2024). First half of a full PD3 mock. Response comes "
            "back 'awaiting_other_part' — no grade yet, no LLM calls spent. "
            "Send the Del 2 example next with the SAME mock_id to trigger "
            "grading."
        ),
        "value": {
            "eval_id": "pd3-official-04-del1",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-official-04",
            "delprove_part": "del1",
            "question": "Situation: Du har fået en mail fra en dansk ven, som skriver om Mihai fra Rumænien. Mailen (uddrag): '...Tak for sidst, det var hyggeligt. Hvad med den jobsamtale, du skulle til i går – tror du, du får jobbet? Kan du huske min ven Mihai? Han er blevet færdig med første del af sin uddannelse og vil måske tage anden del i Danmark. Han har spurgt mig, hvordan det er at komme til Danmark som udlænding. Men det har jeg jo ikke prøvet, så måske kan du hjælpe. Han spørger for eksempel, hvad det er vigtigt at gøre som noget af det første, når man lige er flyttet til Danmark. Han vil også gerne vide, hvad han kan gøre for at få venner her. Og så er han også lidt bekymret for vejret – tror du, han kan vænne sig til det?...' Opgave: Skriv et svar. Tak for mailen. Kom ind på de understregede dele i mailen. Foreslå, at Mihai kontakter dig.",
            "answer": "Hej Erik,\nTak for din e-mail. Jeg har det godt, hvad med dig?\nMin jobsamtale gik godt, tak. Men jeg er ikke færdig med min jobsamtale! Jeg har stadig et møde med den chef næste uge. Det var fire ansøgere til jobsamtalen og intervieweren var tilfreds med mig, så jeg håber, at jeg får jobbet.\nDesværre kan jeg ikke huske din ven Mihai, fordi du har så mange venner! Jeg synes, det er godt at have en uddannelse i Danmark, hvis Mihai har en mulighed at komme her. I dag, er det meget vigtigt at have en international erfaring som en elev.\nDa jeg kom til Danmark for 5 år siden, var det en ny erfaring for mig fordi jeg har aldrig været ud fra XX. Det var lidt anderledes at bo blandt så få mennesker!\nNår Mihai kommer til Danmark, er det meget vigtigt at han anmelder sig selv til sin kommune. Kommunen vil hjælpe ham, at få et CPR nummer. Uden CPR nummeret, er det næsten umugligt at åbne en bank konto.\nJeg kan godt forstå, at Mihai bekymrer sig over, om han vil få venner her i Danmark. Det er meget svært at få venner når vi er voksen. Jeg er indadvendt, så jeg bekymrer mig ikke over venner i Danmark. Jeg anbefaler, at han melder sig ind i en idrætsklub. I klubben, har han muglighed at møde folk fra forskellige lander og også fra Danmark. Hvad synes du om det?\nJeg ved ikke, hvordan vejret er i Rumænien. Men jeg synes ikke at vejret vil være et problem for ham. Han skal kun være parat til regnvejret og blæsevejret på det samme dag!\nJeg anbefaler, at Mihai skriver til mig. Jeg er sikker på, at han har flere spørgsmål, og det er nemmere når han skriver direkt til mig. Du kan give ham mit telefonnummer.\nMange hilsner\nNN",
        },
    },
    "pd3_mock_del2": {
        "summary": "PD3 mock — step 2 of 2 (Del 2, the essay)",
        "description": (
            "Official PD3 ground truth, grade 04 — same case as the Del 1 "
            "example, same mock_id. Sending this triggers background "
            "grading of BOTH halves and the deterministic combination — "
            "poll GET /evaluation/pd3-official-04 for the final result once "
            "status moves past 'pending'."
        ),
        "value": {
            "eval_id": "pd3-official-04-del2",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-official-04",
            "delprove_part": "del2",
            "question": "DEL 2 (Skriftlig fremstilling om et alment og samfundsmæssigt emne, punkt A): Der vises et diagram om salget af økologiske fødevarer i danske butikker 2007-2011 (stigning fra 3,6 til 5,5 mia. kr.). Beskriv kort, hvad diagrammet viser (punkt 1). Forklar mulige årsager til udviklingen (punkt 2). Udtryk og argumenter for din holdning til fordele og ulemper ved økologiske fødevarer (punkt 3).",
            "answer": "Et diagram fra Landbrugsavisen.dk viser salget af økologiske fødevarer i danske butikker i milliarder kroner. Diagrammet viser at salget er steget fra 3,6 milliarder kroner til 5,5 milliarder kroner i 4 år (mellem 2007 og 2011).\nI dag, vider folk mere om økologiske fødevarer end før, på grund af medier og reklamer, der taler ofte om det. Næsten alle folk ved at kemikalier er usund, så salget af almindelige fødvarer taber. Vi tjener også mere penge i dag, så vi er ikke nødt til at tænke meget, om hvordan vi tilbringer vores penge. Man læser også i avisen, at kemikalier, som vi bruger at dyrke almindelige fødvarer påvirker jorden. Så folk foretrækker at købe økologiske fødevarer.\nFordeler med økologiske fødevarer er at det er sund, fordi det har ingen, eller mindre kemikalier. Med sund mad, reducerer vi mulighed for sygdomme som kræft. En anden fordel er, at når vi dyrker uden kemikalier, være jorden også sund og vandet fra kilder er også ren.\nUlemper med økologiske fødevarer er, at det er meget dyrt end almindelige fødevarer. Det er fordi uden kemikalier, kan de ikke dyrke så meget som almindelige fødevarer. Så landsmanden har mere omkostninger.\nEn anden ulempe er at økologiske fødevarer modner hurtige. Så vi kan ikke købe meget og holde i køleskabet for lange tid. Endelige, jeg tror ikke at vi får alle slags fødevarer som økologiske fødevarer.\nPå trods af så mange ulemper, jeg synes, at økologiske fødevarer er et godt valg. Hvis vi får et tilskud for at købe økologiske fødevarer, jeg synes, flere mennesker vil købe det.",
        },
    },
    "pd2_mock_del1": {
        "summary": "PD2 mock — step 1 of 2 (Del 1, en invitation)",
        "description": (
            "Official PD2 ground truth, combined grade 02 (bedømmelsesvejledning "
            "April 2025, prøvegrundlag maj-juni 2021). First half of a full PD2 "
            "mock — proves the two-call mock flow is exam-type agnostic, not "
            "PD3-only. Response comes back 'awaiting_other_part' — no grade "
            "yet, no LLM calls spent. Send the Del 2 example next with the "
            "SAME mock_id to trigger grading. Note PD2's Del 1 varies by mock "
            "(here an invitation; other mocks use a complaint/'en klage') "
            "unlike PD3 where Del 1 is always an e-mail."
        ),
        "value": {
            "eval_id": "pd2-official-02-del1",
            "exam_type": "PD2",
            "submission_mode": "mock",
            "mock_id": "pd2-official-02",
            "delprove_part": "del1",
            "question": "A: En invitation. Situation: Du vil gerne invitere dine venner på restaurant. Du vil skrive en invitation. Opgave: Skriv invitationen. Du skal fortælle: hvorfor du gerne vil invitere dine venner på restaurant; hvor og hvornår I skal mødes (sted, dato og tidspunkt); lidt om restauranten og den mad, I skal have; hvad I skal lave, efter at I har spist. Du skal begynde og afslutte invitationen på en passende måde.",
            "question_description": "PD2 skriftlig del, delprøve 1 (maj-juni 2021 prøvegrundlag). Bedømmes sammen med delprøve 2 efter 7-trins-skalaen (-3 til 12), B1-niveau.",
            "answer": "Hej: mine venner\nJeg vil gerne gift næste månd\nI invitere til i restaurant d. 20.05.2021.\nRestaurat ligger i Arhus. Den hedder Sarayi Kabab.\nDer er adresse rådhusgade 17, 1.\nI restaurant laver kabab og kylling og lam\nog drikker cola og cafe og laver te med\ndesert.\nVi spiser sammen mad efter danske.\nJeg håbe kommer til mig alle sammen.\nI interesserede kontakt (fornavn) på tilfon:\n12 34 56 78 eller sind til mig en mail.",
        },
    },
    "pd2_mock_del2": {
        "summary": "PD2 mock — step 2 of 2 (Del 2, en e-mail)",
        "description": (
            "Official PD2 ground truth, combined grade 02 — same case as the "
            "Del 1 example above, same mock_id. Sending this triggers "
            "background grading of BOTH halves and the deterministic "
            "combination — poll GET /evaluation/pd2-official-02 for the "
            "final result once status moves past 'pending'."
        ),
        "value": {
            "eval_id": "pd2-official-02-del2",
            "exam_type": "PD2",
            "submission_mode": "mock",
            "mock_id": "pd2-official-02",
            "delprove_part": "del2",
            "question": "En e-mail. Situation: Du har fået en e-mail fra din danske ven Adam. I e-mailen skriver han bl.a.: '...Nu har du jo boet i Danmark i et stykke tid. Hvordan er det at bo i Danmark? Og hvordan har du fundet nye venner her?' Opgave: Skriv et svar til Adam og fortæl, hvordan det er at bo i Danmark, og hvordan du har fundet nye venner her. Du skal skrive minimum 100 ord.",
            "question_description": "PD2 skriftlig del, delprøve 2 (maj-juni 2021 prøvegrundlag). Delprøve 2 er afgørende for den endelige karakter.",
            "answer": "Kære: Adam\nHvordan har du det?\nJeg skriver til dig, fordi jeg fortæller dig\nhvordan jeg bor i Danmark.\nJeg bor i (navn på by) siden 5 år og jeg er meget glad\n(Navn på by) er rolig og smukt\nog jeg har lieg fået arbejde tæt på (navn på by)\nJeg har nu meget venner i arbejder\nmin venner hjælp mig altiv og læer mig\nhvordan lever det og det. Jeg arbejder i mobel fabriker\nogså snakker nogl gang på dansk\nfordi vi altid travlt\nVi mødes i weekenden og deler spieser\nsammen og drikke og danser.\nJeg har meget forskellige venner\nde kommer fra forskellig lande\nJeg bor i lejlighed mit min kone mit datter\nsammen med.\nJeg har kendt til neabor.\nJeg er dijlig, mine neabor hjlp mig altid\nog går sammen til supermart.\nvi køber meget ting\nog kontakt mig hver dag snakker\nsammen hvor skal vi går i dag?\neller Hvad skal lever?\nog gå tur 3 gang om måned\nJeg har glemt fortæeller om arbejde\nJeg er aktiv og stabil og dygtigt\ni arbejde\nMin chef meget glad og elske mig\nTak for du høre mig\nKærlig hilsen\n(fornavn)",
        },
    },
    "pd3_mock_del1_02": {
        "summary": "PD3 mock — grade-02 case, step 1 of 2 (Del 1, e-mail to Mia)",
        "description": (
            "Official PD3 ground truth, combined grade 02 (bedømmelsesvejledning "
            "August 2024, sommerprøven 2022) — a different graded case from the "
            "grade-04 pd3_mock_del1/del2 examples above, useful for re-testing "
            "temperature/consistency at the 02 boundary. Response comes back "
            "'awaiting_other_part'. Send the Del 2 example next with the SAME "
            "mock_id to trigger grading."
        ),
        "value": {
            "eval_id": "pd3-official-02-del1",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-official-02",
            "delprove_part": "del1",
            "question": "Situation: Du har fået en mail fra din danske ven Mia. Hun har været udstationeret i New York i tre år for et dansk firma, og nu skal hun snart hjem til Odense sammen med sin familie. Mailen (uddrag): '...Tak for snakken i sidste uge, det var hyggeligt at tale med dig på Skype. Jeg er meget spændt på at høre om din køreprøve. Hvordan gik det? Som du ved, er vores tid i New York snart slut, og vi skal hjem til Odense om tre uger. Jeg skal arbejde sammen med nogle kolleger, jeg ikke kender. Hvad synes du, jeg skal gøre for at få en god start på arbejdet? Børnene er jo blevet teenagere, og de er meget kede af at sige farvel til deres venner her i New York. Jeg er spændt på, om de kan holde kontakten. Tror du, det er muligt? Jens har fået nyt arbejde i Danmark, som han glæder sig til at komme i gang med efter 3 år som hjemmegående. Det nye job ligger dog 30 km fra Odense, så vi overvejer at købe en ekstra bil, men han kan også tage toget. Hvad ville du gøre?' Opgave: Skriv et svar til din ven. Tak for mailen. Kom ind på de understregede dele i mailen. Foreslå, at I mødes, når Mia kommer hjem.",
            "question_description": "PD3 skriftlig del, delprøve 1 (sommerprøven 2022 prøvegrundlag). Bedømmes sammen med delprøve 2 efter 7-trins-skalaen (-3 til 12), B2-niveau.",
            "answer": "Kære Mia\n\nTak for din e-mail. Det var godt at høre fra dig.\n\nJeg er glad for at du husker min køreprøve. Jeg var spændt på at få en kørekort sidste uge. Jeg bestod det i først gang. Jeg var meget nervøst. Nu kan vi tage på min nye bil til forskellige steder sammen.\n\nJeg er meget glad for, at høre dig og dine familie flytter til Danmark igen. Du spurgte mig om hvad du skal gøre for at få en god start på arbejde. Jeg ved, at det er lidt svært at arbejde med nye kolleger, men du skal være ikke bekymre over det. Du er meget venlig og hjælpsom person. Jeg er sikker på at du tilpasser hurtig med nye kolleger og nye arbejdsmiljø. Derudover kan du drikke kaffe og snakke med dem om pausen og du kan holde fest hjem hos dig, eller en restaurant.\n\nI forhold til kontakten med børnenes venner, synes jeg at der er forskellige app i sociale medier, f.eks. whatsapp, facebook osv. De kan snakke med dem. Desuden kan de rejse til Newyork i sommerferien.\n\nTillykke til Jens for et nyt job. Med hensyn til at tage toget eller købe en ekstra bil at tage på arbejdsplads, er det bedre at købe en ny bil. Nogle gange skal vi vente på tog for en lang tid. Det bliver også forsinket nogle gange.\n\nDet er alt for nu. Skal vi mødes hjem hos mig når du kommer hjem fra Newyork? Skriv eller ring til mig, hvis de passer dig. Jeg glæder mig til at møde dig.\n\nMange hilsner\nNN",
        },
    },
    "pd3_mock_del2_02": {
        "summary": "PD3 mock — grade-02 case, step 2 of 2 (Del 2B, en attraktiv arbejdsplads)",
        "description": (
            "Official PD3 ground truth, combined grade 02 — same case as "
            "pd3_mock_del1_02, same mock_id. Sending this triggers background "
            "grading of BOTH halves and the deterministic combination — poll "
            "GET /evaluation/pd3-official-02 for the final result once status "
            "moves past 'pending'."
        ),
        "value": {
            "eval_id": "pd3-official-02-del2",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-official-02",
            "delprove_part": "del2",
            "question": "2B: En attraktiv arbejdsplads. Der er forskellige faktorer, der kan have betydning for, hvor attraktiv en arbejdsplads er for medarbejderne. Eksempler: Godt samarbejde mellem kolleger; Fleksible arbejdstider; Indflydelse på egne arbejdsopgaver; God løn; Mulighed for efteruddannelse; Personalegoder. Opgave: Fortæl kort om en attraktiv arbejdsplads, du har været på eller har hørt om. Kommentér en eller to af faktorerne fra listen. Vurdér, om det er lederens personlige kvalifikationer eller faglige kvalifikationer, der er vigtigst for, om en arbejdsplads er attraktiv (dette punkt skal udgøre ca. 50% af den samlede besvarelse). Du skal skrive minimum 200 ord.",
            "question_description": "PD3 skriftlig del, delprøve 2B (sommerprøven 2022 prøvegrundlag). Delprøve 2 er afgørende for den endelige karakter, især punkt 3 (ca. 50% af besvarelsen), hvor B2-niveauet afprøves.",
            "answer": "En god arbejdsplads er en vigtig til medarbejderne for at fungere vel på arbejde. Jeg arbejdede i et attraktiv arbejdsplads. Min chef var meget venlig og gav mig flere personalegoder f.eks. fleksible arbejdstider, sygesikring osv.\n\nDer kan være flere forskellige faktorer for at skabe arbejdsplads mere attraktiv til medarbejderne. En faktor kan være, at have godt samarbejde mellem kolleger. Jeg synes, at hvis ens kolleger er venlige og hjælpsomme, bliver man bedre at arbejde og altid bliver motivatet.\n\nJeg vurderer, at lederens både personlige kvalifikationer og faglige kvalifikationer er vigtige for at tiltrække ansatter på virksomhederne. De skal skabe venlig miljø i arbejdsplads. Exempelvis er de altid sur, ingen kan lide dem og ikke vil gerne arbejde med dem. De har ansvaret for at styr på alt i et firm. Hvis ansatte har nogle problemer, kan man få løsning fra deres chef. De skal have en viden om hvordan de skal fungere firm på bedre måde. Hvis de har oplevelse på det, bliver arbejdstager mere imponeret af det og viser respekt til dem. Og desuden søger ansatte ikke for et nyt job. På den måde kan fabrik spare penge ved undgå træne til den nye folk.\n\nAlt i alt mener jeg, at lederen spiller en vigtig rolle for at tiltrække lønmodtager og tilskynde dem at lave bedre.",
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
