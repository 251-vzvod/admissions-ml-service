from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.input import CandidateInput

RAW_DIR = ROOT / "data" / "ml_workbench" / "raw" / "generated" / "gap_fill_batch_v7"
PACK_DIR = ROOT / "data" / "ml_workbench" / "processed" / "annotation_packs" / "gap_fill_batch_v7"

RAW_JSONL = RAW_DIR / "gap_fill_batch_v7_api_input.jsonl"
GEN_MANIFEST_JSONL = RAW_DIR / "gap_fill_batch_v7_generation_manifest.jsonl"
SUMMARY_JSON = RAW_DIR / "gap_fill_batch_v7_summary.json"

PACK_JSONL = PACK_DIR / "gap_fill_batch_v7_annotation_pack.jsonl"
PACK_JSON = PACK_DIR / "gap_fill_batch_v7_annotation_pack.json"
PACK_TABLE_CSV = PACK_DIR / "gap_fill_batch_v7_annotation_pack_table.csv"
PACK_MANIFEST_JSON = PACK_DIR / "gap_fill_batch_v7_annotation_pack_manifest.json"

SLICE_TARGETS = {
    "authenticity_manual_review_cases": 18,
    "insufficient_evidence_but_valid_cases": 16,
    "no_interview_cases_across_quality_levels": 16,
    "translated_or_mixed_thinking_english_cases": 12,
    "support_needed_but_not_hidden_star_cases": 10,
}

TEXT_LENGTH_TARGETS = {"short": 12, "medium": 34, "long": 26}

QUESTION_VARIANTS: dict[str, list[str]] = {
    "fit": [
        "Why does inVision U feel like a good learning environment for you?",
        "What about the university environment at inVision U fits how you learn?",
        "Why do you think you would grow in the kind of environment inVision U offers?",
    ],
    "study": [
        "What do you want to study or explore first at university?",
        "Which subjects do you want to begin with, even if your direction is still forming?",
        "What kind of academic path do you want to test first at inVision U?",
    ],
    "peers": [
        "What would you contribute to your peers?",
        "How do you think classmates might experience you in group work?",
        "What kind of contribution would you make to other students around you?",
    ],
    "feedback": [
        "Tell us about difficult feedback and what changed after you heard it.",
        "When did feedback change how you worked?",
        "Describe a moment when criticism was useful and what you changed next.",
    ],
    "support": [
        "What support might help you in your first year?",
        "Where do you think you may need guidance or structure at university?",
        "What kind of support would make your transition to university more realistic?",
    ],
    "different_people": [
        "How do you work with people who are different from you?",
        "What do you do when a team includes people with very different working styles?",
        "How do you handle group work with classmates who think in a different way?",
    ],
    "unfinished": [
        "What is something you started but did not fully finish?",
        "Tell us about a plan or project that did not continue the way you expected.",
        "What did you begin with good intentions but fail to carry through completely?",
    ],
    "local_issue": [
        "What local issue matters to you and why?",
        "Which issue from your school or city do you keep noticing?",
        "What problem close to your daily life would you want to understand better?",
    ],
    "uncertainty": [
        "When did you act even though you were unsure?",
        "Describe a time when you moved forward without full confidence.",
        "What have you done despite not being fully certain about yourself yet?",
    ],
    "learning_env": [
        "What kind of environment helps you learn best?",
        "What conditions usually help you learn in a serious way?",
        "What kind of classroom or peer environment brings out your best work?",
    ],
    "self_unknown": [
        "What do you still not know about yourself?",
        "What part of your future self still feels undecided?",
        "What are you still trying to understand about your own direction?",
    ],
}

LABEL_LEAKAGE_TERMS = [
    "authenticity_manual_review_cases",
    "insufficient_evidence_but_valid_cases",
    "no_interview_cases_across_quality_levels",
    "translated_or_mixed_thinking_english_cases",
    "support_needed_but_not_hidden_star_cases",
    "intended_gap_slice",
    "intended_ambiguity",
    "generator_notes",
    "noise_profile",
    "manual_review_required",
    "insufficient_evidence",
    "hidden star",
    "hidden genius",
]

BOILERPLATE_TERMS = [
    "admissions committee",
    "consideration of admission",
    "esteemed educational institution",
    "favorable response",
    "committee will understand",
    "thank you for considering my application",
    "thank you for your consideration",
]

UNIVERSITY_MISMATCH_TERMS = [
    "accelerator",
    "fellowship",
    "startup program",
    "company",
    "employer",
    "recruiter",
    "job role",
]

FRAUD_CARTOON_TERMS = [
    "fake",
    "forged",
    "forgery",
    "made up",
    "lied on purpose",
    "stole",
]

PLACEHOLDER_RE = re.compile(r"\{[^{}]{1,32}\}|\[[^\[\]]{1,32}\]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіҢңҒғҚқҮүҰұҺһ]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SPACE_RE = re.compile(r"[ \t]+")
NEWLINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CandidateSeed:
    candidate_id: str
    intended_gap_slice: str
    slice_variant: str
    intended_ambiguity: str
    text_length: str
    english_type: str
    english_score: float
    school_type: str
    school_score: float
    city: str
    opening: str
    school_context: str
    study_interest: str
    second_interest: str
    example: str
    example_detail: str
    local_issue: str
    contribution: str
    support_need: str
    uncertainty: str
    qa_profile: str
    interview_mode: str
    interview_hook: str
    completion_rate: float
    returned_to_edit: bool
    skipped_optional_questions: int


def clean(text: str) -> str:
    text = dedent(text).strip()
    text = SPACE_RE.sub(" ", text)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    return NEWLINES_RE.sub("\n\n", text)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def classify_length(text: str) -> str:
    words = word_count(text)
    if words <= 90:
        return "short"
    if words <= 220:
        return "medium"
    return "long"


def candidate_num(seed: CandidateSeed) -> int:
    return int(seed.candidate_id.rsplit("_", 1)[-1])


SEEDS: list[CandidateSeed] = []

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_001",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="polished_thin",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="private tutoring",
            english_score=87.0,
            school_type="City school diploma",
            school_score=86.0,
            city="Astana",
            opening="My application probably sounds more settled on paper than the process actually felt while I was writing it.",
            school_context="I study in a regular city school where I often ended up helping teachers format class materials rather than leading a visible club.",
            study_interest="economics",
            second_interest="public policy",
            example="Last winter I put together a set of shared revision notes and meeting times for students who kept missing the after-school economics circle.",
            example_detail="People later referred to it as a project, but in reality it was a fairly small coordination effort and I am aware that my writing can make it sound larger than it was.",
            local_issue="students in my district often depend on private tutoring because ordinary school guidance is uneven",
            contribution="I can contribute calm preparation, note-sharing, and the habit of making a group plan visible to everyone.",
            support_need="speaking more naturally when I am asked follow-up questions without preparation",
            uncertainty="I moved between economics and policy language during grade 11, so a few parts of my application may sound more certain than my actual exploration has been.",
            qa_profile="generic_polished",
            interview_mode="weaker_nervous",
            interview_hook="In conversation I sound less composed than in writing, which is not ideal but it is true.",
            completion_rate=0.9,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_002",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="timeline_fuzzy",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=81.0,
            school_type="Lyceum certificate",
            school_score=89.0,
            city="Karagandy",
            opening="I did not come to this application through one clean turning point, and that unevenness is probably visible in how I describe myself.",
            school_context="At my lyceum I spent more time on science coursework than on formal leadership roles, but I kept returning to environmental questions around our river area.",
            study_interest="environmental data",
            second_interest="urban studies",
            example="Around the end of grade 9, or maybe the very start of grade 10, I began joining weekend water checks with a teacher and two classmates.",
            example_detail="I can describe the routine and what we measured, but I cannot make it sound bigger than it was because some months we met regularly and some months we stopped during exams.",
            local_issue="industrial dust and litter near ordinary walking routes get discussed by adults, but students rarely turn those conversations into basic observation or records",
            contribution="I would bring patience in fieldwork, reliable note-taking, and a willingness to do the less visible parts of group assignments.",
            support_need="shaping scattered interest into a clearer academic path without pretending the answer is already fixed",
            uncertainty="My timeline is honest but not perfectly sharp, and that may make the application feel less tidy than I originally hoped.",
            qa_profile="mixed_specificity",
            interview_mode="clearer_plain",
            interview_hook="If I explain this aloud, it sounds simpler and less impressive, but maybe also more accurate.",
            completion_rate=0.86,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_003",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="claims_weak_grounding",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="language center",
            english_score=83.0,
            school_type="Standard school diploma",
            school_score=84.0,
            city="Aktobe",
            opening="The strongest version of my motivation sounds exciting, but the honest version is smaller and less finished.",
            school_context="My school does not have a strong technical club, so a lot of what I did around computers happened in improvised ways after class.",
            study_interest="computer science",
            second_interest="learning technology",
            example="I described one of my school efforts as a study platform for younger students.",
            example_detail="In truth it was mostly a carefully organized folder of quiz links, short instructions, and a timetable, useful but not the same as building a real platform from scratch.",
            local_issue="many younger students lose confidence quickly when digital homework systems are confusing even though the problem is usually simple",
            contribution="I can contribute practical troubleshooting, patient explanations, and the habit of checking whether instructions are actually usable.",
            support_need="stronger technical fundamentals so that my interest does not run ahead of my real skill level",
            uncertainty="I am sincere about the interest, but I also know some of my wording has been more ambitious than the underlying evidence.",
            qa_profile="thin_specifics",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.84,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_004",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="section_mismatch",
            intended_ambiguity="hard",
            text_length="long",
            english_type="private tutoring",
            english_score=88.0,
            school_type="City school diploma",
            school_score=87.0,
            city="Shymkent",
            opening="I noticed while filling the application that I describe myself one way in formal writing and another way when I speak more freely.",
            school_context="In school I have usually been the student who can organize a discussion or collect opinions from classmates, even when I am not the person with the strongest grades in every subject.",
            study_interest="psychology",
            second_interest="education analytics",
            example="One example was a small peer reflection group before final exams where I gathered common stress points and shared them with our homeroom teacher.",
            example_detail="It helped a little, but it was not a formal research activity and I am still deciding whether my real direction is psychology, education systems, or something between them.",
            local_issue="students speak openly about pressure and uncertainty with each other, but those conversations rarely become something the school can use in a structured way",
            contribution="I could contribute listening skills, careful summaries, and a steady role in group discussions where quieter people might otherwise stay silent.",
            support_need="clearer academic framing, because my interests connect in my head faster than they connect on paper",
            uncertainty="My letter can sound like I already chose psychology, while some of my answers drift toward systems and planning because the decision is honestly still open.",
            qa_profile="mismatch_drifting",
            interview_mode="clearer_plain",
            interview_hook="In person I usually admit sooner that my focus is still moving than I do in polished writing.",
            completion_rate=0.92,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_005",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="generic_polished",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school + apps",
            english_score=79.0,
            school_type="Regional school certificate",
            school_score=83.0,
            city="Atyrau",
            opening="What attracts me to inVision U is the possibility of learning in a serious community that values initiative, reflection, and responsible growth.",
            school_context="Most of my school years were stable and disciplined rather than visibly impressive.",
            study_interest="management",
            second_interest="social innovation",
            example="I took part in school planning tasks and tried to be useful when group responsibilities needed someone dependable.",
            example_detail="That summary is true, though I understand it still leaves a human reader with many reasonable questions.",
            local_issue="many students know how to prepare for exams but not how to connect learning with real problems around them",
            contribution="I would contribute reliability and positive cooperation.",
            support_need="turning broad motivation into more concrete academic direction",
            uncertainty="Some of my answers are smoother than my actual evidence level.",
            qa_profile="generic_polished",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.73,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_006",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="style_gap",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="language center",
            english_score=82.0,
            school_type="City school diploma",
            school_score=80.0,
            city="Pavlodar",
            opening="I revise formal writing many times, so the page version of me is usually more elegant than the speaking version.",
            school_context="My grades are strongest in biology-related subjects, and I have spent most of my extra energy helping with practical tasks rather than public events.",
            study_interest="public health",
            second_interest="communication",
            example="For several weekends I helped a relative at a small clinic desk by guiding visitors to rooms and checking whether forms were filled correctly.",
            example_detail="It was basic work, not a medical role, but it made me notice how often stress grows from bad explanation rather than from the original problem.",
            local_issue="families often lose time in health settings because information is scattered or explained in a way that assumes too much confidence",
            contribution="I can contribute steadiness, respectful communication, and practical attention to details that other people may ignore.",
            support_need="speaking with confidence in live discussions because I still over-prepare before saying simple things aloud",
            uncertainty="The gap between my polished written voice and my ordinary spoken voice is real and sometimes larger than I want it to be.",
            qa_profile="measured",
            interview_mode="weaker_nervous",
            interview_hook="I know my interview answers may sound much less refined than the letter, even though the motivation itself is still sincere.",
            completion_rate=0.88,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
    ]
)


QUESTION_SET_BANK: dict[str, list[tuple[str, ...]]] = {
    "authenticity_manual_review_cases": [
        ("fit", "study", "local_issue", "support"),
        ("fit", "peers", "feedback", "uncertainty"),
        ("study", "different_people", "unfinished", "support"),
        ("fit", "local_issue", "self_unknown", "peers"),
    ],
    "insufficient_evidence_but_valid_cases": [
        ("fit", "study"),
        ("fit", "support", "peers"),
        ("study", "learning_env", "support"),
        ("fit", "local_issue", "support"),
    ],
    "no_interview_cases_across_quality_levels": [
        ("fit", "study", "local_issue", "peers"),
        ("fit", "support", "uncertainty", "learning_env"),
        ("study", "feedback", "local_issue", "peers"),
        ("fit", "different_people", "unfinished", "support"),
    ],
    "translated_or_mixed_thinking_english_cases": [
        ("fit", "study", "local_issue", "peers"),
        ("study", "support", "uncertainty", "learning_env"),
        ("fit", "different_people", "local_issue", "self_unknown"),
        ("study", "peers", "support", "feedback"),
    ],
    "support_needed_but_not_hidden_star_cases": [
        ("fit", "support", "peers", "feedback"),
        ("study", "support", "learning_env", "different_people"),
        ("fit", "peers", "support", "self_unknown"),
        ("fit", "study", "support", "uncertainty"),
    ],
}

TARGET_WORD_RANGES = {
    "short": (45, 90),
    "medium": (115, 220),
    "long": (230, 340),
}


def first_sentence(text: str) -> str:
    match = re.search(r"^.*?[.!?](?:\s|$)", text.strip())
    return match.group(0).strip() if match else text.strip()


def lower_first(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].lower() + text[1:]


def choose_question_text(seed: CandidateSeed, key: str) -> str:
    variants = QUESTION_VARIANTS[key]
    return variants[candidate_num(seed) % len(variants)]


def fit_sentence(seed: CandidateSeed) -> str:
    options = [
        f"I am applying to inVision U because I want to study {seed.study_interest} in a setting that is more serious and collaborative than my current one.",
        f"inVision U appeals to me as a university where I could begin with {seed.study_interest} and still keep room to explore {seed.second_interest}.",
        f"What pulls me toward inVision U is the chance to test {seed.study_interest} with stronger peers, clearer feedback, and more demanding study habits.",
        f"I see inVision U as a place where {seed.study_interest} can be studied with structure while I keep refining how {seed.second_interest} connects to it.",
    ]
    return options[candidate_num(seed) % len(options)]


def contribution_sentence(seed: CandidateSeed) -> str:
    return f"What I could bring to peers is {lower_first(seed.contribution)}"


def support_sentence(seed: CandidateSeed) -> str:
    return f"What I would likely need at the beginning is {lower_first(seed.support_need)}"


def expansion_pool(seed: CandidateSeed) -> list[str]:
    common = [
        f"I want university study to test this interest with more method than I currently have.",
        f"I also think I would learn better in a peer environment where people take questions seriously and do not only perform certainty.",
        f"I am not applying as someone fully formed, only as someone ready for harder structure and clearer feedback.",
    ]
    if seed.intended_gap_slice == "authenticity_manual_review_cases":
        common.append("I know a careful reader may still want stronger grounding than I can fully provide right now.")
    elif seed.intended_gap_slice == "insufficient_evidence_but_valid_cases":
        common.append("At this stage the application carries more intention than proof, and I understand that clearly.")
    elif seed.intended_gap_slice == "translated_or_mixed_thinking_english_cases":
        common.append("Maybe I explain it in a simple way, but this is still the real point for me.")
    elif seed.intended_gap_slice == "support_needed_but_not_hidden_star_cases":
        common.append("Support would help me use effort better, not replace effort.")
    else:
        common.append("The direction feels real for me even when the current evidence is not dramatic.")
    return common


def adjust_letter_length(seed: CandidateSeed, paragraphs: list[str]) -> str:
    low, high = TARGET_WORD_RANGES[seed.text_length]
    working = [clean(item) for item in paragraphs if clean(item)]

    if seed.text_length == "short" and len(working) >= 2:
        working = [working[0], first_sentence(working[1])]

    count = word_count("\n\n".join(working))
    extra_idx = 0
    extras = expansion_pool(seed)
    while count < low and extra_idx < len(extras):
        working.append(extras[extra_idx])
        extra_idx += 1
        count = word_count("\n\n".join(working))

    while count > high and len(working) > 1:
        if word_count(working[-1]) <= 18:
            working.pop()
        else:
            working[-1] = first_sentence(working[-1])
            if word_count(working[-1]) <= 18 and len(working) > 1:
                working.pop()
        count = word_count("\n\n".join(working))

    return clean("\n\n".join(working))


def literalize(text: str) -> str:
    updated = text
    replacements = {
        "important to me": "important for me",
        "makes sense to me": "feels logical for me",
        "grew from": "came from",
        "became important": "became important",
        "I want to": "I want to",
        "I am interested": "I am interested",
    }
    for before, after in replacements.items():
        updated = updated.replace(before, after)
    return updated


def base_answer(seed: CandidateSeed, key: str) -> str:
    if key == "fit":
        return f"inVision U fits me because I need a place with serious peers, clearer feedback, and room to study {seed.study_interest} without pretending my path is already finished."
    if key == "study":
        return f"I want to begin with {seed.study_interest} and keep some space for {seed.second_interest} until I understand which part stays strongest after real coursework."
    if key == "peers":
        return seed.contribution
    if key == "feedback":
        return "A teacher once told me that my ideas become clearer when I move from one concrete example first and only then to the bigger conclusion. Since then I try to start from evidence instead of only intention."
    if key == "support":
        return f"The support that would help most is {lower_first(seed.support_need)}"
    if key == "different_people":
        return "When people work differently, I usually start by listening and making the task itself clearer. It reduces unnecessary tension and helps quieter people take a role earlier."
    if key == "unfinished":
        return "I tried to extend one small effort into something more regular, but exams and weak planning stopped the momentum. It taught me that good intention is not enough without structure."
    if key == "local_issue":
        return f"The issue that stays with me is that {seed.local_issue}. I want to study it with better methods than I have now."
    if key == "uncertainty":
        return f"I moved forward even while uncertain because {lower_first(seed.example)} The interest became clearer through action before it became clear in words."
    if key == "learning_env":
        return "I learn best where expectations are clear, questions are welcome, and classmates take the work seriously without turning everything into competition."
    if key == "self_unknown":
        return f"I am still trying to understand whether my path will stay centered on {seed.study_interest} or keep leaning toward {seed.second_interest}, and I think university is where that answer can become more honest."
    raise ValueError(f"Unknown question key: {key}")


def shape_answer(seed: CandidateSeed, key: str, answer: str) -> str:
    profile = seed.qa_profile
    text = answer

    if profile == "generic_polished":
        text = first_sentence(answer) + " I see this as part of my wider growth and contribution at university."
    elif profile == "thin_specifics":
        text = first_sentence(answer) + " The example is real, though still quite limited."
    elif profile == "mismatch_drifting":
        if key in {"fit", "study", "self_unknown"}:
            text = f"I talk about {seed.study_interest}, but I also keep returning to {seed.second_interest}, which is why my direction may sound slightly split."
        else:
            text = answer
    elif profile == "mildly_drifting":
        text = first_sentence(answer) + " I am still putting the full connection into words."
    elif profile == "thin_partial":
        text = first_sentence(answer) + " I cannot answer this much more fully yet."
    elif profile == "thin_basic":
        text = first_sentence(answer)
    elif profile == "repetitive_generic":
        text = "I want to grow, learn seriously, and contribute responsibly. " + first_sentence(answer)
    elif profile == "thin_examples":
        text = first_sentence(answer) + " I do not have a stronger example yet."
    elif profile == "long_vague":
        text = answer + " I know this still stays broader than ideal."
    elif profile == "grounded":
        text = answer
    elif profile == "borderline":
        text = answer + " The example is not large, but it is real."
    elif profile == "translated_clear":
        text = literalize(answer) + " I say it in simple way, but this is the real point for me."
    elif profile == "translated_reflective":
        text = literalize(answer) + " Maybe sentence sounds a little direct, still this is how I understand it."
    elif profile == "translated_thin":
        text = literalize(first_sentence(answer)) + " This is my answer now."
    elif profile == "support_direct":
        text = answer + " I prefer to say this early and not only after problems appear."
    elif profile == "measured":
        text = first_sentence(answer)
    elif profile == "mixed_specificity":
        text = answer + " Some parts are easier for me to describe than others."

    return clean(text)


def pick_question_keys(seed: CandidateSeed) -> list[str]:
    options = QUESTION_SET_BANK[seed.intended_gap_slice]
    selected = list(options[candidate_num(seed) % len(options)])

    if seed.text_length == "short":
        return selected[:2]
    if seed.qa_profile in {"thin_basic", "thin_partial"}:
        return selected[:3]
    if seed.text_length == "medium":
        return selected[:3]
    return selected[:4]


def build_questions(seed: CandidateSeed) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in pick_question_keys(seed):
        rows.append(
            {
                "question": choose_question_text(seed, key),
                "answer": shape_answer(seed, key, base_answer(seed, key)),
            }
        )
    return rows


def build_letter(seed: CandidateSeed) -> str:
    intro = f"{seed.opening} {fit_sentence(seed)}"
    context = f"{seed.school_context} {seed.example} {seed.example_detail}"
    issue = f"A local issue that keeps returning in my mind is that {seed.local_issue}."
    contribution = contribution_sentence(seed)
    support = support_sentence(seed)
    uncertainty = seed.uncertainty

    if seed.intended_gap_slice == "authenticity_manual_review_cases":
        paragraphs = [intro, context, issue, uncertainty, f"{contribution} {support}"]
    elif seed.intended_gap_slice == "insufficient_evidence_but_valid_cases":
        paragraphs = [intro, context, f"{uncertainty} {support}"]
    elif seed.intended_gap_slice == "no_interview_cases_across_quality_levels":
        paragraphs = [intro, context, issue, f"{contribution} {support}", uncertainty]
    elif seed.intended_gap_slice == "translated_or_mixed_thinking_english_cases":
        paragraphs = [intro, context, issue, f"{contribution} {support}", uncertainty]
    elif seed.intended_gap_slice == "support_needed_but_not_hidden_star_cases":
        paragraphs = [intro, context, issue, f"{contribution} {support}", uncertainty]
    else:
        raise ValueError(seed.intended_gap_slice)

    if seed.text_length == "short":
        chosen = [paragraphs[0], paragraphs[1]]
    elif seed.text_length == "medium":
        chosen = [paragraphs[0], paragraphs[1], paragraphs[-1]]
    else:
        chosen = paragraphs

    if seed.slice_variant in {"very_short", "thin_no_interview"}:
        chosen = chosen[:2] if seed.text_length != "long" else chosen[:4]

    if seed.qa_profile.startswith("translated"):
        chosen = [literalize(item) for item in chosen]

    return adjust_letter_length(seed, chosen)


def build_interview(seed: CandidateSeed) -> str:
    mode = seed.interview_mode
    if mode == "none":
        return ""

    if mode == "weaker_nervous":
        text = (
            f"{seed.interview_hook} I still want to study {seed.study_interest}, but I explain it in a less organized way when I am put on the spot. "
            f"The example I trust most is that {lower_first(seed.example)}"
        )
    elif mode == "clearer_plain":
        text = (
            f"{seed.interview_hook} The simplest version is that I care about {seed.study_interest} because {seed.local_issue}. "
            f"I am not presenting huge achievements. Mostly I learned through small things like when {lower_first(seed.example)}"
        )
    elif mode == "balanced_plain":
        text = (
            f"{seed.interview_hook} My experience is mostly small-scale, like when {lower_first(seed.example)} "
            f"What I can offer is {lower_first(seed.contribution)}"
        )
    elif mode == "mixed_casual":
        text = (
            f"{seed.interview_hook} On paper I sound more finished than I am. In normal conversation I would say the motivation is real, the examples are smaller, and I still need {lower_first(seed.support_need)}"
        )
    elif mode == "translated_oral":
        text = literalize(
            f"{seed.interview_hook} I want to study {seed.study_interest} because {seed.local_issue}. "
            f"My experience is mostly from small real situations, like when {lower_first(seed.example)}"
        ) + " I speak simply, but for me the meaning is stable."
    elif mode == "support_candid":
        text = (
            f"{seed.interview_hook} I think I can handle serious coursework, but I would probably need {lower_first(seed.support_need)} "
            f"At the same time, I can already bring {lower_first(seed.contribution)}"
        )
    else:
        raise ValueError(f"Unknown interview mode: {mode}")

    return clean(text)


def manifest_signals(seed: CandidateSeed) -> list[str]:
    if seed.intended_gap_slice == "authenticity_manual_review_cases":
        return ["plausible academic motivation", "usable application quality", seed.slice_variant.replace("_", " ")]
    if seed.intended_gap_slice == "insufficient_evidence_but_valid_cases":
        return ["valid but thin submission", "sincere intent", "limited evidence density"]
    if seed.intended_gap_slice == "no_interview_cases_across_quality_levels":
        return [seed.slice_variant.replace("_", " "), "written-only signal", "mixed evidence level"]
    if seed.intended_gap_slice == "translated_or_mixed_thinking_english_cases":
        return ["understandable English submission", "local-context reasoning", "translated-thinking rhythm"]
    return ["modest promise", "coachable attitude", "visible support need"]


def manifest_risks(seed: CandidateSeed) -> list[str]:
    if seed.intended_gap_slice == "authenticity_manual_review_cases":
        return ["grounding may lag behind framing", "human verification helpful"]
    if seed.intended_gap_slice == "insufficient_evidence_but_valid_cases":
        return ["too little evidence to judge strongly", "initiative depth unclear"]
    if seed.intended_gap_slice == "no_interview_cases_across_quality_levels":
        return ["no interview corroboration", "written interpretation burden"]
    if seed.intended_gap_slice == "translated_or_mixed_thinking_english_cases":
        return ["literal phrasing may obscure merit", "language-form mismatch risk"]
    return ["transition support need", "confidence or language adaptation risk"]


def manifest_noise(seed: CandidateSeed) -> list[str]:
    return [seed.qa_profile, seed.interview_mode, seed.text_length]


def generator_notes(seed: CandidateSeed) -> str:
    return (
        f"{seed.intended_gap_slice} case with variant={seed.slice_variant}, "
        f"qa_profile={seed.qa_profile}, interview_mode={seed.interview_mode}."
    )


def ordered_seeds() -> list[CandidateSeed]:
    return sorted(SEEDS, key=candidate_num)


def sanitize_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_id": record["candidate_id"],
        "structured_data": record["structured_data"],
        "text_inputs": record["text_inputs"],
        "behavioral_signals": record["behavioral_signals"],
    }


def reviewer_table_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        education = ((record.get("structured_data") or {}).get("education") or {})
        english = education.get("english_proficiency") or {}
        school = education.get("school_certificate") or {}
        text_inputs = record.get("text_inputs") or {}
        behavioral = record.get("behavioral_signals") or {}
        rows.append(
            {
                "candidate_id": record.get("candidate_id"),
                "english_proficiency_type": english.get("type"),
                "english_proficiency_score": english.get("score"),
                "school_cert_type": school.get("type"),
                "school_cert_score": school.get("score"),
                "motivation_letter_length": len(str(text_inputs.get("motivation_letter_text") or "")),
                "num_questions": len(text_inputs.get("motivation_questions") or []),
                "has_interview": int(bool(str(text_inputs.get("interview_text") or "").strip())),
                "completion_rate": behavioral.get("completion_rate"),
                "returned_to_edit": int(bool(behavioral.get("returned_to_edit"))),
                "skipped_optional": behavioral.get("skipped_optional_questions"),
            }
        )
    return rows


def find_near_duplicates(records: list[dict[str, object]]) -> list[list[str | float]]:
    assembled: list[tuple[str, str]] = []
    for record in records:
        text_inputs = record.get("text_inputs") or {}
        parts = [str(text_inputs.get("motivation_letter_text") or "")]
        for item in text_inputs.get("motivation_questions") or []:
            if isinstance(item, dict):
                parts.append(str(item.get("question") or ""))
                parts.append(str(item.get("answer") or ""))
        parts.append(str(text_inputs.get("interview_text") or ""))
        assembled.append((str(record.get("candidate_id")), re.sub(r"\s+", " ", " ".join(parts).lower()).strip()))

    pairs: list[list[str | float]] = []
    for left in range(len(assembled)):
        for right in range(left + 1, len(assembled)):
            left_id, left_text = assembled[left]
            right_id, right_text = assembled[right]
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= 0.92:
                pairs.append([left_id, right_id, round(ratio, 4)])
    return pairs


def build_records() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    base_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    raw_records: list[dict[str, object]] = []
    sanitized_records: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []

    for idx, seed in enumerate(ordered_seeds()):
        submitted_at = (base_time + timedelta(hours=4 * idx)).isoformat().replace("+00:00", "Z")
        raw = {
            "candidate_id": seed.candidate_id,
            "structured_data": {
                "education": {
                    "english_proficiency": {
                        "type": seed.english_type,
                        "score": seed.english_score,
                    },
                    "school_certificate": {
                        "type": seed.school_type,
                        "score": seed.school_score,
                    },
                }
            },
            "text_inputs": {
                "motivation_letter_text": build_letter(seed),
                "motivation_questions": build_questions(seed),
                "interview_text": build_interview(seed),
            },
            "behavioral_signals": {
                "completion_rate": seed.completion_rate,
                "returned_to_edit": seed.returned_to_edit,
                "skipped_optional_questions": seed.skipped_optional_questions,
            },
            "metadata": {
                "source": "gap_fill_batch_v7",
                "submitted_at": submitted_at,
                "scoring_version": None,
            },
        }
        raw = CandidateInput.model_validate(raw).model_dump(mode="json", exclude_none=False)
        sanitized = sanitize_record(raw)
        manifest = {
            "candidate_id": seed.candidate_id,
            "intended_gap_slice": seed.intended_gap_slice,
            "intended_ambiguity": seed.intended_ambiguity,
            "intended_primary_signals": manifest_signals(seed),
            "intended_primary_risks": manifest_risks(seed),
            "noise_profile": manifest_noise(seed),
            "generator_notes": generator_notes(seed),
        }
        raw_records.append(raw)
        sanitized_records.append(sanitized)
        manifest_records.append(manifest)

    return raw_records, sanitized_records, manifest_records


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    seeds = ordered_seeds()
    raw_records, sanitized_records, manifest_records = build_records()

    expected_ids = [f"syn_gap_v7_{idx:03d}" for idx in range(1, 73)]
    ids = [str(record.get("candidate_id")) for record in raw_records]
    sanitized_ids = [str(record.get("candidate_id")) for record in sanitized_records]

    slice_counts: dict[str, int] = {}
    ambiguity_counts: dict[str, int] = {}
    text_length_counts = {"short": 0, "medium": 0, "long": 0}
    text_length_mismatches: list[str] = []

    for seed, record in zip(seeds, raw_records):
        slice_counts[seed.intended_gap_slice] = slice_counts.get(seed.intended_gap_slice, 0) + 1
        ambiguity_counts[seed.intended_ambiguity] = ambiguity_counts.get(seed.intended_ambiguity, 0) + 1
        actual_length = classify_length(str((record.get("text_inputs") or {}).get("motivation_letter_text") or ""))
        text_length_counts[actual_length] += 1
        if actual_length != seed.text_length:
            text_length_mismatches.append(f"{seed.candidate_id}:{seed.text_length}->{actual_length}")

    interview_with = sum(
        1
        for record in raw_records
        if str(((record.get("text_inputs") or {}).get("interview_text") or "")).strip()
    )
    interview_counts = {
        "with_interview": interview_with,
        "without_interview": len(raw_records) - interview_with,
    }

    no_interview_seed_ids = {
        seed.candidate_id
        for seed in seeds
        if seed.intended_gap_slice == "no_interview_cases_across_quality_levels"
    }
    no_interview_empty_count = sum(
        1
        for record in raw_records
        if record["candidate_id"] in no_interview_seed_ids
        and not str(((record.get("text_inputs") or {}).get("interview_text") or "")).strip()
    )
    no_interview_quality_counts = dict(
        Counter(
            seed.slice_variant
            for seed in seeds
            if seed.intended_gap_slice == "no_interview_cases_across_quality_levels"
        )
    )

    completion_values = [
        float((record.get("behavioral_signals") or {}).get("completion_rate") or 0.0)
        for record in raw_records
    ]
    returned_to_edit_count = sum(
        1
        for record in raw_records
        if bool((record.get("behavioral_signals") or {}).get("returned_to_edit"))
    )
    skipped_values = [
        int((record.get("behavioral_signals") or {}).get("skipped_optional_questions") or 0)
        for record in raw_records
    ]
    skipped_histogram = dict(sorted(Counter(skipped_values).items()))
    completion_rate_bands = {
        "0.58-0.69": sum(1 for value in completion_values if 0.58 <= value < 0.70),
        "0.70-0.79": sum(1 for value in completion_values if 0.70 <= value < 0.80),
        "0.80-0.89": sum(1 for value in completion_values if 0.80 <= value < 0.90),
        "0.90-0.95": sum(1 for value in completion_values if 0.90 <= value <= 0.95),
    }

    near_duplicate_pairs = find_near_duplicates(sanitized_records)

    raw_allowed_fields = {"candidate_id", "structured_data", "text_inputs", "behavioral_signals", "metadata"}
    sanitized_allowed_fields = {"candidate_id", "structured_data", "text_inputs", "behavioral_signals"}
    manifest_allowed_fields = {
        "candidate_id",
        "intended_gap_slice",
        "intended_ambiguity",
        "intended_primary_signals",
        "intended_primary_risks",
        "noise_profile",
        "generator_notes",
    }

    sanitized_payload = json.dumps(sanitized_records, ensure_ascii=False)
    lowered_sanitized = sanitized_payload.lower()
    leakage_hits = [term for term in LABEL_LEAKAGE_TERMS if term.lower() in lowered_sanitized]
    boilerplate_hits = [term for term in BOILERPLATE_TERMS if term.lower() in lowered_sanitized]
    mismatch_hits = [term for term in UNIVERSITY_MISMATCH_TERMS if term.lower() in lowered_sanitized]
    fraud_hits = [term for term in FRAUD_CARTOON_TERMS if term.lower() in lowered_sanitized]
    placeholder_hit = bool(PLACEHOLDER_RE.search(sanitized_payload))
    cyrillic_hit = bool(CYRILLIC_RE.search(sanitized_payload))
    hidden_star_hit = any(term in lowered_sanitized for term in ["hidden star", "hidden genius", "secret potential"])

    validation_checks = {
        "all_ids_correct_and_in_order": ids == expected_ids,
        "all_ids_unique": len(ids) == len(set(ids)),
        "raw_sanitized_one_to_one": ids == sanitized_ids,
        "raw_fields_match_required_schema": all(set(record.keys()) == raw_allowed_fields for record in raw_records),
        "sanitized_fields_only_allowed": all(set(record.keys()) == sanitized_allowed_fields for record in sanitized_records),
        "sanitized_has_no_metadata": all("metadata" not in record for record in sanitized_records),
        "manifest_shape_valid": all(set(record.keys()) == manifest_allowed_fields for record in manifest_records),
        "slice_counts_target_met": slice_counts == SLICE_TARGETS,
        "text_length_target_met": text_length_counts == TEXT_LENGTH_TARGETS,
        "text_length_tag_consistency": len(text_length_mismatches) == 0,
        "no_interview_target_matched_exactly": no_interview_empty_count == 16,
        "no_interview_quality_mix_present": no_interview_quality_counts
        == {
            "strong_no_interview": 5,
            "borderline_no_interview": 6,
            "thin_no_interview": 5,
        },
        "overall_interview_balance_ok": interview_counts["with_interview"] > interview_counts["without_interview"]
        and (interview_counts["with_interview"] - interview_counts["without_interview"]) <= 12,
        "schema_valid": True,
        "hidden_metadata_leakage_absent": len(leakage_hits) == 0,
        "template_placeholders_absent": not placeholder_hit,
        "committee_boilerplate_absent": len(boilerplate_hits) == 0,
        "university_framing_intact": all(
            "invision u" in str(((record.get("text_inputs") or {}).get("motivation_letter_text") or "")).lower()
            for record in raw_records
        )
        and len(mismatch_hits) == 0,
        "translated_slice_english_only": not cyrillic_hit,
        "support_slice_has_no_hidden_star_leakage": not hidden_star_hit,
        "authenticity_slice_not_cartoon_fraud": len(fraud_hits) == 0,
        "no_near_duplicates": len(near_duplicate_pairs) == 0,
    }

    failed = [name for name, ok in validation_checks.items() if not ok]
    if failed:
        raise ValueError(
            "Validation failed for gap_fill_batch_v7: "
            + ", ".join(failed)
            + f" | text_length_mismatches={text_length_mismatches}"
            + f" | leakage_hits={leakage_hits}"
            + f" | boilerplate_hits={boilerplate_hits}"
            + f" | mismatch_hits={mismatch_hits}"
            + f" | near_duplicate_pairs={near_duplicate_pairs[:10]}"
        )

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    summary = {
        "batch_name": "gap_fill_batch_v7",
        "total_candidates": len(raw_records),
        "generated_at": now_iso,
        "distributions": {
            "slices": slice_counts,
            "ambiguity": ambiguity_counts,
            "text_lengths": text_length_counts,
            "interview_presence": interview_counts,
            "no_interview_quality_mix": no_interview_quality_counts,
        },
        "behavioral_signal_stats": {
            "completion_rate_min": round(min(completion_values), 3),
            "completion_rate_max": round(max(completion_values), 3),
            "completion_rate_avg": round(mean(completion_values), 3),
            "completion_rate_bands": completion_rate_bands,
            "returned_to_edit_count": returned_to_edit_count,
            "returned_to_edit_false_count": len(raw_records) - returned_to_edit_count,
            "skipped_optional_min": min(skipped_values),
            "skipped_optional_max": max(skipped_values),
            "skipped_optional_avg": round(mean(skipped_values), 3),
            "skipped_optional_histogram": skipped_histogram,
        },
        "validation_checks": validation_checks,
        "validation_fixes": [],
    }

    with RAW_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with GEN_MANIFEST_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with SUMMARY_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with PACK_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sanitized_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with PACK_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(sanitized_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with PACK_TABLE_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "english_proficiency_type",
                "english_proficiency_score",
                "school_cert_type",
                "school_cert_score",
                "motivation_letter_length",
                "num_questions",
                "has_interview",
                "completion_rate",
                "returned_to_edit",
                "skipped_optional",
            ],
        )
        writer.writeheader()
        writer.writerows(reviewer_table_rows(sanitized_records))

    pack_manifest = {
        "pack_name": "gap_fill_batch_v7",
        "batch_source": "gap_fill_batch_v7_api_input.jsonl",
        "total_candidates": len(sanitized_records),
        "created_at": now_iso,
        "schema_version": "1.0",
        "fields_included": [
            "candidate_id",
            "structured_data",
            "text_inputs",
            "behavioral_signals",
        ],
        "fields_excluded": ["metadata"],
        "notes": "Sanitized for human annotation. No hidden labels or generation metadata.",
    }

    with PACK_MANIFEST_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(pack_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))



SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_068",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="structure_support",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=77.0,
            school_type="City school diploma",
            school_score=84.0,
            city="Shymkent",
            opening="I am applying for psychology because I care about learning and motivation, but also because I think a structured university environment would help me become much more effective than I am in a looser setting.",
            school_context="At school I often worked seriously yet unevenly: when expectations were concrete, I did well; when tasks were open and long, I sometimes delayed the start too much.",
            study_interest="psychology",
            second_interest="education",
            example="I helped classmates make revision sheets and tried to keep small study routines going before exams, especially for students who felt overwhelmed by where to begin.",
            example_detail="That work showed me that structure is not only a personal preference for me; it is also something I naturally try to offer to other people when they are stuck.",
            local_issue="many students are treated as unmotivated when the more accurate problem is that they do not know how to start or how to sustain a plan",
            contribution="I would contribute practical organization, patience with people who are discouraged, and a willingness to use feedback instead of pretending I already know enough.",
            support_need="advisor check-ins, help breaking large assignments into steps, and early guidance on academic writing so open tasks do not become heavier than they need to be",
            uncertainty="I do not think of myself as a candidate who only needs a chance to shine; I think I am a candidate who can become much steadier with the right scaffolding.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="What I need most is not lower standards but clearer structure around meeting those standards.",
            completion_rate=0.86,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_069",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="confidence_support",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=72.0,
            school_type="Standard school diploma",
            school_score=82.0,
            city="Semey",
            opening="Urban studies and local systems interest me, but I should be honest that I still hesitate more than I should when I have to explain ideas in front of unfamiliar people.",
            school_context="My school contribution has usually been practical and behind the scenes rather than highly visible.",
            study_interest="urban studies",
            second_interest="public policy",
            example="I helped collect small observations about transport and access around our area and I was usually dependable when notes had to be organized after a group task.",
            example_detail="This gave me a real interest in how local systems affect learning, though I still understate or delay my own contribution in live settings.",
            local_issue="students adapt to inconvenient routes and study conditions so fully that adults stop seeing those conditions as problems",
            contribution="I would contribute careful observation, note organization, and steady work that keeps a team moving quietly.",
            support_need="confidence support in discussion-heavy classes and encouragement to speak earlier rather than only when I feel fully ready",
            uncertainty="I am not aiming to sound more accomplished than I am; I am trying to describe the kind of support that would make my effort more visible and useful.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="One of my real risks is becoming too quiet in a new environment if I do not push myself deliberately.",
            completion_rate=0.8,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_070",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="presentation_support",
            intended_ambiguity="hard",
            text_length="long",
            english_type="private tutoring",
            english_score=78.0,
            school_type="Lyceum certificate",
            school_score=88.0,
            city="Pavlodar",
            opening="Mathematics and data work make sense to me when I can sit with them calmly, but I am less strong at presenting my reasoning quickly in front of others.",
            school_context="My grades are stable and I can work hard for long periods, yet I know that university will ask for more visible communication than my school context did.",
            study_interest="mathematics",
            second_interest="data analysis",
            example="I helped classmates understand step-by-step solutions before tests and I was often more useful in one-to-one explanation than in a fast group presentation.",
            example_detail="That pattern does not mean I cannot contribute; it means the support I need is around presentation, confidence, and learning how to speak before I feel completely finished.",
            local_issue="students who think carefully can still look weaker than they are when classrooms reward speed of response more than depth of understanding",
            contribution="I would contribute persistence, careful preparation, and willingness to help others understand a task without making them feel small for asking.",
            support_need="presentation practice, seminar participation support, and guidance on how to speak more directly instead of over-preparing silently",
            uncertainty="I am promising in a modest way, not because there is some hidden dramatic upside, but because steady academic effort could become more complete with the right support around communication.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="My challenge is not doing the work; it is becoming visible enough in the room while doing it.",
            completion_rate=0.88,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_071",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="language_support",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=71.0,
            school_type="City school diploma",
            school_score=83.0,
            city="Aktobe",
            opening="Environmental topics matter to me in a very practical way, but I know that my transition into English-medium university study will need deliberate support and not only goodwill.",
            school_context="In school I was dependable and serious, and I gradually became more interested in how ordinary local environmental problems shape daily routines.",
            study_interest="environmental science",
            second_interest="community planning",
            example="I joined cleanup work, helped record what we saw, and kept thinking afterward about why such small recurring problems stay normal for so long.",
            example_detail="The action itself was modest, yet it gave me a real anchor for study; what remains difficult is explaining that anchor in polished academic English without flattening it.",
            local_issue="students notice drainage, waste, and dust problems every day, but they often do not yet have the language to connect those observations with bigger systems",
            contribution="I would contribute seriousness, careful fieldwork, and willingness to keep doing the routine parts of a project after the first enthusiasm drops.",
            support_need="language support in the first year, especially for writing and seminar discussion, so that slower English processing does not become mistaken for low engagement",
            uncertainty="I do not need to be rescued, but I do need a realistic academic bridge into a setting where language speed matters much more than it does now.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="If I receive structured language support early, I think I can adapt steadily instead of always working from behind.",
            completion_rate=0.82,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_072",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="adaptation_support",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=74.0,
            school_type="Regional school certificate",
            school_score=84.0,
            city="Taldykorgan",
            opening="I want to study business and community development in a way that is more disciplined than my current environment allows, but I know discipline alone is not the whole transition story for me.",
            school_context="My school profile is modestly positive: steady grades, some practical reliability, and a tendency to improve when expectations are clear and someone checks whether I actually understood them.",
            study_interest="business",
            second_interest="community development",
            example="I helped organize small school events and donation tasks, usually by doing the planning details that other people forgot until the last moment.",
            example_detail="These are not remarkable achievements, yet they show the kind of contribution I can already make when the environment is organized enough for me to stay effective.",
            local_issue="many students from ordinary schools are willing to work hard, but they lose momentum during transition because nobody explains what good university-level self-management really looks like",
            contribution="I would contribute reliability, practical planning, and a respectful team style that does not disappear once the exciting part of a task is over.",
            support_need="onboarding support, regular check-ins, and clear expectations in the first year so that I can adapt to faster pace and more independent planning without quietly falling behind",
            uncertainty="My case is not about secret excellence that only needs to be discovered; it is about a student with some real promise who is more likely to succeed if support and accountability are both present from the start.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="I think I can become a solid university student, but probably not by pretending I will need no support at all.",
            completion_rate=0.85,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_063",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="writing_support",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school classes",
            english_score=70.0,
            school_type="City school diploma",
            school_score=83.0,
            city="Almaty",
            opening="I am applying to inVision U because I think I am ready for a more serious learning environment, but also because I know I work better when support is built into that environment.",
            school_context="My school record is decent and my contribution has mostly been small and repeated rather than highly visible.",
            study_interest="education",
            second_interest="social science",
            example="I often helped classmates make shorter summaries and study plans before tests when they were falling behind.",
            example_detail="This is not a big leadership story, but it is the kind of steady work that feels true to me.",
            local_issue="students who need structure most often feel embarrassed to ask for it early",
            contribution="I would contribute reliability, practical help, and a respectful tone in group work.",
            support_need="academic writing support and regular feedback so I do not spend too much time guessing what a strong answer should look like",
            uncertainty="I am motivated, but I do not think motivation alone will solve the transition.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="I prefer to be direct about needing support because hiding it would not help anyone later.",
            completion_rate=0.81,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_064",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="adaptation_support",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=73.0,
            school_type="Regional school certificate",
            school_score=84.0,
            city="Baikonur",
            opening="My reason for applying is connected not only to what I want to study, but also to the kind of structure I think I need in order to study well after school.",
            school_context="I come from a smaller and more predictable learning context, where I could keep stable grades without always being pushed to speak up or work in open-ended ways.",
            study_interest="economics",
            second_interest="public systems",
            example="I helped classmates with shared notes and exam planning, and I was usually dependable when a task had to be finished calmly rather than quickly.",
            example_detail="These are modest signals, but they show the kind of student I currently am: serious, useful in practical ways, and not yet fully confident in new academic settings.",
            local_issue="students from less resourced contexts can be capable yet unsure how to adapt when expectations become less explicit and more discussion-based",
            contribution="I would contribute persistence, dependable teamwork, and willingness to use support early instead of waiting until problems grow.",
            support_need="transition support in the first semester, especially around seminar participation, academic writing, and understanding how to manage less structured assignments",
            uncertainty="I do not think of myself as hidden talent waiting to be discovered; I think of myself as a student who can improve meaningfully if the environment is demanding and supportive at the same time.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="I can handle hard work, but I probably will need help learning the style of university work faster than some students do.",
            completion_rate=0.84,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_065",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="time_management_support",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + apps",
            english_score=75.0,
            school_type="City school diploma",
            school_score=85.0,
            city="Karagandy",
            opening="I am interested in computer science and analytics, but the most honest way to describe my current profile is that I need stronger structure around how I learn.",
            school_context="At school I can do well when tasks are clear, and I often help others once I understand the practical steps, but I still misjudge time badly on open work.",
            study_interest="computer science",
            second_interest="data analysis",
            example="I helped younger students understand how to submit assignments online and I usually became the person who solved simple practical confusion in class.",
            example_detail="That gives me some confidence that I can be useful, though it does not cancel the fact that independent project planning is still a weak area for me.",
            local_issue="students who are not naturally organized can look less capable than they really are once tasks become open-ended and deadlines multiply",
            contribution="I would contribute practical problem-solving, patience with technical confusion, and willingness to share methods that reduce unnecessary stress.",
            support_need="time-management coaching, milestone-based feedback, and accountability early in the semester so I do not let avoidable confusion build up quietly",
            uncertainty="I do not need rescue or special treatment, but I do need a realistic transition into a more independent academic environment.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="If I join, one important thing will be learning to ask for guidance before I am already behind.",
            completion_rate=0.83,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_066",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="language_support",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="language center",
            english_score=68.0,
            school_type="Standard school diploma",
            school_score=81.0,
            city="Taraz",
            opening="I want to study biology and health-related subjects at inVision U, and I also know I will need language support to show what I understand at the level I want.",
            school_context="My school work is decent, and I have usually been reliable rather than outstanding.",
            study_interest="biology",
            second_interest="public health",
            example="I helped relatives and classmates read through health information more slowly when the first explanation felt too fast or too technical.",
            example_detail="This is a modest example, but it is close to why I still care about this field.",
            local_issue="people can misunderstand useful health information simply because the language feels too far from daily speech",
            contribution="I would contribute seriousness, patience, and careful listening in groups.",
            support_need="English support for academic writing and speaking so that slower language does not hide the level of care I can bring",
            uncertainty="I do not think the problem is ability only; it is also confidence and adaptation.",
            qa_profile="support_direct",
            interview_mode="support_candid",
            interview_hook="My English is workable, but I would grow faster if I had structured support instead of only trying to manage it alone.",
            completion_rate=0.79,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_067",
            intended_gap_slice="support_needed_but_not_hidden_star_cases",
            slice_variant="confidence_support",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school classes",
            english_score=69.0,
            school_type="Regional school certificate",
            school_score=80.0,
            city="Kokshetau",
            opening="I want to study psychology at inVision U because I care about how students gain confidence, and I know my own confidence is still something I am building.",
            school_context="My profile is modest but steady.",
            study_interest="psychology",
            second_interest="education",
            example="I usually help quietly and do not speak first.",
            example_detail="That is both a strength and a limit for me now.",
            local_issue="students can be capable and still stay invisible because they hesitate to step forward",
            contribution="I would contribute patience and consistency.",
            support_need="confidence-building in seminars and regular feedback",
            uncertainty="I think I can grow well with structure.",
            qa_profile="support_direct",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.67,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_057",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="reflective_literal",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=79.0,
            school_type="City school diploma",
            school_score=85.0,
            city="Taraz",
            opening="Law for me is interesting not because it sounds prestigious, but because rules are where society becomes visible in ordinary life, and I started feeling this quite personally.",
            school_context="At school many discussions about fairness stayed emotional, but I wanted to understand what stands behind such arguments in more structured form.",
            study_interest="law",
            second_interest="public policy",
            example="I paid attention when school rules were interpreted differently by different teachers and noticed how quickly students lose trust when the same situation receives two explanations.",
            example_detail="This did not become a big school initiative, but it changed how I think: before I reacted only emotionally, later I wanted to ask what system produces that confusion and who has power to clarify it.",
            local_issue="young people often speak about justice in very big words while lacking simple legal understanding that could help in daily situations",
            contribution="I would contribute careful reading, calm discussion, and readiness to hear a different point fully before I answer it.",
            support_need="more experience connecting public concerns with actual structured work and more fluency in formal academic English",
            uncertainty="I still think in a more literal way than I write, so some phrasing may sound unusual, but the logic is sincere.",
            qa_profile="translated_reflective",
            interview_mode="translated_oral",
            interview_hook="When I speak, I am slower, but the connection between fairness and daily life becomes easier for me to show.",
            completion_rate=0.84,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_058",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="literal_but_clear",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school classes",
            english_score=67.0,
            school_type="Standard school diploma",
            school_score=78.0,
            city="Turkistan",
            opening="Education matters to me because one supportive explanation can change the whole mood of study for a person.",
            school_context="My examples are simple and school-level.",
            study_interest="education",
            second_interest="psychology",
            example="I helped classmates with notes and simple planning before tests.",
            example_detail="It is small, but meaningful for me.",
            local_issue="many students understand after the second explanation, not after the first one",
            contribution="I would contribute patience and regular help.",
            support_need="stronger writing and more confidence in English discussion",
            uncertainty="I am still learning how to explain myself fully.",
            qa_profile="translated_thin",
            interview_mode="translated_oral",
            interview_hook="My speaking is simple, but it usually carries my intention better.",
            completion_rate=0.69,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_059",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="local_context_literal",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=72.0,
            school_type="Regional school certificate",
            school_score=80.0,
            city="Aktobe",
            opening="Journalism came to me not from some romantic dream, but from irritation that important local things are spoken in a noisy way and not an exact way.",
            school_context="In school I paid attention to how announcements, rumors, and half-information change the mood faster than facts do.",
            study_interest="journalism",
            second_interest="social research",
            example="I rewrote information for classmates more than once because the original version was technically correct but still hard to understand.",
            example_detail="This is a very local example, though it is the most honest line from where my interest started.",
            local_issue="people receive information every day, but clarity and trust are often missing at the same time",
            contribution="I would contribute careful wording and attention to whether communication really lands.",
            support_need="better method, stronger reporting practice, and more confidence to ask questions in public",
            uncertainty="The written part carries a slightly literal style, but I hope it still sounds clear and human.",
            qa_profile="translated_clear",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.76,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_060",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="code_switch_logic",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=80.0,
            school_type="Lyceum certificate",
            school_score=88.0,
            city="Pavlodar",
            opening="Engineering became important for me not because I wanted a status profession, but because I like when a difficult thing becomes sobrano, checked, and working in real conditions.",
            school_context="At school I liked physics and practical problem solving, especially when a task had to survive outside perfect classroom assumptions.",
            study_interest="engineering",
            second_interest="energy systems",
            example="I helped assemble and fix simple school equipment more than once, and I liked the process where disorder slowly becomes understandable through sequence.",
            example_detail="It was not advanced engineering, still it gave me a feeling that technical work is also about character: patience, order, and not becoming angry when the first attempt does not work.",
            local_issue="energy and infrastructure questions are often discussed only at the big level, while ordinary reliability problems shape daily trust much more than speeches do",
            contribution="I would contribute persistence, calm troubleshooting, and respect for work that is important even when it looks repetitive.",
            support_need="stronger formal foundation and more experience presenting technical reasoning in English without losing precision",
            uncertainty="My sentence rhythm can be local and literal, but the technical motivation is very stable for me.",
            qa_profile="translated_reflective",
            interview_mode="translated_oral",
            interview_hook="When I speak, I am more direct and less polished, but the technical logic usually becomes cleaner.",
            completion_rate=0.87,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_061",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="reflective_literal",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=74.0,
            school_type="City school diploma",
            school_score=82.0,
            city="Semey",
            opening="Sociology is close to me because it gives language to things people feel every day but explain only roughly.",
            school_context="I noticed this especially in conversations about study pressure, migration, and how families imagine a good future.",
            study_interest="sociology",
            second_interest="youth studies",
            example="I listened to classmates and cousins speaking about choices after school and began writing short notes for myself about repeated fears and repeated hopes.",
            example_detail="It was private observation, not formal research, but it moved me from general sympathy to a wish for more structured understanding.",
            local_issue="many young people speak about future plans with confidence outside and uncertainty inside at the same time",
            contribution="I would contribute careful listening, serious discussion, and respect for different life situations inside one classroom.",
            support_need="research basics and stronger confidence to ask follow-up questions in English",
            uncertainty="My English may sound translated in thinking, but I hope it still reads as human and clear.",
            qa_profile="translated_reflective",
            interview_mode="translated_oral",
            interview_hook="Speaking helps me make the emotional logic more visible even if the grammar stays simple.",
            completion_rate=0.78,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_062",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="local_context_literal",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school classes",
            english_score=71.0,
            school_type="Regional school certificate",
            school_score=84.0,
            city="Petropavl",
            opening="Agriculture and data came together for me in a very ordinary way, through family talk, school tasks, and the feeling that local decisions are often made from habit and not from clear information.",
            school_context="I am from a context where practical work is respected, but analytical language around that work is not always easy for students to access early.",
            study_interest="agriculture",
            second_interest="data analysis",
            example="I kept simple records for myself about seasonal price and yield talk inside the family and then started comparing those impressions with what we were learning in class.",
            example_detail="This was not an official project, but it showed me that data becomes meaningful exactly when it returns back to ordinary local decisions and not only to school exercises.",
            local_issue="many local economic and agricultural choices are discussed through habit, memory, and trust, while structured evidence enters the conversation too late",
            contribution="I would contribute patience, respect for practical knowledge, and willingness to do detailed background work that other students may skip.",
            support_need="better analytical tools, stronger writing, and more confidence bridging local experience with university-level language",
            uncertainty="The reader sees mainly my written style here, which is simpler than the full reasoning in my head.",
            qa_profile="translated_reflective",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.81,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_051",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="literal_but_clear",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=74.0,
            school_type="City school diploma",
            school_score=83.0,
            city="Almaty Region",
            opening="This decision did not come to me fast, but after some time it became near to me that I want to study education and technology together.",
            school_context="In my school many things were explained formally, but in practical way students still got confused, especially with online homework systems.",
            study_interest="education technology",
            second_interest="computer science",
            example="I often sat with younger students and showed them not only where to click, but why one step follows another step.",
            example_detail="It is not a big project if I say honestly, still for me it was the moment when technology stopped being only a subject and became a tool for other people.",
            local_issue="students lose courage quickly when digital tools look simple to adults but not simple from the student side",
            contribution="I would bring patient explanation, calm work in a group, and readiness to repeat a thing without making another person feel stupid.",
            support_need="stronger technical base and more confidence to speak in English seminar style",
            uncertainty="I still do not know if my future will stand more on coding side or on learning side, but both feel connected for me.",
            qa_profile="translated_clear",
            interview_mode="translated_oral",
            interview_hook="When I speak, my sentences are not elegant, but usually my meaning becomes more open.",
            completion_rate=0.8,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_052",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="local_context_literal",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="language center",
            english_score=76.0,
            school_type="Standard school diploma",
            school_score=82.0,
            city="Kokshetau",
            opening="I am applying to inVision U because economics for me is not only about numbers, it is about how people in one city feel possibilities or no possibilities.",
            school_context="In our area many family talks are about prices, transport, and what is becoming harder every month, so I started to look at economics not as abstract topic.",
            study_interest="economics",
            second_interest="social statistics",
            example="I kept small notes about price changes in ordinary shops and compared them with what adults in my family were saying.",
            example_detail="This was a home-level observation, not research, but it made my interest more alive and less school-only.",
            local_issue="students hear a lot of adult worry about money, but rarely learn a calm method for understanding what is really changing",
            contribution="I would contribute disciplined note-taking, steady discussion, and a serious attitude in shared assignments.",
            support_need="better analytical method and more courage to trust my own explanation in English",
            uncertainty="I know my writing can sound slightly translated in thinking, but the interest itself is direct.",
            qa_profile="translated_clear",
            interview_mode="translated_oral",
            interview_hook="In interview I usually speak more simply and less academically, but maybe also more honestly.",
            completion_rate=0.82,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_053",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="literal_but_clear",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school classes",
            english_score=68.0,
            school_type="Regional school certificate",
            school_score=79.0,
            city="Shymkent",
            opening="Biology became close to me not in one day, but slowly, when I started to notice how health questions in family life are explained unclearly.",
            school_context="My examples are still simple and from everyday life.",
            study_interest="biology",
            second_interest="public health",
            example="I mostly listened, read, and tried to understand practical health information better.",
            example_detail="This is not much, but it is true.",
            local_issue="people often postpone small health decisions because information reaches them in a difficult form",
            contribution="I would contribute seriousness and patience.",
            support_need="stronger academic base and more concrete experience",
            uncertainty="My English is simple, but I hope still understandable.",
            qa_profile="translated_thin",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.64,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_054",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="code_switch_logic",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=78.0,
            school_type="City school diploma",
            school_score=84.0,
            city="Kyzylorda",
            opening="When I think about urban planning, I think not first in beautiful theory, but in very zemnoy way, about how daily routes make a person tired or calm before study even starts.",
            school_context="In my city, school and home routes teach many things if a person looks carefully, especially about heat, transport, waiting, and how public space is actually used.",
            study_interest="urban planning",
            second_interest="environmental design",
            example="I began writing down what I noticed on ordinary routes: where students stand without shade, where they cross in uncomfortable places, where people stop because the path itself asks them to stop.",
            example_detail="It was not official research and I do not want to pretend it was, but from that moment planning became for me not decorative interest, rather question of daily human behavior.",
            local_issue="small city design decisions create invisible pressure on students, especially when transport and climate make simple movement heavier than it should be",
            contribution="I would contribute observation, careful written notes, and respect for local details that look too small until someone names them clearly.",
            support_need="formal methods, stronger design vocabulary, and confidence to move from watching to testing ideas with others",
            uncertainty="My phrasing can sound literal because I think through another language first, but the link between place and learning is deeply real for me.",
            qa_profile="translated_reflective",
            interview_mode="translated_oral",
            interview_hook="Speaking is easier when I can show the logic step by step and not only one perfect sentence.",
            completion_rate=0.86,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_055",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="literal_but_clear",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=73.0,
            school_type="City school diploma",
            school_score=81.0,
            city="Atyrau",
            opening="Psychology became important for me because I saw that many students look calm outside, but inside they are already tired and disconnected.",
            school_context="At school I was often the person who listened longer than others, even when I did not have a big answer.",
            study_interest="psychology",
            second_interest="peer support",
            example="I helped friends before exams by making simple study plans and just sitting with them so they did not drop the task in the middle.",
            example_detail="It is a small thing, but in our context small support is sometimes the reason a person continues or stops.",
            local_issue="students often understand material only after someone explains it in human language and not only in teacher language",
            contribution="I would contribute patience, quiet support, and seriousness in cooperative work.",
            support_need="stronger academic language and more confidence in seminar speaking",
            uncertainty="I still search whether my path should be psychology only or also something about education.",
            qa_profile="translated_clear",
            interview_mode="translated_oral",
            interview_hook="My speech is not polished, but I can usually explain the human side of the issue more directly.",
            completion_rate=0.77,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_056",
            intended_gap_slice="translated_or_mixed_thinking_english_cases",
            slice_variant="local_context_literal",
            intended_ambiguity="hard",
            text_length="medium",
            english_type="school + self-study",
            english_score=75.0,
            school_type="Lyceum certificate",
            school_score=86.0,
            city="Karagandy",
            opening="Computer science attracts me because I like when a problem becomes not only discussed, but razobrano step by step until it is workable.",
            school_context="In school I often solved technical confusion for classmates, though mostly on ordinary level and not in advanced programming form.",
            study_interest="computer science",
            second_interest="information systems",
            example="I showed younger students how to submit tasks, rename files, and not get lost in simple digital instructions.",
            example_detail="Maybe this sounds too basic, but for me it showed that technology is useful exactly where people think the problem is too small to mention.",
            local_issue="students lose courage when digital processes are treated as obvious while in reality many first-generation users are only pretending they understand",
            contribution="I would contribute calm practical help and careful work with technical details.",
            support_need="deeper coding skill and more experience owning one project from start to finish",
            uncertainty="The motivation is clear for me, even if my English carries local rhythm inside it.",
            qa_profile="translated_clear",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.79,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_043",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="borderline_no_interview",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=81.0,
            school_type="Lyceum certificate",
            school_score=86.0,
            city="Kostanay",
            opening="I want the application to be read for what it is: a plausible but not fully convincing written case.",
            school_context="My strongest subjects are history and literature, and I gradually became interested in law and public systems through school reading and local discussion.",
            study_interest="law",
            second_interest="public policy",
            example="I helped prepare discussion questions for one school debate and became the person classmates sometimes asked when school rules or public topics were confusing.",
            example_detail="Those things show a direction, but they still do not amount to a strong independent record yet.",
            local_issue="young people talk about fairness and rules often, but usually in reactive ways rather than as a subject that can be studied carefully",
            contribution="I would contribute serious reading, thoughtful discussion, and reliable preparation for shared tasks.",
            support_need="more real evidence and more confidence moving from discussion into action",
            uncertainty="This is not a weak application, but it is one that still depends heavily on interpretation because the evidence base is limited.",
            qa_profile="borderline",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.82,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_044",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="borderline_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school classes",
            english_score=72.0,
            school_type="Standard school diploma",
            school_score=80.0,
            city="Petropavl",
            opening="My application is probably a straightforward middle case rather than a dramatic one.",
            school_context="I am interested in media and communication, but most of my evidence comes from small school situations.",
            study_interest="media studies",
            second_interest="communications",
            example="I often rewrote class group messages so information would be clearer and fewer people missed deadlines.",
            example_detail="This is useful in ordinary life, although it does not prove much beyond a basic communication habit.",
            local_issue="students miss opportunities not only because of motivation, but because information reaches them in confusing forms",
            contribution="I would contribute practical communication support and steady participation.",
            support_need="larger examples and more confidence explaining why this field matters to me",
            uncertainty="I think the application is valid, only not very deep.",
            qa_profile="borderline",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.74,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_045",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="borderline_no_interview",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school + apps",
            english_score=71.0,
            school_type="Regional school certificate",
            school_score=78.0,
            city="Turkistan",
            opening="I want to study business at inVision U because I think I learn better when expectations are higher and the environment is more serious.",
            school_context="My record is ordinary and my examples are small.",
            study_interest="business",
            second_interest="management",
            example="Mostly I have been reliable in class tasks and open to feedback.",
            example_detail="That is enough for a valid application but not enough for a strong one.",
            local_issue="many students wait for motivation to appear instead of building habits around clear goals",
            contribution="I would contribute reliability and cooperative work.",
            support_need="more initiative and clearer examples",
            uncertainty="This is a borderline written-only case.",
            qa_profile="borderline",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.65,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_046",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="thin_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school classes",
            english_score=69.0,
            school_type="City school diploma",
            school_score=77.0,
            city="Atyrau",
            opening="The thinness of my written material is easier to notice than I would like.",
            school_context="I am interested in biology, but my profile outside school lessons is limited.",
            study_interest="biology",
            second_interest="public health",
            example="I mostly did regular coursework and sometimes shared notes with classmates.",
            example_detail="This is a sincere application, only not one with much evidence behind it.",
            local_issue="students say they care about health topics, but very few of us know how to turn that into practical learning early on",
            contribution="I would contribute steady effort.",
            support_need="more concrete exposure and stronger answers about what I have already done",
            uncertainty="The case may be too thin to judge confidently.",
            qa_profile="thin_basic",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.63,
            returned_to_edit=False,
            skipped_optional_questions=4,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_047",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="thin_no_interview",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=74.0,
            school_type="Standard school diploma",
            school_score=81.0,
            city="Aktau",
            opening="A long written application can still be thin, and I think mine risks exactly that problem because reflection is ahead of evidence.",
            school_context="I care about social questions and community life, but most of my school years were quiet and centered on regular academic duties.",
            study_interest="social science",
            second_interest="community development",
            example="I have helped with notes, spoken with classmates about their worries, and thought seriously about what kind of university environment helps students grow.",
            example_detail="All of this is honest, yet it still leaves the reader with mostly intentions and atmosphere rather than concrete initiative.",
            local_issue="many students want meaningful education but have little practice turning concern into consistent action",
            contribution="I would contribute seriousness, respect, and willingness to participate responsibly.",
            support_need="more specific examples, stronger ownership, and less dependence on broad values language",
            uncertainty="If a reviewer called this sincere but under-evidenced, I would understand the judgment.",
            qa_profile="long_vague",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.76,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_048",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="thin_no_interview",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school + apps",
            english_score=68.0,
            school_type="City school diploma",
            school_score=76.0,
            city="Kyzylorda",
            opening="I am applying for information systems because I want a stronger future path, but my current application is still very simple.",
            school_context="I mostly completed regular tasks and have not done much outside them.",
            study_interest="information systems",
            second_interest="computer science",
            example="My interest is real, but the examples are weak.",
            example_detail="That is the honest summary.",
            local_issue="many students want digital skills but stay at the level of intention only",
            contribution="I would contribute effort and willingness to learn.",
            support_need="clearer habits, stronger examples, and more initiative",
            uncertainty="This is usable, but very thin.",
            qa_profile="thin_basic",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.58,
            returned_to_edit=False,
            skipped_optional_questions=4,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_049",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="thin_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school classes",
            english_score=70.0,
            school_type="Regional school certificate",
            school_score=79.0,
            city="Shymkent",
            opening="I think this application mainly shows that I am interested, not that I have already done much.",
            school_context="I want to study education and peer learning, but my own activity history is modest.",
            study_interest="education",
            second_interest="psychology",
            example="I helped friends before tests and tried to explain material in simple words.",
            example_detail="That matters to me, though I know it stays at the level of a very ordinary example.",
            local_issue="students often need peer explanation more than another formal instruction",
            contribution="I would contribute patience and regular attendance.",
            support_need="more depth, more initiative, and more confidence in describing my motivation",
            uncertainty="The application is readable but thin in substance.",
            qa_profile="thin_basic",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.66,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_050",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="thin_no_interview",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=75.0,
            school_type="Lyceum certificate",
            school_score=84.0,
            city="Uralsk",
            opening="I can write at length about why university matters to me, but I know that length does not solve the deeper issue that my case is still short on evidence.",
            school_context="My grades are acceptable and I care about economics and social systems, yet most of my profile stayed close to assigned work rather than self-started action.",
            study_interest="economics",
            second_interest="public systems",
            example="I read, discussed ideas, and paid attention to how policy questions touch ordinary life in my city.",
            example_detail="What is missing is a clearer track record of doing something with that concern beyond reflection and class assignments.",
            local_issue="young people often form opinions on public issues before they have experience studying them carefully",
            contribution="I would contribute thoughtful participation and dependable work in structured settings.",
            support_need="more initiative, stronger independent habits, and more evidence than general concern",
            uncertainty="This is a genuine application, but not one that gives a reader much firm ground.",
            qa_profile="long_vague",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.73,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_035",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="strong_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=82.0,
            school_type="Lyceum certificate",
            school_score=89.0,
            city="Almaty",
            opening="I am applying because the written sections can still carry the full picture of why I am serious about this path.",
            school_context="At school I became interested in education and data through repeated peer-support work rather than one formal title.",
            study_interest="education policy",
            second_interest="data analysis",
            example="For two exam periods I helped keep a shared revision schedule and notes bank alive for classmates who usually lost track of deadlines.",
            example_detail="It was not dramatic leadership, but it was consistent work and it changed how I think about systems that quietly support learning.",
            local_issue="students who are capable still fall behind when nobody makes expectations and resources visible in a simple way",
            contribution="I would contribute careful coordination, follow-through, and attention to people who stop participating silently.",
            support_need="more advanced analytical tools so I can test these observations more seriously",
            uncertainty="My direction is real even if it is still broad between policy and data.",
            qa_profile="grounded",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.86,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_036",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="strong_no_interview",
            intended_ambiguity="hard",
            text_length="long",
            english_type="private tutoring",
            english_score=88.0,
            school_type="City school diploma",
            school_score=90.0,
            city="Astana",
            opening="I tried to make the written application plain and specific instead of polished for its own sake.",
            school_context="My strongest school subjects are mathematics and informatics, but the part that matters most for this application is how those interests became useful in small peer settings.",
            study_interest="computer science",
            second_interest="learning technology",
            example="I helped a teacher clean up a messy digital homework process by rewriting instructions, checking broken links, and showing younger students how to submit work without panic.",
            example_detail="That effort stayed local and practical, yet it taught me something important about technology: good systems are often less about complexity and more about whether people can actually use them.",
            local_issue="students lose time and confidence because school digital tools are introduced as if everyone already knows how to navigate them",
            contribution="I would contribute practical troubleshooting, patience with people who are embarrassed to ask basic questions, and steady work on the unglamorous side of group tasks.",
            support_need="broader technical foundations and more experience building something that is truly my own",
            uncertainty="I know the case may look modest compared with stronger technical applicants, but the signal here is real and usable.",
            qa_profile="grounded",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.9,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_037",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="strong_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=79.0,
            school_type="Regional school certificate",
            school_score=85.0,
            city="Kokshetau",
            opening="I want the written parts to show an ordinary but credible version of who I am.",
            school_context="I became interested in public health through small family and school experiences rather than a formal club or competition.",
            study_interest="public health",
            second_interest="health communication",
            example="I helped relatives handle clinic forms and later noticed similar confusion among families in our area about where to ask basic preventive questions.",
            example_detail="That is not a large project, but it is a grounded reason for why I now care about communication inside health systems.",
            local_issue="families often delay simple health decisions because the process feels harder than it should",
            contribution="I would contribute seriousness, respectful communication, and willingness to do practical support work in teams.",
            support_need="research training and stronger academic writing so the concern can become a discipline and not only an observation",
            uncertainty="I am still deciding how much the direction is science and how much it is communication, but the issue itself feels stable.",
            qa_profile="grounded",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.83,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_038",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="strong_no_interview",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=84.0,
            school_type="Lyceum certificate",
            school_score=87.0,
            city="Karagandy",
            opening="The risk in a tidy document is that a reader may miss the slower process that produced it.",
            school_context="My real strength is not that I already know exactly what I will become, but that I stay with a question long enough to test it in small ways.",
            study_interest="urban studies",
            second_interest="public policy",
            example="Over the last year I tracked how students in our area get to school, where they wait, and which small route problems adults call normal because they happen every day.",
            example_detail="The work was simple observation and informal note-taking, not a formal study, but it taught me to connect ordinary movement with access, safety, and stress.",
            local_issue="small transport and space problems shape whether students arrive tired, late, or ready to participate, yet they are rarely treated as educational issues",
            contribution="I would contribute patient observation, careful preparation, and a style of group work that takes local evidence seriously before jumping to slogans.",
            support_need="more formal methods and feedback so I can test local observations instead of relying only on intuition",
            uncertainty="The path still sits between policy and planning, but the core interest has become stronger, not weaker, over time.",
            qa_profile="grounded",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.88,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_039",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="strong_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=78.0,
            school_type="City school diploma",
            school_score=84.0,
            city="Pavlodar",
            opening="I am comfortable being judged from the written application only because the signal I want to show is mostly about consistency, not performance in one conversation.",
            school_context="At school I usually became useful in group tasks when someone had to keep practical details from falling apart.",
            study_interest="psychology",
            second_interest="education",
            example="I made short revision summaries for classmates during two difficult terms and checked on students who had attendance gaps so they would not disappear from the class rhythm.",
            example_detail="This was small-scale work, but it made me care more about motivation, confidence, and the quiet side of academic adaptation.",
            local_issue="students who need support most often ask for it last because they do not want to look weak in front of peers",
            contribution="I would contribute reliability, quiet encouragement, and the habit of following up after the first discussion ends.",
            support_need="stronger theoretical background and more confidence in seminar discussion",
            uncertainty="The profile is not flashy, but it is more solid than it first appears.",
            qa_profile="grounded",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.84,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_040",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="borderline_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school classes",
            english_score=74.0,
            school_type="Standard school diploma",
            school_score=81.0,
            city="Taraz",
            opening="What matters here is whether the written material is enough to show a believable direction.",
            school_context="My school results are acceptable, and my interests became clearer only during the last year.",
            study_interest="economics",
            second_interest="business",
            example="I helped classmates prepare for one economics exam and later started reading simple articles outside class.",
            example_detail="This is a real start, but it is not yet a strong portfolio of initiative.",
            local_issue="students often learn formulas about money and markets without understanding how those ideas affect ordinary family choices",
            contribution="I would contribute discipline and practical teamwork.",
            support_need="more independent initiative and a better sense of direction",
            uncertainty="The case is workable but still sits near the middle.",
            qa_profile="borderline",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.77,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_041",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="borderline_no_interview",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=76.0,
            school_type="Regional school certificate",
            school_score=82.0,
            city="Aktobe",
            opening="I think this application should be read as a borderline case with some real signal and some obvious gaps.",
            school_context="I care about environmental systems, but my school context offered few structured ways to explore that beyond ordinary classes and occasional volunteer events.",
            study_interest="environmental science",
            second_interest="community planning",
            example="I joined cleanup work near a local river area, collected simple notes about what people kept ignoring, and tried to keep the issue in my own mind after the event ended.",
            example_detail="The problem is that I do not yet have a stronger second step to show after that first useful experience.",
            local_issue="people get used to waste and drainage problems because each one looks small by itself, even when the total effect is obvious",
            contribution="I would contribute reliable fieldwork, willingness to do practical tasks, and respect for team roles that are not very visible.",
            support_need="more structure, more method, and more courage to turn observation into actual small projects",
            uncertainty="The motivation feels more mature than the track record, which is why I think this remains a borderline written-only case.",
            qa_profile="borderline",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.8,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_042",
            intended_gap_slice="no_interview_cases_across_quality_levels",
            slice_variant="borderline_no_interview",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=75.0,
            school_type="City school diploma",
            school_score=83.0,
            city="Semey",
            opening="I know a no-interview application needs enough detail to stand on its own, and mine probably lands somewhere in the middle.",
            school_context="My interests in sociology and education grew from watching classmates make very different choices with similar abilities.",
            study_interest="sociology",
            second_interest="education",
            example="I often listened to classmates about study pressure and wrote down patterns for myself, though I did not turn that habit into a formal project.",
            example_detail="That means the core interest is real, but the visible evidence remains narrow.",
            local_issue="many students interpret their own uncertainty as personal failure when some of it comes from weak guidance and unclear expectations",
            contribution="I would contribute listening, respectful discussion, and seriousness in collaborative work.",
            support_need="stronger methods and more initiative outside observation",
            uncertainty="There is enough here for standard reading, but not enough to remove doubt.",
            qa_profile="borderline",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.79,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_027",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="repetitive_generic",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="language center",
            english_score=74.0,
            school_type="Standard school diploma",
            school_score=82.0,
            city="Pavlodar",
            opening="I want a university where I can learn biology seriously, grow as a person, and become more useful for society.",
            school_context="My school record is mostly stable and I try to do what is expected from me.",
            study_interest="biology",
            second_interest="public health",
            example="I read extra materials sometimes and I help in class when group tasks need cooperation.",
            example_detail="When I reread my application, I see that I keep returning to similar general sentences instead of giving better proof.",
            local_issue="students often say they care about health and science, but their understanding stays at a very general level",
            contribution="I would contribute effort, responsibility, and a positive attitude.",
            support_need="more concrete experience and more specific self-presentation",
            uncertainty="There is enough to read, but not enough to judge strongly.",
            qa_profile="repetitive_generic",
            interview_mode="balanced_plain",
            interview_hook="I can mainly say that I am serious, though I know that still sounds general.",
            completion_rate=0.74,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_028",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="partial_answers",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=77.0,
            school_type="City school diploma",
            school_score=83.0,
            city="Turkistan",
            opening="Law interests me because rules shape everyday life more than people notice, and I want to understand that better.",
            school_context="My school life was quite ordinary, mostly focused on lessons and exams.",
            study_interest="law",
            second_interest="public systems",
            example="I paid attention when school rules felt unfair or unclear, and sometimes discussed that with classmates.",
            example_detail="Still, I do not have a developed example that turns this interest into strong evidence.",
            local_issue="students often experience rules only as pressure and not as something that can be examined or improved",
            contribution="I would contribute careful discussion and respect for other views.",
            support_need="more experience connecting abstract interest with real action",
            uncertainty="Some questions got only half-answers because my experience is still small.",
            qa_profile="thin_partial",
            interview_mode="balanced_plain",
            interview_hook="My interest is ahead of my track record right now.",
            completion_rate=0.76,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_029",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="thin_examples",
            intended_ambiguity="hard",
            text_length="medium",
            english_type="school classes",
            english_score=70.0,
            school_type="Regional school certificate",
            school_score=78.0,
            city="Kostanay",
            opening="Environmental study attracts me because I notice local problems and want to do more than only complain about them.",
            school_context="At the same time, most of my application comes from ordinary school life rather than clear projects.",
            study_interest="environmental science",
            second_interest="community planning",
            example="I joined one cleanup day and followed local discussions about waste and drainage.",
            example_detail="Those are real experiences, but not enough to carry a strong evaluation alone.",
            local_issue="small environmental problems become normal in daily life because nobody expects students to observe them seriously",
            contribution="I can contribute reliability and willingness to help with practical tasks.",
            support_need="more structured project experience and better written reflection",
            uncertainty="The case is thin but not empty.",
            qa_profile="thin_examples",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.66,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_030",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="thin_examples",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=78.0,
            school_type="City school diploma",
            school_score=84.0,
            city="Astana",
            opening="Management and organization interest me because I like when a group becomes clearer and less chaotic.",
            school_context="In school I have mostly shown this in small ways and not in major positions.",
            study_interest="management",
            second_interest="organizational psychology",
            example="I kept track of deadlines in group tasks and sometimes reminded classmates what was still missing.",
            example_detail="It is a real contribution, but not one that says much about initiative depth by itself.",
            local_issue="many student groups lose energy because nobody keeps the simple organizational side in view",
            contribution="I would contribute planning and follow-up.",
            support_need="bigger responsibilities and stronger examples beyond routine reliability",
            uncertainty="Right now the application shows a useful habit more than a developed record.",
            qa_profile="thin_examples",
            interview_mode="balanced_plain",
            interview_hook="I can describe small responsibilities, not a strong project history.",
            completion_rate=0.79,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_031",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="partial_answers",
            intended_ambiguity="hard",
            text_length="medium",
            english_type="private tutoring",
            english_score=80.0,
            school_type="Lyceum certificate",
            school_score=87.0,
            city="Aktobe",
            opening="I am applying for mathematics because I enjoy difficult material and want a place where I can keep pushing that side of myself.",
            school_context="Academically I am fine, but my application outside grades is much thinner.",
            study_interest="mathematics",
            second_interest="data analysis",
            example="Mostly I have solved problems, helped a few classmates, and prepared for exams.",
            example_detail="That is honest, but it leaves several form questions only partly answered in substance.",
            local_issue="students who are good at exam tasks do not always learn how to connect that strength with broader initiative",
            contribution="I can contribute persistence and serious preparation.",
            support_need="more breadth and more evidence outside coursework",
            uncertainty="The application may be too thin to judge fairly even though it is not weak in every respect.",
            qa_profile="thin_partial",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.71,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_032",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="thin_examples",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=73.0,
            school_type="Standard school diploma",
            school_score=79.0,
            city="Petropavl",
            opening="Media and communication interest me because I notice how much tone and wording change whether people pay attention.",
            school_context="My school experience in this area is still limited and mostly informal.",
            study_interest="media studies",
            second_interest="communications",
            example="I sometimes rewrote messages for group chats so classmates understood deadlines better.",
            example_detail="It is a useful habit, but I know it is a narrow example for a university application.",
            local_issue="important information is often available but still not understood because it is written badly or too late",
            contribution="I would contribute careful wording and practical help in group coordination.",
            support_need="more visible work and stronger examples that go beyond everyday usefulness",
            uncertainty="The profile is readable but underdeveloped.",
            qa_profile="thin_examples",
            interview_mode="balanced_plain",
            interview_hook="Most of my signal is in small everyday communication rather than projects.",
            completion_rate=0.75,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_033",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="long_but_vague",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=77.0,
            school_type="City school diploma",
            school_score=83.0,
            city="Uralsk",
            opening="I have enough motivation to write at length about psychology and learning, but not enough concrete material yet to make that length fully persuasive.",
            school_context="My school years were steady, and I often noticed how confidence, attention, and classroom atmosphere shape who participates.",
            study_interest="psychology",
            second_interest="education",
            example="I listened to classmates, helped with summaries, and paid attention to how different students reacted under pressure.",
            example_detail="These are all real observations, but when I try to turn them into an application narrative, I end up expanding reflection more than evidence. Even my strongest examples stay close to everyday support and not to a clear project with outcomes.",
            local_issue="many students are described as lazy when the real problem is a mix of uncertainty, poor habits, and not asking for help early enough",
            contribution="I would contribute empathy, steadiness, and a serious attitude toward group responsibilities.",
            support_need="more concrete practice and stronger examples before broad reflection becomes convincing",
            uncertainty="This is a long application with a thin center and no strong outcome story behind it yet.",
            qa_profile="long_vague",
            interview_mode="balanced_plain",
            interview_hook="I can talk at length, but I know that length is not the same as proof.",
            completion_rate=0.85,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_034",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="long_but_vague",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=79.0,
            school_type="Lyceum certificate",
            school_score=86.0,
            city="Taldykorgan",
            opening="I can articulate why public systems and economics matter to me, but I am aware that articulation is carrying more weight in this application than lived evidence.",
            school_context="My academics are acceptable, and I have spent time thinking about how institutions affect ordinary people in ways students do not always notice.",
            study_interest="economics",
            second_interest="public policy",
            example="I followed local discussions, talked with teachers, and tried to connect school subjects with wider questions.",
            example_detail="Still, most of this stayed at the level of reading and reflection rather than action, which is why the application remains difficult to judge well.",
            local_issue="people discuss prices, transport, and public services every day, but students rarely have structured chances to examine those systems seriously",
            contribution="I would contribute thoughtful reading and respectful discussion.",
            support_need="more real examples, more initiative, and less dependence on broad analytical language",
            uncertainty="A reviewer could read this as sincere and still conclude that there is not enough evidence yet.",
            qa_profile="long_vague",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.8,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_019",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="very_short",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school classes",
            english_score=71.0,
            school_type="City school diploma",
            school_score=80.0,
            city="Almaty",
            opening="I am applying to inVision U because I want a more serious place to study economics than the one I can build for myself now.",
            school_context="My school record is stable and I mostly focused on ordinary class work.",
            study_interest="economics",
            second_interest="data literacy",
            example="I sometimes helped classmates compare homework answers before tests.",
            example_detail="I know this is not a strong example, but it is the most honest one.",
            local_issue="many students talk about future jobs without understanding the systems behind daily money decisions",
            contribution="I would bring regular attendance and a cooperative attitude.",
            support_need="clearer direction and stronger examples of independent work",
            uncertainty="The motivation is real, but the evidence is still thin.",
            qa_profile="thin_basic",
            interview_mode="balanced_plain",
            interview_hook="I do not have a big story here, only a real wish to study more seriously.",
            completion_rate=0.68,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_020",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="very_short",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school + apps",
            english_score=69.0,
            school_type="Standard school diploma",
            school_score=78.0,
            city="Shymkent",
            opening="I want to study psychology at inVision U because I think university can help me understand people and myself better.",
            school_context="At school I usually did what was required and did not build many activities around it.",
            study_interest="psychology",
            second_interest="education",
            example="I was mostly the person who listened to friends when they were stressed before exams.",
            example_detail="This matters to me, although I know it is not strong evidence on its own.",
            local_issue="students often hide stress until it becomes bigger than it should be",
            contribution="I would contribute a calm presence and willingness to learn with others.",
            support_need="more concrete experience and more confidence describing what I can already do",
            uncertainty="Right now the application shows intention more than proof.",
            qa_profile="thin_basic",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.64,
            returned_to_edit=False,
            skipped_optional_questions=4,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_021",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="very_short",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school + self-study",
            english_score=73.0,
            school_type="Regional school certificate",
            school_score=82.0,
            city="Kyzylorda",
            opening="I am interested in computer science and I want a university where I can build better habits from the beginning.",
            school_context="My school work in math and informatics is acceptable, but most of my profile stays inside normal assignments.",
            study_interest="computer science",
            second_interest="information systems",
            example="I watched tutorials and practiced small tasks, but I have not built a clear personal project yet.",
            example_detail="That is exactly why the application may feel incomplete in substance.",
            local_issue="students from ordinary schools often know they should learn digital skills but do not know where to start beyond random videos",
            contribution="I can contribute patience and consistent work if expectations are clear.",
            support_need="stronger project ownership and less passive learning",
            uncertainty="I am applying before I have enough concrete work to show.",
            qa_profile="thin_basic",
            interview_mode="balanced_plain",
            interview_hook="I can explain the interest better than the evidence.",
            completion_rate=0.7,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_022",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="very_short",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="language center",
            english_score=72.0,
            school_type="City school diploma",
            school_score=79.0,
            city="Taraz",
            opening="Public health interests me because it connects science with ordinary family life, and I want to study that more seriously.",
            school_context="I do not have many activities to describe beyond school and family responsibilities.",
            study_interest="public health",
            second_interest="biology",
            example="Mostly I have paid attention to health topics in my own reading and conversations at home.",
            example_detail="I understand this is limited evidence for an application.",
            local_issue="people often wait too long to ask simple health questions because they do not know whom to ask first",
            contribution="I would contribute seriousness and respect for other people in group work.",
            support_need="more structured exposure to the field and stronger written examples",
            uncertainty="At this stage I am asking to be read as a sincere but still thin case.",
            qa_profile="thin_basic",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.62,
            returned_to_edit=False,
            skipped_optional_questions=4,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_023",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="very_short",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="school classes",
            english_score=70.0,
            school_type="Standard school diploma",
            school_score=77.0,
            city="Aktau",
            opening="I want to study education because I keep thinking about how much a school atmosphere changes whether students try or only wait.",
            school_context="My own profile is not very active, and I know that shows.",
            study_interest="education",
            second_interest="social science",
            example="I sometimes shared notes and reminders with classmates who missed lessons.",
            example_detail="It was useful in a simple way, but that is the level of evidence I currently have.",
            local_issue="in many classrooms the loud students shape the pace while quieter students fall behind without attention",
            contribution="I could contribute reliability and patience.",
            support_need="more concrete experience and more confidence speaking about my motivation",
            uncertainty="The application is usable but not rich.",
            qa_profile="thin_basic",
            interview_mode="balanced_plain",
            interview_hook="I know there is not much here yet, but the motivation itself is honest.",
            completion_rate=0.67,
            returned_to_edit=True,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_024",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="repetitive_generic",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=75.0,
            school_type="City school diploma",
            school_score=81.0,
            city="Karagandy",
            opening="I want inVision U because I want to grow, learn with serious people, and become more useful in the future.",
            school_context="My school experience was mostly normal, with regular classes and some group tasks.",
            study_interest="business",
            second_interest="management",
            example="I did school work responsibly and tried to help when there was group work.",
            example_detail="The problem is that when I explain myself, I repeat this same idea instead of giving a stronger example.",
            local_issue="many students want to improve themselves but do not know how to move from general goals to actual practice",
            contribution="I would contribute teamwork and responsibility.",
            support_need="better examples, clearer self-description, and less repetition",
            uncertainty="The application stays valid, but it does not say enough beyond good intentions.",
            qa_profile="repetitive_generic",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.72,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_025",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="thin_examples",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=76.0,
            school_type="Lyceum certificate",
            school_score=85.0,
            city="Atyrau",
            opening="Data science interests me because I like structured thinking, but my application does not yet show much beyond that starting point.",
            school_context="At school I usually complete tasks carefully and get decent marks in math.",
            study_interest="data science",
            second_interest="economics",
            example="I worked on class assignments with spreadsheets and once helped check a few errors in a shared file.",
            example_detail="This is not nothing, but it is also not enough to make a strong judgment about initiative.",
            local_issue="students collect information in school but rarely learn how to organize it into something useful",
            contribution="I can contribute careful routine work and steady participation.",
            support_need="opportunities to move from coursework into real small projects",
            uncertainty="I think the case is readable, only under-evidenced.",
            qa_profile="thin_examples",
            interview_mode="balanced_plain",
            interview_hook="I mostly have habits and interest, not many visible examples yet.",
            completion_rate=0.78,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_026",
            intended_gap_slice="insufficient_evidence_but_valid_cases",
            slice_variant="partial_answers",
            intended_ambiguity="hard",
            text_length="medium",
            english_type="school classes",
            english_score=71.0,
            school_type="Regional school certificate",
            school_score=80.0,
            city="Semey",
            opening="I am applying because sociology seems close to questions I notice in everyday life, especially around how students decide what is possible for them.",
            school_context="I have more observations than real actions so far.",
            study_interest="sociology",
            second_interest="education",
            example="I talked with classmates about study choices and sometimes wrote down their concerns for myself.",
            example_detail="Beyond that, my application becomes less complete because I did not always know how to answer the form in detail.",
            local_issue="students often follow the loudest advice available and not the advice that fits them",
            contribution="I would contribute listening and seriousness.",
            support_need="better reflection in writing and more experience outside ordinary class life",
            uncertainty="Some answers are partial because I genuinely did not have enough to say yet.",
            qa_profile="thin_partial",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.69,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_013",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="thin_qa_neighbor",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=76.0,
            school_type="Standard school diploma",
            school_score=81.0,
            city="Petropavl",
            opening="The part of my application that feels most honest is the interest; the part that feels weakest is how thinly I sometimes answer once the form asks for detail.",
            school_context="My school record is steady and I have spent extra time around informatics and math, though my activities outside class remained modest.",
            study_interest="data science",
            second_interest="education",
            example="I helped classmates clean up messy spreadsheet work for one school survey and enjoyed the practical side more than I expected.",
            example_detail="That example is small, but it is the truest one I have, which is why the application may look stronger in the letter than in the shorter answers.",
            local_issue="students are often asked for feedback in school, but the results are rarely organized in a way that makes anyone trust the process",
            contribution="I would contribute accuracy in routine tasks, patience with confused data, and a cooperative attitude in group assignments.",
            support_need="more practice turning small evidence into clear written explanation without either overselling it or hiding it",
            uncertainty="Some of my short answers probably make the overall case look more generic than I intended.",
            qa_profile="thin_partial",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.78,
            returned_to_edit=False,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_014",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="overpolished_thin",
            intended_ambiguity="hard",
            text_length="long",
            english_type="private tutoring",
            english_score=91.0,
            school_type="City school diploma",
            school_score=88.0,
            city="Taldykorgan",
            opening="I know that some readers become cautious when an application sounds polished but keeps returning to the same level of evidence.",
            school_context="I read and write comfortably in English, so it is possible for me to produce a smooth statement even when the underlying examples are still at school scale.",
            study_interest="education",
            second_interest="media studies",
            example="My most repeated example is helping teachers and classmates prepare concise material for discussion sessions and school presentations.",
            example_detail="That work was useful and not invented, but it also remained ordinary support work, which means the fluency of my language may create more confidence than the evidence itself deserves.",
            local_issue="students often consume information quickly but struggle to distinguish between a clear explanation and a convincing but shallow one",
            contribution="I would contribute writing discipline, serious reading, and a willingness to support group preparation before a task becomes urgent.",
            support_need="more demanding environments that test whether my communication can stay strong when the evidence has to become deeper",
            uncertainty="If someone asked for a stricter audit of substance versus style, I would understand why.",
            qa_profile="generic_polished",
            interview_mode="weaker_nervous",
            interview_hook="In interview I sound more like an ordinary applicant than the letter might suggest.",
            completion_rate=0.95,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_015",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="sincere_oddly_framed",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=74.0,
            school_type="Regional school certificate",
            school_score=79.0,
            city="Ekibastuz",
            opening="I keep noticing that when I care about a topic, I start describing it in future-oriented language before I have enough present evidence.",
            school_context="At school I was usually more involved in practical logistics than in the visible center of student activities.",
            study_interest="renewable energy",
            second_interest="public systems",
            example="I once helped a physics teacher gather examples and simple visuals for a classroom discussion about local energy use.",
            example_detail="It was a narrow task, but I started talking about it as if it were the beginning of a larger civic interest because that is genuinely how my mind connected it.",
            local_issue="people speak about energy and infrastructure as distant technical matters even though they shape ordinary family decisions every winter",
            contribution="I could contribute seriousness, background preparation, and respect for people who are less confident speaking early in group settings.",
            support_need="clearer distinction between first interest, actual evidence, and future possibility",
            uncertainty="The sincerity is real, but the framing is sometimes one step ahead of the proof.",
            qa_profile="mildly_drifting",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.75,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_016",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="style_gap",
            intended_ambiguity="hard",
            text_length="long",
            english_type="private tutoring",
            english_score=89.0,
            school_type="City school diploma",
            school_score=87.0,
            city="Almaty Region",
            opening="The written form catches my reflective side very well, but it hides how ordinary and hesitant I can sound in unscripted conversation.",
            school_context="I am interested in journalism and social research, though my actual school experience has been less formal than those labels may imply.",
            study_interest="journalism",
            second_interest="social research",
            example="I helped collect short comments from students about how they receive school announcements and then rewrote the information into a cleaner format for one teacher.",
            example_detail="It was a small communication task, not a newsroom or a research internship, and that difference matters because my polished writing can blur it.",
            local_issue="important information at school is often technically available but still missed because it is written in a way students stop reading",
            contribution="I would contribute careful editing, thoughtful listening, and a habit of asking whether communication actually reaches the people it is supposed to reach.",
            support_need="more confidence in live speaking, because my written phrasing currently performs above my real-time communication",
            uncertainty="The mismatch is not intentional image-making, but it is a real mismatch.",
            qa_profile="measured",
            interview_mode="mixed_casual",
            interview_hook="When I answer quickly, I sound much more unfinished than the letter.",
            completion_rate=0.91,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_017",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="claims_weak_grounding",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=75.0,
            school_type="City school diploma",
            school_score=84.0,
            city="Kyzylorda",
            opening="My application makes the most sense if it is read as a real interest with uneven proof rather than as a finished record of achievement.",
            school_context="I study in a city school where economics became interesting to me mostly through ordinary family and market conversations, not formal competitions.",
            study_interest="agricultural economics",
            second_interest="data analysis",
            example="I described one class assignment about tracking local fruit and vegetable prices as if I had coordinated a market study.",
            example_detail="What really happened was that I compared prices with two classmates for several weeks and wrote the summary more confidently than the project itself deserved.",
            local_issue="families pay close attention to price changes, but students rarely learn how to observe those shifts carefully instead of turning them into loose opinions",
            contribution="I would contribute discipline with simple records, patience in group tasks, and practical curiosity about everyday economic questions.",
            support_need="better methodological habits so that my interest in data becomes more than a vocabulary around basic observations",
            uncertainty="The evidence is genuine but smaller than the phrasing can make it sound.",
            qa_profile="thin_specifics",
            interview_mode="balanced_plain",
            interview_hook="In speech I explain the scale more honestly and less elegantly.",
            completion_rate=0.83,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_018",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="generic_high_confidence",
            intended_ambiguity="hard",
            text_length="long",
            english_type="language center",
            english_score=85.0,
            school_type="Lyceum certificate",
            school_score=86.0,
            city="Aktau",
            opening="I know how to speak confidently about growth, collaboration, and long-term goals, and I also know that confidence is not the same thing as groundedness.",
            school_context="My school background is solid and I have tried to stay engaged, but the visible examples remain more ordinary than my framing can sound.",
            study_interest="international relations",
            second_interest="social analysis",
            example="I joined discussion events, helped with preparation for one school forum, and spent time reading about regional issues beyond class requirements.",
            example_detail="All of that is true, but it still amounts to a candidate whose intellectual language is more developed than the scale of direct evidence underneath it.",
            local_issue="students often talk about global change with confidence while staying vague about what they have actually done in local settings",
            contribution="I could contribute reading discipline, serious discussion, and a willingness to help prepare shared work before deadlines become chaotic.",
            support_need="more pressure to attach claims to concrete evidence and not rely on broad strategic language",
            uncertainty="I would not call the application false, but I would understand why it invites a second look.",
            qa_profile="generic_polished",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.87,
            returned_to_edit=False,
            skipped_optional_questions=1,
        ),
    ]
)

SEEDS.extend(
    [
        CandidateSeed(
            candidate_id="syn_gap_v7_007",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="polished_thin",
            intended_ambiguity="hard",
            text_length="long",
            english_type="private tutoring",
            english_score=90.0,
            school_type="Lyceum certificate",
            school_score=88.0,
            city="Kokshetau",
            opening="A careful reader will probably notice that my application is strongest when it is talking about intentions and weakest when it is forced to quantify outcomes.",
            school_context="I have done well in literature and social science courses, and I have also tried to make myself useful in peer spaces where school stress becomes visible.",
            study_interest="sociology",
            second_interest="youth transitions",
            example="The example I return to most is a peer survey on study habits that I coordinated with classmates before graduation.",
            example_detail="I can explain why we made it and what patterns we noticed, but the truth is that the survey was small, informal, and not the kind of evidence that fully supports the polished way I sometimes summarize it.",
            local_issue="many students appear capable on paper while privately feeling lost about how to move from school routines into adult decisions",
            contribution="I would contribute prepared discussion, good note discipline, and a habit of taking group work seriously even when no one is supervising it.",
            support_need="stronger methodological discipline so that my interpretation does not outrun the quality of the evidence underneath it",
            uncertainty="I believe the motivation is sincere, but I also understand why some parts could sound over-produced compared with the modest size of the work itself.",
            qa_profile="generic_polished",
            interview_mode="weaker_nervous",
            interview_hook="In speaking I become much less assured, which may actually reveal the true level of certainty better than the letter does.",
            completion_rate=0.94,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_008",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="sincere_oddly_framed",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + self-study",
            english_score=77.0,
            school_type="Regional school certificate",
            school_score=82.0,
            city="Taraz",
            opening="I sometimes write like I am defending an idea instead of simply describing what I did, and this application may carry that habit.",
            school_context="My school did not have a formal urban studies track, so most of my interest in planning came from observing ordinary places and talking with people in my neighborhood.",
            study_interest="architecture",
            second_interest="urban planning",
            example="I joined a simple neighborhood mapping walk with a teacher and later drew a cleaner version of the notes for classmates who missed it.",
            example_detail="It was a small activity, but I keep describing it as if it were the beginning of a bigger civic process because that is honestly how it felt to me.",
            local_issue="public spaces near schools are often treated as background even though they shape whether students feel safe, hurried, or willing to stay after class",
            contribution="I would contribute observation, patience, and a habit of turning scattered discussion into something written down and usable.",
            support_need="more grounded academic language so that my enthusiasm does not sound like exaggeration",
            uncertainty="I do not think I am inventing anything, but I know my framing can make simple things sound more strategic than they were.",
            qa_profile="mildly_drifting",
            interview_mode="balanced_plain",
            interview_hook="When I speak, the same story sounds smaller and maybe more believable.",
            completion_rate=0.81,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_009",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="timeline_fuzzy",
            intended_ambiguity="hard",
            text_length="long",
            english_type="school + self-study",
            english_score=80.0,
            school_type="City school diploma",
            school_score=85.0,
            city="Semey",
            opening="I do not want to hide the fact that some of my story is built from repeated small things rather than one clear project with a clean timeline.",
            school_context="Most of my school contribution has been ordinary and recurring, especially around helping classmates prepare before exams and keeping small peer sessions from disappearing.",
            study_interest="education policy",
            second_interest="community development",
            example="For two exam seasons, maybe closer to one and a half if counted strictly, I helped keep weekend revision meetings going for students who were likely to skip them without reminders.",
            example_detail="There were breaks, changes in who came, and periods when it was only a few people, so I understand why a reader might want more precision than I can confidently provide now.",
            local_issue="schools often reward the students who already know how to navigate deadlines while quieter students drift away without much notice",
            contribution="I could contribute follow-through, practical check-ins, and a style of teamwork that notices who has stopped participating.",
            support_need="stronger analytical writing, because my observations are real but still mostly intuitive",
            uncertainty="The activity was real and useful, but the edges of it are less formal than the neat summary version suggests.",
            qa_profile="mixed_specificity",
            interview_mode="clearer_plain",
            interview_hook="I can usually explain the repeated small reality better than the summarized version.",
            completion_rate=0.89,
            returned_to_edit=True,
            skipped_optional_questions=1,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_010",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="generic_polished",
            intended_ambiguity="borderline",
            text_length="short",
            english_type="language center",
            english_score=84.0,
            school_type="Standard school diploma",
            school_score=85.0,
            city="Kostanay",
            opening="I am drawn to inVision U because it seems like the kind of place where intellectually serious students can grow into responsible contributors.",
            school_context="My academic results are consistent and I approach tasks with discipline.",
            study_interest="law",
            second_interest="public policy",
            example="I have tried to take school responsibilities seriously and to support collaborative work when needed.",
            example_detail="At the same time, I know that this description remains broad and does not fully show the scale of my activity.",
            local_issue="many students discuss fairness and public rules abstractly without seeing how policy shapes ordinary decisions",
            contribution="I would contribute thoughtful participation and reliability.",
            support_need="clearer evidence behind my larger interests",
            uncertainty="A human reviewer could reasonably feel that the polish exceeds the detail.",
            qa_profile="generic_polished",
            interview_mode="none",
            interview_hook="",
            completion_rate=0.7,
            returned_to_edit=False,
            skipped_optional_questions=3,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_011",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="claims_weak_grounding",
            intended_ambiguity="borderline",
            text_length="medium",
            english_type="school + apps",
            english_score=78.0,
            school_type="City school diploma",
            school_score=82.0,
            city="Uralsk",
            opening="One risk in my application is that I naturally describe effort in a forward-looking way, even when the completed result is still modest.",
            school_context="At school I was often interested in entrepreneurship language, but the actual things I did stayed small and local.",
            study_interest="entrepreneurship",
            second_interest="social enterprise",
            example="I wrote about launching an eco-point for recyclables near our school entrance.",
            example_detail="A more precise description is that I helped place one collection box, kept a handwritten schedule for a while, and talked to students about using it before the routine faded.",
            local_issue="environmental discussion among students often becomes symbolic very quickly because the follow-through is weaker than the first enthusiasm",
            contribution="I can contribute energy at the start of a task and honest reflection about what still needs stronger structure.",
            support_need="discipline that turns good framing into steadier execution",
            uncertainty="I do not want to oversell what was partly an experiment and partly a first attempt.",
            qa_profile="thin_specifics",
            interview_mode="balanced_plain",
            interview_hook="Saying it aloud usually makes me describe the project in a less inflated way.",
            completion_rate=0.82,
            returned_to_edit=True,
            skipped_optional_questions=2,
        ),
        CandidateSeed(
            candidate_id="syn_gap_v7_012",
            intended_gap_slice="authenticity_manual_review_cases",
            slice_variant="section_mismatch",
            intended_ambiguity="hard",
            text_length="long",
            english_type="private tutoring",
            english_score=86.0,
            school_type="Lyceum certificate",
            school_score=90.0,
            city="Turkistan",
            opening="This application shows a real intellectual path, but not yet a fully consistent one.",
            school_context="My school years were strongest in biology and chemistry, though the questions that stayed with me were often about how people understand health information rather than science alone.",
            study_interest="biology",
            second_interest="health communication",
            example="I helped prepare short health-awareness talks for younger students and also answered practical questions from relatives when appointments and forms became confusing.",
            example_detail="That mix is exactly why some parts of my application sound scientific while others sound closer to psychology or communication, because I am still trying to understand where the true center is.",
            local_issue="people often avoid preventive care not because they reject it, but because the process feels distant, confusing, or embarrassing",
            contribution="I would contribute careful preparation, respectful listening, and a willingness to connect technical material with ordinary human concerns.",
            support_need="time and advising to refine a direction that is interdisciplinary in a genuine way and not only in a vague way",
            uncertainty="The overlap between my interests is real, but the coherence is still developing.",
            qa_profile="mismatch_drifting",
            interview_mode="clearer_plain",
            interview_hook="When I speak directly, I admit faster that the center of gravity is still moving.",
            completion_rate=0.93,
            returned_to_edit=True,
            skipped_optional_questions=0,
        ),
    ]
)

if __name__ == "__main__":
    main()
