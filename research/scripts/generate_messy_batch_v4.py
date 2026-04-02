from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.input import CandidateInput

RAW_DIR = ROOT / "data" / "ml_workbench" / "raw" / "generated" / "messy_batch_v4"
PACK_DIR = ROOT / "data" / "ml_workbench" / "processed" / "annotation_packs" / "messy_batch_v4"

RAW_JSONL = RAW_DIR / "messy_batch_v4_api_input.jsonl"
GEN_MANIFEST_JSONL = RAW_DIR / "messy_batch_v4_generation_manifest.jsonl"
SUMMARY_JSON = RAW_DIR / "messy_batch_v4_summary.json"

PACK_JSONL = PACK_DIR / "messy_batch_v4_annotation_pack.jsonl"
PACK_JSON = PACK_DIR / "messy_batch_v4_annotation_pack.json"
PACK_TABLE_CSV = PACK_DIR / "messy_batch_v4_annotation_pack_table.csv"
PACK_MANIFEST_JSON = PACK_DIR / "messy_batch_v4_annotation_pack_manifest.json"

WORD_RE = re.compile(r"\b[\w']+\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

REALISM_TARGET = {
    "sincere_low_polish_real_signal": 10,
    "polished_but_socially_desirable_thin": 8,
    "patchy_partially_answered_borderline": 8,
    "mild_consistency_noise_not_fraud": 6,
    "translated_thinking_english_or_oral_style": 8,
}

AMBIGUITY_TARGET = {"clear": 8, "borderline": 20, "hard": 12}
TEXT_TARGET = {"short": 8, "medium": 18, "long": 14}
INTERVIEW_TARGET = {"with_interview": 28, "without_interview": 12}

QUESTION_BANK: list[tuple[str, str]] = [
    ("local_problem", "What local problem would you want to work on through study at inVision U?"),
    ("contribute_peers", "How would you contribute to your peers in the learning environment?"),
    ("hard_feedback", "Tell us about difficult feedback you received and what you changed."),
    ("changed_mind", "Describe a time you changed your mind after learning something new."),
    ("low_confidence", "When did you continue despite low confidence?"),
    ("difficult_people", "What kind of people are difficult for you to work with, and why?"),
    ("values_easy_hard", "Which value is easy for you and which is difficult in practice?"),
    ("support_need", "What support do you think you might need to succeed at university?"),
    ("helped_consistently", "Tell us about someone you helped consistently, not just once."),
    ("started_not_finish", "Describe something you started but could not finish. What happened?"),
    ("why_fit", "Why do you think inVision U is a good learning fit for you?"),
    ("study_interest", "What do you want to study or explore first and why?"),
    ("learning_environment", "What kind of environment helps you learn best?"),
    ("group_role", "What role do you usually take in group settings?"),
]

CITIES = [
    "Almaty",
    "Astana",
    "Shymkent",
    "Karaganda",
    "Aktobe",
    "Pavlodar",
    "Semey",
    "Taraz",
    "Kostanay",
    "Atyrau",
    "Ust-Kamenogorsk",
    "Kokshetau",
    "Turkistan",
    "Petropavl",
    "Oral",
    "Ekibastuz",
    "Zhezkazgan",
    "Kyzylorda",
    "Rudny",
    "Taldykorgan",
    "a village near Pavlodar",
    "a town near Kostanay",
    "a district outside Shymkent",
    "a small town near Semey",
    "a neighborhood in Karaganda",
]

INTERESTS = [
    "public health",
    "environmental science",
    "education policy",
    "psychology",
    "economics",
    "sociology",
    "data science",
    "urban studies",
    "biology",
    "chemistry",
    "mathematics",
    "computer science",
    "history",
    "linguistics",
    "social entrepreneurship",
    "community development",
]

SECONDARY_INTERESTS = [
    "communications",
    "policy analysis",
    "learning design",
    "behavioral research",
    "civic technology",
    "public systems",
    "digital literacy",
    "project management",
    "statistics",
    "human-centered design",
]

LOCAL_ISSUES = [
    "students in my area lose confidence after school because there are few structured spaces",
    "public transport delays make it hard for students to join after-class activities",
    "many families cannot access simple academic support outside school",
    "waste and litter around school routes make the neighborhood feel neglected",
    "young people often leave small towns because they do not see learning pathways",
    "local schools have limited lab resources so practical learning is weak",
    "mental-health conversations are still avoided in many student groups",
    "parents and students do not always understand scholarship and application processes",
    "there are not enough inclusive peer spaces for students with different backgrounds",
    "air quality concerns are discussed but rarely turned into practical local action",
    "digital skills are expected, but many students still struggle with basic tools",
    "group projects at school often focus on grades, not actual problem solving",
    "younger students need role models who can show realistic academic growth",
    "many community ideas start with energy but stop because structure is missing",
    "students with part-time family duties can be underestimated in school",
]

INITIATIVES = [
    "started a small peer study table after classes and kept shared notes for classmates",
    "organized a weekly revision chat where we tracked deadlines and tasks",
    "helped set up a simple neighborhood cleanup and data log with two classmates",
    "created a basic question bank for juniors preparing for final exams",
    "ran a short reading group for younger students in my building",
    "helped a teacher organize digital assignment folders and submission instructions",
    "coordinated a small support circle for classmates who were often absent",
    "tried a low-cost mini project to map common student commute problems",
    "prepared simple learning summaries and shared them before exam periods",
    "organized two practical discussion sessions about study habits and planning",
    "kept a spreadsheet of opportunities and deadlines and shared it with peers",
    "started a small experiment notebook and discussed results with classmates",
    "helped classmates with speaking practice before oral presentations",
    "set up a borrowed-material shelf with notebooks and reference sheets",
    "created a simple checklist system for group-project task ownership",
]

OUTCOMES = [
    "a few classmates used it regularly, though the effort stayed small and informal",
    "it helped during one exam cycle, but we still lacked long-term consistency",
    "students said it reduced confusion, even if participation was uneven",
    "teachers noticed better preparation in some groups, but not across all classes",
    "it worked for a while and then slowed down during exam season",
    "the process was useful but depended too much on me reminding everyone",
    "it gave clearer structure, but the scale remained limited",
    "the result was modest, yet it showed me what should be improved next",
    "it improved confidence for several students, even without formal recognition",
    "the initiative stayed local but still felt meaningful to those involved",
]

SUPPORT_NEEDS = [
    "academic writing structure",
    "seminar speaking confidence",
    "time management in open-ended assignments",
    "project planning with realistic milestones",
    "English discussion fluency",
    "early mentoring on course choices",
    "feedback on argument clarity",
    "transition support for first-semester workload",
]

ENGLISH_TYPES = [
    "school classes",
    "school + self-study",
    "language center",
    "school + tutoring",
    "school + online courses",
    "school + apps",
    "private tutoring",
    "school + YouTube",
    "school + TV series",
]

SCHOOL_TYPES = [
    "Kazakhstan high school diploma",
    "Kazakhstan school certificate",
    "Standard school diploma",
    "Urban gymnasium diploma",
    "Lyceum certificate",
    "NIS certificate",
    "Rural school certificate",
    "City school diploma",
]


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    intended_realism_slice: str
    intended_ambiguity: str
    text_length_bucket: str
    has_interview: bool
    interview_quality: str
    city: str
    interest: str
    secondary_interest: str
    local_issue: str
    initiative: str
    initiative_outcome: str
    support_need: str
    repeated_example: str
    english_type: str
    english_score: float
    school_type: str
    school_score: float
    completion_rate: float
    returned_to_edit: bool
    skipped_optional_questions: int


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def bucket_from_word_count(word_count: int) -> str:
    if word_count <= 90:
        return "short"
    if word_count <= 220:
        return "medium"
    return "long"


def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    compact = "\n".join(lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    compact = re.sub(r"[ \t]+", " ", compact)
    return compact.strip()


def trim_to_words(text: str, max_words: int) -> str:
    words = WORD_RE.findall(text)
    if len(words) <= max_words:
        return text
    keep = words[:max_words]
    return " ".join(keep).rstrip(" ,;") + "."


def expand_for_bucket(text: str, plan: CandidatePlan, rng: random.Random) -> str:
    target_ranges = {"short": (55, 90), "medium": (120, 220), "long": (240, 360)}
    low, high = target_ranges[plan.text_length_bucket]

    expansions = {
        "sincere_low_polish_real_signal": [
            "I know this is not a dramatic story, but it is real in my daily life.",
            "I am still learning how to explain my work better, and this is one reason I apply now.",
            "I want to study in a place where steady effort is valued, not only polished language.",
        ],
        "polished_but_socially_desirable_thin": [
            "I believe collaborative values and responsible leadership are central for meaningful growth.",
            "This motivates me to keep developing both academically and personally in a diverse environment.",
            "I hope to contribute positively while continuing to refine my long-term direction.",
        ],
        "patchy_partially_answered_borderline": [
            "I may be explaining this in not the best order, but the point for me is still important.",
            "Sometimes I have a clear idea and then I lose structure when writing too much.",
            "I think I can improve this with feedback and stronger habits in university.",
        ],
        "mild_consistency_noise_not_fraud": [
            "My focus changed a little over time, but the main motivation to study seriously stayed the same.",
            "I can see some parts of my application are uneven, and I prefer to be open about that.",
            "I am still connecting these interests into one clearer direction.",
        ],
        "translated_thinking_english_or_oral_style": [
            "For me it is important to say this direct, even if language is not always smooth.",
            "I am trying to show real situation, not only nice words in application.",
            "Maybe structure is not perfect, but motivation is honest and practical for me.",
        ],
    }

    text = normalize_spaces(text)
    while count_words(text) < low:
        text += "\n\n" + rng.choice(expansions[plan.intended_realism_slice])

    if count_words(text) > high:
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
        while sentences and count_words(" ".join(sentences)) > high:
            sentences.pop()
        text = " ".join(sentences).strip()

    if count_words(text) > high:
        text = trim_to_words(text, high)

    return normalize_spaces(text)


def style_polished(sentence: str) -> str:
    return sentence


def style_translated(sentence: str) -> str:
    return sentence.replace("I ", "I ").replace("because", "because, for me,")


def compose_letter(plan: CandidatePlan, rng: random.Random) -> str:
    anchor = plan.repeated_example

    if plan.intended_realism_slice == "sincere_low_polish_real_signal":
        parts = [
            f"I am applying to inVision U from {plan.city} because I want to study {plan.interest} in a stronger learning environment. My school results are mostly steady, but my presentation is not always strong.",
            f"One real thing I did is that I {plan.initiative}. {plan.initiative_outcome}. {anchor}",
            f"I am not trying to sound perfect. I know I still need support with {plan.support_need}. At the same time, I can contribute steady effort and practical teamwork.",
            f"A local issue that stays with me is that {plan.local_issue}. I want to learn methods that are more structured, so small efforts can become reliable over time.",
            "For me, university is also about learning with peers from different backgrounds and building better habits, not only getting grades.",
        ]

    elif plan.intended_realism_slice == "polished_but_socially_desirable_thin":
        parts = [
            f"I am motivated to join inVision U because I value responsible learning, collaboration, and meaningful growth. My current interests include {plan.interest} and {plan.secondary_interest}.",
            f"In school I have tried to stay engaged in a constructive way. I {plan.initiative}, and this reinforced my commitment to contributing to the wider student community.",
            f"I am particularly attentive to the fact that {plan.local_issue}. I hope to approach this through interdisciplinary learning and thoughtful collaboration.",
            "I believe I can contribute a positive mindset, respect for others, and willingness to work hard across diverse teams.",
            f"At the same time, I understand I should further develop concrete execution depth and {plan.support_need} to maximize long-term impact.",
        ]

    elif plan.intended_realism_slice == "patchy_partially_answered_borderline":
        parts = [
            f"I want to apply to inVision U because in {plan.city} I started to care more about {plan.interest}, and also because I feel my current school context is limited in practical opportunities.",
            f"I {plan.initiative}. {plan.initiative_outcome}. This part was useful, though I am not sure I explained it fully.",
            f"About local context, {plan.local_issue}. I think this matters, maybe more than we discuss at school. The way I connect this to study is still in progress.",
            "Sometimes I write too much in one section and then answer other parts too briefly. I know this weakness and I am trying to improve it.",
            f"I would probably need support with {plan.support_need}, but I still want to contribute to peers in practical and consistent ways.",
        ]

    elif plan.intended_realism_slice == "mild_consistency_noise_not_fraud":
        parts = [
            f"I am applying to inVision U mainly for {plan.interest}, but I also keep interest in {plan.secondary_interest}. This changed over time, not as contradiction, more as exploration.",
            f"From my city context, {plan.local_issue}. Because of this, I {plan.initiative}. {plan.initiative_outcome}.",
            f"In some periods I focused more on project work, in some periods more on exam stability, and this made my profile a little uneven.",
            f"I can contribute careful group work and persistence, but I still need to improve {plan.support_need} and clearer long-term focus.",
            f"{anchor} I think university can help me align motivation, method, and communication in a more coherent way.",
        ]

    else:
        parts = [
            f"I am writing from {plan.city}, and I want to study at inVision U because for me it is important to learn {plan.interest} with real practice, not only theory.",
            f"In my school life, I {plan.initiative}. {plan.initiative_outcome}. This is small scale, but for me it shows direction.",
            f"Also in local context, {plan.local_issue}. When I see this, I feel I should not only complain but try action and learning together.",
            f"Sometimes my writing structure is awkward, and my English is not always smooth. I still try, and I know I need support with {plan.support_need}.",
            f"{anchor} I want to become student who can connect community problem and study method in more mature way.",
        ]

    if plan.text_length_bucket == "short":
        letter = " ".join(parts[:2])
    elif plan.text_length_bucket == "medium":
        letter = "\n\n".join(parts[:3])
    else:
        letter = "\n\n".join(parts)

    return expand_for_bucket(letter, plan, rng)


def base_answer(plan: CandidatePlan, key: str) -> str:
    if key == "local_problem":
        return f"The issue I want to work on is that {plan.local_issue}. I would like to study methods that connect data, people, and practical implementation."
    if key == "contribute_peers":
        return "I can contribute reliable preparation, patient explanation, and follow-through in group work, especially when tasks are not glamorous."
    if key == "hard_feedback":
        return "A teacher told me my ideas were stronger than my structure. I started outlining first and asking for concrete feedback before final submission."
    if key == "changed_mind":
        return f"I used to think only top grades matter, but after I {plan.initiative} I changed my mind and saw that practical consistency also matters."
    if key == "low_confidence":
        return "I continued one project period even when I was unsure about my English and presentation. Doing small steps helped me stay in motion."
    if key == "difficult_people":
        return "I find it difficult when someone dismisses others quickly without listening. I try to ask clarifying questions and keep the group focused on the task."
    if key == "values_easy_hard":
        return "Responsibility is easier for me than self-promotion. It is easier to do the work than to describe it in a confident way."
    if key == "support_need":
        return f"I would benefit from support in {plan.support_need}, especially in the first semester when pace and expectations are higher."
    if key == "helped_consistently":
        return f"I regularly helped younger students with revision materials over several weeks, not one time only. {plan.repeated_example}"
    if key == "started_not_finish":
        return "I started one larger extension of my project but could not keep momentum during exam period. I learned I need better milestone planning."
    if key == "why_fit":
        return "inVision U fits me because I need a place where learning includes discussion, feedback, and practical project work, not only exam output."
    if key == "study_interest":
        return f"I want to begin with {plan.interest} courses and keep some exploration in {plan.secondary_interest} before narrowing my track."
    if key == "learning_environment":
        return "I learn best where expectations are clear, peers are serious, and questions are welcomed without shame."
    if key == "group_role":
        return "In group settings I am often the person who keeps deadlines visible and checks if tasks are actually completed."
    return "I try to answer honestly and show both strengths and limits."


def stylize_answer(plan: CandidatePlan, raw_answer: str, thin: bool) -> str:
    if thin:
        thin_pool = {
            "sincere_low_polish_real_signal": [
                "I am still learning this and trying to improve step by step.",
                "This is important for me, but I still cannot explain it perfectly.",
            ],
            "polished_but_socially_desirable_thin": [
                "I see this as part of my broader commitment to growth and contribution.",
                "I would approach this with responsibility, collaboration, and continuous reflection.",
            ],
            "patchy_partially_answered_borderline": [
                "I think this matters, but I need more time to answer in detail.",
                "I am not sure I can explain fully now, still I am trying.",
            ],
            "mild_consistency_noise_not_fraud": [
                "I am still connecting this point with my wider plan, so answer is not fully complete yet.",
                "My view changed over time and maybe this answer is still in progress.",
            ],
            "translated_thinking_english_or_oral_style": [
                "For me this question is important, but answer maybe not very full now.",
                "I can say short: I am trying, and still improving this part.",
            ],
        }
        return random.choice(thin_pool[plan.intended_realism_slice])

    if plan.intended_realism_slice == "polished_but_socially_desirable_thin":
        return style_polished(raw_answer + " I would also frame this as part of a long-term values-driven trajectory.")

    if plan.intended_realism_slice == "patchy_partially_answered_borderline":
        return raw_answer + " Maybe this answer sounds a bit mixed, but this is how it happened for me."

    if plan.intended_realism_slice == "mild_consistency_noise_not_fraud":
        return raw_answer + " At the same time my emphasis changed in different periods, so some parts were stronger than others."

    if plan.intended_realism_slice == "translated_thinking_english_or_oral_style":
        return style_translated(raw_answer + " I am trying to explain clear, even if sentence sometimes comes awkward.")

    return raw_answer


def pick_question_keys(plan: CandidatePlan, rng: random.Random) -> list[str]:
    keys = [item[0] for item in QUESTION_BANK]

    if plan.intended_realism_slice == "patchy_partially_answered_borderline":
        count = rng.choice([2, 3, 4])
    elif plan.intended_realism_slice == "polished_but_socially_desirable_thin":
        count = rng.choice([3, 4])
    elif plan.intended_realism_slice == "translated_thinking_english_or_oral_style":
        count = rng.choice([3, 4, 5])
    elif plan.intended_realism_slice == "sincere_low_polish_real_signal":
        count = rng.choice([3, 4, 5])
    else:
        count = rng.choice([3, 4, 5])

    return rng.sample(keys, count)


def compose_motivation_questions(plan: CandidatePlan, rng: random.Random) -> list[dict[str, str]]:
    selected = pick_question_keys(plan, rng)

    if plan.intended_realism_slice == "patchy_partially_answered_borderline":
        thin_count = 2 if len(selected) >= 4 else 1
    elif plan.intended_realism_slice == "polished_but_socially_desirable_thin":
        thin_count = 1
    elif plan.intended_realism_slice == "translated_thinking_english_or_oral_style":
        thin_count = 1 if len(selected) >= 3 else 0
    elif plan.intended_realism_slice == "mild_consistency_noise_not_fraud":
        thin_count = 1 if len(selected) == 5 else 0
    else:
        thin_count = 0 if len(selected) <= 3 else 1

    thin_keys = set(rng.sample(selected, thin_count)) if thin_count > 0 else set()

    questions: list[dict[str, str]] = []
    for key in selected:
        question_text = next(text for k, text in QUESTION_BANK if k == key)
        answer = stylize_answer(plan, base_answer(plan, key), key in thin_keys)
        questions.append({"question": question_text, "answer": normalize_spaces(answer)})

    return questions


def compose_interview(plan: CandidatePlan, rng: random.Random) -> str:
    if not plan.has_interview:
        return ""

    if plan.interview_quality == "clearer":
        body = (
            f"To be honest, I explain myself better in conversation than in formal letter. I care about {plan.interest} because in {plan.city} I saw that {plan.local_issue}. "
            f"I {plan.initiative}, and {plan.initiative_outcome}. I know I still need support with {plan.support_need}, but I am ready for serious study and practical teamwork."
        )
    elif plan.interview_quality == "weaker":
        body = (
            f"I am a bit nervous in interview, so maybe I repeat things. I want to study {plan.interest}. I wrote more in my letter. "
            f"I can work hard, but now I am not explaining very smooth. I think inVision U can help me get clearer with time."
        )
    else:
        body = (
            f"In interview I can say direct: I am applying because I want stronger learning in {plan.interest} and better project habits. "
            f"My experience is mostly small scale, like when I {plan.initiative}. I am not presenting huge achievements, but I am serious about growth."
        )

    if plan.intended_realism_slice == "translated_thinking_english_or_oral_style":
        body += " For me, this chance is important, and I am ready to improve step by step."
    elif plan.intended_realism_slice == "patchy_partially_answered_borderline" and plan.interview_quality != "clearer":
        body += " I know answer can sound a bit uneven now, but motivation is real for me."
    elif plan.intended_realism_slice == "polished_but_socially_desirable_thin" and plan.interview_quality == "weaker":
        body += " I think maybe my written tone sounds better than my speaking right now."

    return normalize_spaces(body)


def sample_score(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 1)


def build_candidate_plans(seed: int = 20260402) -> list[CandidatePlan]:
    rng = random.Random(seed)

    candidate_ids = [f"syn_messy_v4_{idx:03d}" for idx in range(1, 41)]

    slices: list[str] = []
    for name, count in REALISM_TARGET.items():
        slices.extend([name] * count)
    rng.shuffle(slices)

    ambiguities: list[str] = []
    for name, count in AMBIGUITY_TARGET.items():
        ambiguities.extend([name] * count)
    rng.shuffle(ambiguities)

    buckets: list[str] = []
    for name, count in TEXT_TARGET.items():
        buckets.extend([name] * count)
    rng.shuffle(buckets)

    interviews = [True] * INTERVIEW_TARGET["with_interview"] + [False] * INTERVIEW_TARGET["without_interview"]
    rng.shuffle(interviews)

    interview_quality_pool = ["clearer"] * 10 + ["weaker"] * 8 + ["balanced"] * 10
    rng.shuffle(interview_quality_pool)
    interview_quality_iter = iter(interview_quality_pool)

    completion_rates = [1.0] * 6
    completion_rates.extend(round(rng.uniform(0.92, 0.98), 2) for _ in range(14))
    completion_rates.extend(round(rng.uniform(0.78, 0.90), 2) for _ in range(20))
    rng.shuffle(completion_rates)

    returned_flags = [True] * 17 + [False] * 23
    rng.shuffle(returned_flags)

    skipped_values = [0] * 9 + [1] * 9 + [2] * 10 + [3] * 8 + [4] * 4
    rng.shuffle(skipped_values)

    used_triples: set[tuple[str, str, str]] = set()

    plans: list[CandidatePlan] = []
    for idx, candidate_id in enumerate(candidate_ids):
        slice_name = slices[idx]

        if slice_name == "translated_thinking_english_or_oral_style":
            english_score = sample_score(rng, 56.0, 76.0)
        elif slice_name == "sincere_low_polish_real_signal":
            english_score = sample_score(rng, 60.0, 82.0)
        elif slice_name == "patchy_partially_answered_borderline":
            english_score = sample_score(rng, 58.0, 80.0)
        elif slice_name == "mild_consistency_noise_not_fraud":
            english_score = sample_score(rng, 62.0, 85.0)
        else:
            english_score = sample_score(rng, 72.0, 93.0)

        school_score = sample_score(rng, 68.0, 95.0)

        city = rng.choice(CITIES)
        interest = rng.choice(INTERESTS)
        secondary_interest = rng.choice([item for item in SECONDARY_INTERESTS if item != interest])

        for _ in range(30):
            issue = rng.choice(LOCAL_ISSUES)
            initiative = rng.choice(INITIATIVES)
            outcome = rng.choice(OUTCOMES)
            triple = (issue, initiative, outcome)
            if triple not in used_triples:
                used_triples.add(triple)
                break

        has_interview = interviews[idx]
        interview_quality = next(interview_quality_iter) if has_interview else "none"

        plan = CandidatePlan(
            candidate_id=candidate_id,
            intended_realism_slice=slice_name,
            intended_ambiguity=ambiguities[idx],
            text_length_bucket=buckets[idx],
            has_interview=has_interview,
            interview_quality=interview_quality,
            city=city,
            interest=interest,
            secondary_interest=secondary_interest,
            local_issue=issue,
            initiative=initiative,
            initiative_outcome=outcome,
            support_need=rng.choice(SUPPORT_NEEDS),
            repeated_example=rng.choice(
                [
                    "This same example appears in different parts of my application because it is the most concrete experience I had.",
                    "I return to this example often because it shaped how I think about study and contribution.",
                    "I mention this again because it is still my clearest real case, not because I want to overstate it.",
                ]
            ),
            english_type=rng.choice(ENGLISH_TYPES),
            english_score=english_score,
            school_type=rng.choice(SCHOOL_TYPES),
            school_score=school_score,
            completion_rate=completion_rates[idx],
            returned_to_edit=returned_flags[idx],
            skipped_optional_questions=skipped_values[idx],
        )
        plans.append(plan)

    return plans


def build_raw_record(plan: CandidatePlan, submitted_at: str, rng: random.Random) -> dict[str, object]:
    letter = compose_letter(plan, rng)
    questions = compose_motivation_questions(plan, rng)
    interview_text = compose_interview(plan, rng)

    raw = {
        "candidate_id": plan.candidate_id,
        "structured_data": {
            "education": {
                "english_proficiency": {"type": plan.english_type, "score": plan.english_score},
                "school_certificate": {"type": plan.school_type, "score": plan.school_score},
            }
        },
        "text_inputs": {
            "motivation_letter_text": letter,
            "motivation_questions": questions,
            "interview_text": interview_text,
        },
        "behavioral_signals": {
            "completion_rate": plan.completion_rate,
            "returned_to_edit": plan.returned_to_edit,
            "skipped_optional_questions": plan.skipped_optional_questions,
        },
        "metadata": {
            "source": "messy_batch_v4",
            "submitted_at": submitted_at,
            "scoring_version": None,
        },
    }

    raw_validated = CandidateInput.model_validate(raw).model_dump(mode="json", exclude_none=False)
    raw_validated["behavioral_signals"] = {
        key: raw_validated["behavioral_signals"].get(key)
        for key in ("completion_rate", "returned_to_edit", "skipped_optional_questions")
    }

    return raw_validated


def build_hidden_manifest(plan: CandidatePlan) -> dict[str, object]:
    signals_map = {
        "sincere_low_polish_real_signal": [
            "credible small-scale initiative",
            "steady responsibility",
            "genuine motivation",
            "peer contribution",
        ],
        "polished_but_socially_desirable_thin": [
            "strong communication polish",
            "values alignment language",
            "academic consistency",
            "cooperative stance",
        ],
        "patchy_partially_answered_borderline": [
            "some concrete effort",
            "partial reflective capacity",
            "real but uneven motivation",
            "adaptability under uncertainty",
        ],
        "mild_consistency_noise_not_fraud": [
            "honest self-presentation",
            "moderate initiative signal",
            "growth orientation",
            "plausible context fit",
        ],
        "translated_thinking_english_or_oral_style": [
            "sincere intent",
            "practical focus",
            "persistence despite language limits",
            "community orientation",
        ],
    }

    risks_map = {
        "sincere_low_polish_real_signal": [
            "weak presentation quality",
            "limited formal achievements",
            "understated self-advocacy",
        ],
        "polished_but_socially_desirable_thin": [
            "thin concrete evidence",
            "generic impact framing",
            "possible over-framing",
        ],
        "patchy_partially_answered_borderline": [
            "incomplete evidence chain",
            "uneven response quality",
            "low consistency of depth",
        ],
        "mild_consistency_noise_not_fraud": [
            "mild narrative drift",
            "uncertain focus stability",
            "moderate proof density",
        ],
        "translated_thinking_english_or_oral_style": [
            "communication clarity gap",
            "language-mediated under-expression",
            "possible underestimation by polish",
        ],
    }

    noise_base = {
        "sincere_low_polish_real_signal": ["awkward but understandable phrasing", "plain structure"],
        "polished_but_socially_desirable_thin": ["generic language", "low evidence density"],
        "patchy_partially_answered_borderline": ["partial answers", "uneven granularity"],
        "mild_consistency_noise_not_fraud": ["mild emphasis shift", "repeated example"],
        "translated_thinking_english_or_oral_style": ["translated constructions", "oral-style syntax"],
    }

    noise_profile = list(noise_base[plan.intended_realism_slice])
    if plan.interview_quality == "clearer":
        noise_profile.append("interview clearer than letter")
    elif plan.interview_quality == "weaker":
        noise_profile.append("interview weaker than letter")

    return {
        "candidate_id": plan.candidate_id,
        "intended_realism_slice": plan.intended_realism_slice,
        "intended_ambiguity": plan.intended_ambiguity,
        "intended_primary_signals": signals_map[plan.intended_realism_slice][:3],
        "intended_primary_risks": risks_map[plan.intended_realism_slice][:3],
        "noise_profile": noise_profile,
        "generator_notes": (
            f"Messy realism case with bucket={plan.text_length_bucket}, interview={plan.has_interview}, "
            f"quality_shift={plan.interview_quality}."
        ),
    }


def find_near_duplicate_pairs(records: list[dict[str, object]], threshold: float = 0.92) -> list[list[object]]:
    packed: list[tuple[str, str]] = []
    for record in records:
        text_inputs = record["text_inputs"]
        chunks = [text_inputs.get("motivation_letter_text") or ""]
        for qa in text_inputs.get("motivation_questions") or []:
            chunks.append(qa.get("question") or "")
            chunks.append(qa.get("answer") or "")
        chunks.append(text_inputs.get("interview_text") or "")
        merged = re.sub(r"\s+", " ", " ".join(chunks).strip().lower())
        packed.append((record["candidate_id"], merged))

    pairs: list[list[object]] = []
    for i in range(len(packed)):
        left_id, left_text = packed[i]
        for j in range(i + 1, len(packed)):
            right_id, right_text = packed[j]
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= threshold:
                pairs.append([left_id, right_id, round(ratio, 4)])
    return pairs


def completion_rate_bands(records: list[dict[str, object]]) -> dict[str, int]:
    bands = {"exact_1_0": 0, "0_92_to_0_98": 0, "0_78_to_0_90": 0, "other": 0}
    for record in records:
        value = float(record["behavioral_signals"].get("completion_rate") or 0.0)
        if value == 1.0:
            bands["exact_1_0"] += 1
        elif 0.92 <= value <= 0.98:
            bands["0_92_to_0_98"] += 1
        elif 0.78 <= value <= 0.90:
            bands["0_78_to_0_90"] += 1
        else:
            bands["other"] += 1
    return bands


def reviewer_table_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        education = record["structured_data"]["education"]
        text_inputs = record["text_inputs"]
        behavioral = record["behavioral_signals"]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "english_proficiency_type": education["english_proficiency"]["type"],
                "english_proficiency_score": education["english_proficiency"]["score"],
                "school_cert_type": education["school_certificate"]["type"],
                "school_cert_score": education["school_certificate"]["score"],
                "motivation_letter_length": len(text_inputs.get("motivation_letter_text") or ""),
                "num_questions": len(text_inputs.get("motivation_questions") or []),
                "has_interview": int(bool((text_inputs.get("interview_text") or "").strip())),
                "completion_rate": behavioral.get("completion_rate"),
                "returned_to_edit": int(bool(behavioral.get("returned_to_edit"))),
                "skipped_optional": behavioral.get("skipped_optional_questions"),
            }
        )
    return rows


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260402)
    plans = build_candidate_plans()

    base_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    raw_records: list[dict[str, object]] = []
    sanitized_records: list[dict[str, object]] = []
    hidden_manifest_records: list[dict[str, object]] = []

    for idx, plan in enumerate(plans):
        submitted_at = (base_time + timedelta(hours=4 * idx)).isoformat().replace("+00:00", "Z")
        raw = build_raw_record(plan, submitted_at, rng)
        sanitized = {
            "candidate_id": raw["candidate_id"],
            "structured_data": raw["structured_data"],
            "text_inputs": raw["text_inputs"],
            "behavioral_signals": raw["behavioral_signals"],
        }
        CandidateInput.model_validate(sanitized)

        raw_records.append(raw)
        sanitized_records.append(sanitized)
        hidden_manifest_records.append(build_hidden_manifest(plan))

    candidate_ids = [record["candidate_id"] for record in raw_records]
    expected_ids = [f"syn_messy_v4_{idx:03d}" for idx in range(1, 41)]
    sanitized_ids = [record["candidate_id"] for record in sanitized_records]

    realism_counts = Counter(item["intended_realism_slice"] for item in hidden_manifest_records)
    ambiguity_counts = Counter(item["intended_ambiguity"] for item in hidden_manifest_records)

    text_length_counts = Counter(
        bucket_from_word_count(count_words(record["text_inputs"].get("motivation_letter_text") or ""))
        for record in raw_records
    )

    with_interview_count = sum(1 for record in raw_records if (record["text_inputs"].get("interview_text") or "").strip())
    without_interview_count = len(raw_records) - with_interview_count

    completion_bands = completion_rate_bands(raw_records)
    returned_to_edit_count = sum(1 for record in raw_records if record["behavioral_signals"].get("returned_to_edit"))
    skipped_distribution = Counter(
        int(record["behavioral_signals"].get("skipped_optional_questions") or 0) for record in raw_records
    )

    near_duplicate_pairs = find_near_duplicate_pairs(sanitized_records)

    sanitized_text = json.dumps(sanitized_records, ensure_ascii=False).lower()
    hidden_hint_terms = [
        "sincere_low_polish_real_signal",
        "polished_but_socially_desirable_thin",
        "patchy_partially_answered_borderline",
        "mild_consistency_noise_not_fraud",
        "translated_thinking_english_or_oral_style",
        "intended_realism_slice",
        "generator_notes",
        "noise_profile",
    ]
    hidden_hint_hits = [term for term in hidden_hint_terms if term in sanitized_text]

    leakage_terms = [
        "hidden_potential",
        "manual_review_required",
        "authenticity_review",
        "synthetic label",
        "archetype",
    ]
    leakage_hits = [term for term in leakage_terms if term in sanitized_text]

    raw_fields_ok = all(
        set(record.keys())
        == {"candidate_id", "structured_data", "text_inputs", "behavioral_signals", "metadata"}
        for record in raw_records
    )
    sanitized_fields_ok = all(
        set(record.keys()) == {"candidate_id", "structured_data", "text_inputs", "behavioral_signals"}
        for record in sanitized_records
    )

    interview_quality_counts = Counter(plan.interview_quality for plan in plans if plan.has_interview)
    awkward_sincere_cases = sum(
        1 for plan in plans if plan.intended_realism_slice == "sincere_low_polish_real_signal"
    )
    clear_signal_weak_presentation_cases = sum(
        1
        for plan in plans
        if plan.intended_realism_slice in {"sincere_low_polish_real_signal", "translated_thinking_english_or_oral_style"}
        and plan.english_score <= 75
    )

    validation_status = {
        "candidate_input_schema_raw": True,
        "candidate_input_schema_sanitized": True,
        "candidate_ids_exact_and_unique": candidate_ids == expected_ids and len(candidate_ids) == len(set(candidate_ids)),
        "raw_sanitized_one_to_one": candidate_ids == sanitized_ids,
        "reviewer_pack_no_metadata": all("metadata" not in record for record in sanitized_records),
        "reviewer_pack_no_hidden_hints": len(hidden_hint_hits) == 0,
        "no_near_duplicates": len(near_duplicate_pairs) == 0,
        "no_blatant_label_leakage": len(leakage_hits) == 0,
        "raw_shape_frozen_contract": raw_fields_ok,
        "sanitized_shape_correct": sanitized_fields_ok,
        "realism_mix_target_met": dict(realism_counts) == REALISM_TARGET,
        "ambiguity_mix_target_met": dict(ambiguity_counts) == AMBIGUITY_TARGET,
        "text_mix_target_met": dict(text_length_counts) == TEXT_TARGET,
        "interview_mix_target_met": {
            "with_interview": with_interview_count,
            "without_interview": without_interview_count,
        }
        == INTERVIEW_TARGET,
        "qualitative_self_audit": {
            "awkward_but_sincere_cases": awkward_sincere_cases >= 4,
            "overpolished_but_thin_cases": realism_counts["polished_but_socially_desirable_thin"] >= 4,
            "letter_worse_interview_clearer_cases": interview_quality_counts["clearer"] >= 6,
            "clear_signal_weak_presentation_cases": clear_signal_weak_presentation_cases >= 6,
            "committee_disagreement_prone_cases": ambiguity_counts["hard"] >= 10,
        },
    }

    failed_checks = []
    for key, value in validation_status.items():
        if isinstance(value, bool) and not value:
            failed_checks.append(key)
    if not all(validation_status["qualitative_self_audit"].values()):
        failed_checks.append("qualitative_self_audit")

    if failed_checks:
        raise ValueError(
            "Validation failed for messy_batch_v4: "
            + ", ".join(failed_checks)
            + f" | hidden_hint_hits={hidden_hint_hits}"
            + f" | leakage_hits={leakage_hits}"
            + f" | near_duplicate_pairs={near_duplicate_pairs}"
            + f" | realism_counts={dict(realism_counts)}"
            + f" | ambiguity_counts={dict(ambiguity_counts)}"
            + f" | text_counts={dict(text_length_counts)}"
            + f" | interviews={{'with': {with_interview_count}, 'without': {without_interview_count}}}"
        )

    notes = [
        "Batch V4 focuses on realistic messy admissions narratives with understandable language noise.",
        "Interview style is intentionally more conversational for most candidates, with selected reverse-gap cases.",
        "No hidden metadata is exposed in reviewer-visible packs.",
        "Realism noise includes uneven detail density, partial answers, repeated examples, and mild emphasis drift.",
        "Quality gate confirms schema validity, one-to-one pack alignment, and near-duplicate control.",
    ]

    summary = {
        "candidate_count": len(raw_records),
        "realism_slice_counts": dict(realism_counts),
        "ambiguity_counts": dict(ambiguity_counts),
        "text_length_counts": dict(text_length_counts),
        "with_interview_count": with_interview_count,
        "without_interview_count": without_interview_count,
        "completion_rate_bands": completion_bands,
        "returned_to_edit_count": returned_to_edit_count,
        "skipped_optional_questions_distribution": {
            str(key): skipped_distribution[key] for key in sorted(skipped_distribution)
        },
        "validation_status": validation_status,
        "notes": notes,
    }

    with RAW_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with GEN_MANIFEST_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in hidden_manifest_records:
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
        "pack_name": "messy_batch_v4",
        "batch_source": "messy_batch_v4_api_input.jsonl",
        "total_candidates": len(sanitized_records),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_version": "1.0",
        "fields_included": [
            "candidate_id",
            "structured_data",
            "text_inputs",
            "behavioral_signals",
        ],
        "fields_excluded": ["metadata"],
        "notes": "Sanitized for annotation. Hidden generation metadata remains only in raw hidden manifest.",
    }

    with PACK_MANIFEST_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(pack_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    completion_values = [float(record["behavioral_signals"].get("completion_rate") or 0.0) for record in raw_records]

    print(
        json.dumps(
            {
                "candidate_count": len(raw_records),
                "realism_slice_counts": dict(realism_counts),
                "ambiguity_counts": dict(ambiguity_counts),
                "text_length_counts": dict(text_length_counts),
                "with_interview_count": with_interview_count,
                "without_interview_count": without_interview_count,
                "completion_rate_min": round(min(completion_values), 3),
                "completion_rate_max": round(max(completion_values), 3),
                "completion_rate_avg": round(mean(completion_values), 3),
                "returned_to_edit_count": returned_to_edit_count,
                "validation_passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
