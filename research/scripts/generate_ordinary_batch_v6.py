from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "ml_workbench" / "raw" / "generated" / "ordinary_batch_v6"
PACK_DIR = ROOT / "data" / "ml_workbench" / "processed" / "annotation_packs" / "ordinary_batch_v6"

RAW_JSONL = RAW_DIR / "ordinary_batch_v6_api_input.jsonl"
GEN_MANIFEST_JSONL = RAW_DIR / "ordinary_batch_v6_generation_manifest.jsonl"
SUMMARY_JSON = RAW_DIR / "ordinary_batch_v6_summary.json"

PACK_JSONL = PACK_DIR / "ordinary_batch_v6_annotation_pack.jsonl"
PACK_JSON = PACK_DIR / "ordinary_batch_v6_annotation_pack.json"
PACK_TABLE_CSV = PACK_DIR / "ordinary_batch_v6_annotation_pack_table.csv"
PACK_MANIFEST_JSON = PACK_DIR / "ordinary_batch_v6_annotation_pack_manifest.json"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    intended_slice: str
    intended_ambiguity: str
    text_length: str
    english_type: str
    english_score: float
    school_type: str
    school_score: float
    motivation_letter_text: str
    motivation_questions: list[dict[str, str]]
    interview_text: str
    completion_rate: float
    returned_to_edit: bool
    skipped_optional_questions: int
    intended_primary_signals: list[str]
    intended_primary_risks: list[str]
    noise_profile: list[str]
    generator_notes: str


def clean(text: str) -> str:
    return dedent(text).strip()


def qa(question: str, answer: str) -> dict[str, str]:
    return {"question": question, "answer": answer}


SPECS: list[CandidateSpec] = [
    CandidateSpec(
        candidate_id="syn_ord_v6_001",
        intended_slice="ordinary_borderline_university_applicant",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school + self-study",
        english_score=79.0,
        school_type="City school diploma",
        school_score=84.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U because I want a place where I can explore social science and data at the same time. In school I enjoyed history and basic statistics, but I am still deciding which path should be first for me.

            Last year I helped my class teacher run peer study sessions before exams. I was not the main organizer of a big club, but I was steady and learned how different classmates need different explanations.

            I also noticed that students in my district often lose confidence when classes move from clear homework to open projects. I want to study how a learning environment can support that transition.

            At inVision U I hope to learn with people who are serious but not competitive in a negative way. I can contribute patience in group work, though I still need to improve how clearly I present my ideas in English.
            """
        ),
        motivation_questions=[
            qa(
                "Why do you think inVision U's learning environment fits you?",
                "I learn best when theory and practical assignments are mixed. The seminar and project style at inVision U feels right because I need structure and room to test ideas.",
            ),
            qa(
                "Tell us about a time you were unsure but still tried.",
                "I volunteered to lead one review session even though public speaking made me nervous. I prepared simple notes, asked for feedback after class, and did better the second time.",
            ),
            qa(
                "What could you contribute to your peers?",
                "I am patient when someone is stuck. I can sit with a classmate and break the task into smaller steps without making them feel bad.",
            ),
        ],
        interview_text=clean(
            """
            In conversation I am usually less polished than in writing, so I will be direct. I want a university where I can ask questions without feeling embarrassed. I am interested in education policy and maybe applied statistics, but I am still comparing both. At school I helped classmates revise before exams, and I liked that work because it was useful and practical. I am not claiming huge impact. I can offer steady effort, and I hope to grow into a more confident speaker.
            """
        ),
        completion_rate=0.89,
        returned_to_edit=True,
        skipped_optional_questions=1,
        intended_primary_signals=["steady effort", "peer support", "self-awareness"],
        intended_primary_risks=["limited standout evidence", "direction still forming"],
        noise_profile=["minor_awkward_phrasing", "moderate_specificity"],
        generator_notes="Balanced profile with realistic strengths and no clear storyline advantage.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_002",
        intended_slice="ordinary_borderline_university_applicant",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="language center",
        english_score=72.0,
        school_type="Standard school diploma",
        school_score=81.0,
        motivation_letter_text=clean(
            """
            I started thinking about university seriously during a class project on household budgeting. I enjoyed the part where we compared choices and trade offs, and that led me to read basic economics articles outside school.

            My grades are stable but not outstanding in every subject. Math is usually stronger for me than essay writing, and I still spend too much time trying to perfect one paragraph. I am working on that by drafting faster and revising later.

            I am applying to inVision U because I want a learning environment where students can discuss ideas openly and still be held to clear standards. I also like that students from different backgrounds work together in project courses.

            I can contribute reliability in team tasks. I am usually the person who keeps deadlines visible and checks if everyone understands the plan. What I need to improve is speaking up earlier when I disagree.
            """
        ),
        motivation_questions=[
            qa(
                "What do you want to study or explore first at university?",
                "I want to begin with economics and public systems classes, then decide if I should move toward policy analysis or business economics.",
            ),
            qa(
                "Describe a time you received difficult feedback and what you changed.",
                "A teacher said my arguments were too long and not focused. I started outlining one main claim before writing and my assignments became clearer.",
            ),
            qa(
                "How do you work with people who think differently from you?",
                "I ask them to explain their reason first and repeat it in my own words before I disagree. This reduces conflict in group work.",
            ),
            qa(
                "What support might you need in your first semester?",
                "I may need support with speaking confidence in seminars. I can prepare, but I still hesitate when I have to answer quickly.",
            ),
        ],
        interview_text="",
        completion_rate=0.83,
        returned_to_edit=False,
        skipped_optional_questions=2,
        intended_primary_signals=["consistent academics", "team reliability", "reflective improvement"],
        intended_primary_risks=["moderate communication hesitation", "limited initiative scope"],
        noise_profile=["plain_style", "modest_claim_density"],
        generator_notes="Typical applicant with believable upside and ordinary limitations.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_003",
        intended_slice="ordinary_borderline_university_applicant",
        intended_ambiguity="hard",
        text_length="long",
        english_type="private tutoring",
        english_score=86.0,
        school_type="Lyceum certificate",
        school_score=88.0,
        motivation_letter_text=clean(
            """
            When I first imagined university, I thought I needed to pick one exact profession before applying. Over the last two years I changed that view. I still care about planning my future, but I learned that early university is also for testing direction in an honest way.

            At school I enjoy psychology classes because they help me understand how people make decisions. At the same time I like practical technology lessons where we map a process and improve it step by step. Because of that mix, I am currently deciding between educational psychology and learning technology. I do not see this as confusion only. I see it as a question I can answer through structured study.

            I am not presenting myself as a student with dramatic achievements. Most of my work has been ordinary and local: helping our homeroom teacher coordinate peer revision hours, tracking attendance for a small after school reading group, and supporting two younger students who were often absent. None of this is large scale leadership, but it taught me consistency and communication.

            A local issue that matters to me is the gap between high performing students and those who silently fall behind. In my school the gap is not about intelligence only. It is often about confidence, study habits, and whether someone asks for help early. I want to study this problem with better methods than my current assumptions.

            inVision U appeals to me because of its collaborative learning structure and the chance to learn from peers with different strengths. I hope to contribute thoughtful teamwork and careful follow through. I also expect to need support in advanced academic writing during the first year, and I am ready to use that support.
            """
        ),
        motivation_questions=[
            qa(
                "What are you hoping to understand better during university?",
                "I want to understand why some students disengage even when they are capable, and what practical interventions actually help.",
            ),
            qa(
                "Tell us about something you started but did not finish.",
                "I tried to launch a digital note sharing page for our grade, but participation dropped after exam week. I learned I need a simpler structure and shared ownership.",
            ),
            qa(
                "How do you contribute in group settings?",
                "I usually coordinate timelines and make sure quieter people can speak. I am not the loudest member, but I keep teams moving.",
            ),
            qa(
                "What kind of support might help you at inVision U?",
                "Academic writing mentorship would help. I can analyze ideas, but I still need to improve formal structure and concise argumentation.",
            ),
        ],
        interview_text=clean(
            """
            In person I explain things in simpler words. I am deciding between educational psychology and learning technology, and I do not want to pretend that decision is final yet. The strongest thing I can offer now is steady work and willingness to learn from feedback. My school experience was mostly small scale, like peer revision hours and helping students who miss classes. I want to test these interests with stronger research methods at inVision U, not just intuition.
            """
        ),
        completion_rate=0.95,
        returned_to_edit=True,
        skipped_optional_questions=0,
        intended_primary_signals=["grounded curiosity", "consistent follow through", "peer minded contribution"],
        intended_primary_risks=["unclear specialization", "ordinary evidence level"],
        noise_profile=["moderate_repetition", "long_form_reflection"],
        generator_notes="Designed to trigger reasonable committee disagreement without dramatic factors.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_004",
        intended_slice="ordinary_borderline_university_applicant",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school classes",
        english_score=68.0,
        school_type="Regional school certificate",
        school_score=78.0,
        motivation_letter_text=clean(
            """
            I come from a school where most students focus on exam subjects and do not often join optional projects. I followed the same pattern for a while, then joined a small community mapping activity in grade 11. We interviewed local shop owners and families about transport and market access.

            My academic record is acceptable and steady. I do better in structured tasks than in open presentations. Sometimes my writing sounds too general, and I am trying to improve that by using concrete examples.

            I am applying to inVision U because I want a university setting where I can build stronger analytical skills and still stay connected to local community questions. I hope to explore economics, data literacy, and public systems courses before choosing a narrower track.

            I can contribute persistence and respectful teamwork. My weakness is that I hesitate before asking for clarification, which slows me down.
            """
        ),
        motivation_questions=[
            qa(
                "What local issue matters to you and why?",
                "Transport and market access matter to me because small design choices can affect students and families every day.",
            ),
            qa(
                "How do you balance responsibilities when workload increases?",
                "I make a simple weekly list and mark urgent items first. I still struggle when tasks are unstructured, but lists help me stay consistent.",
            ),
            qa(
                "What environment helps you learn best?",
                "I learn best when expectations are clear, and there is room to ask practical questions without being judged.",
            ),
        ],
        interview_text="",
        completion_rate=0.77,
        returned_to_edit=False,
        skipped_optional_questions=3,
        intended_primary_signals=["realistic self-assessment", "community observation", "steady academics"],
        intended_primary_risks=["modest initiative depth", "communication hesitation"],
        noise_profile=["low_detail_sections", "plain_tone"],
        generator_notes="Ordinary profile with mixed but plausible readiness signals.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_005",
        intended_slice="essay_weak_interview_stronger",
        intended_ambiguity="hard",
        text_length="short",
        english_type="school + apps",
        english_score=71.0,
        school_type="Standard school diploma",
        school_score=80.0,
        motivation_letter_text=clean(
            """
            I want to study at inVision U because I like biology and people. My writing is not very strong yet, so this letter is simple. In school I mostly did normal class work and sometimes helped my cousin with science homework. I am still deciding exact major, but I want a university where students help each other and learn by doing.
            """
        ),
        motivation_questions=[
            qa(
                "What do you want to explore in your first year?",
                "Biology and health related classes first, then maybe public health if I can handle the writing part better.",
            ),
            qa(
                "What support might help you succeed?",
                "I need writing support and maybe speaking practice, because I explain better when talking than when writing long text.",
            ),
        ],
        interview_text=clean(
            """
            In interview I can explain my motivation better. I became interested in health topics after helping my aunt at a local clinic registration desk on weekends. It was simple work, mostly organizing forms and guiding people to the right room, but I saw how communication problems create stress for families. I want to study in a university where I can combine science basics with practical community work. At school I am not top ranked, but teachers describe me as dependable. I know my essay style is weak, and I am ready to improve that.
            """
        ),
        completion_rate=0.74,
        returned_to_edit=False,
        skipped_optional_questions=2,
        intended_primary_signals=["clear oral motivation", "practical service exposure", "coachable attitude"],
        intended_primary_risks=["weak writing quality", "limited documented outcomes"],
        noise_profile=["short_essay", "interview_detail_recovery"],
        generator_notes="Essay under-expresses candidate; interview restores grounded signal.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_006",
        intended_slice="essay_weak_interview_stronger",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="language center",
        english_score=75.0,
        school_type="City school diploma",
        school_score=76.0,
        motivation_letter_text=clean(
            """
            I want to join inVision U and I think it can fit me because I need strict study and also active learning with others. In school, many times I did tasks late, but when project is real and people depend on me, I become more focused. I am interested in computer science and maybe education tools, not fully sure.

            This letter may sound mixed. Main point is I am serious to study and improve. I do not have olympiad medals. I have small things: I helped one teacher set up digital quiz files and explained to younger students how to submit homework online.

            At university I want to build stronger habits and learn with classmates who are different from me.
            """
        ),
        motivation_questions=[
            qa(
                "Describe a time you were unsure but still moved forward.",
                "I agreed to help with school quiz setup even though I had not used the platform before. I learned it step by step and asked for help when needed.",
            ),
            qa(
                "How do you work with people who have different styles?",
                "I try to split tasks by strengths first. If there is tension, I ask each person what they need to complete their part.",
            ),
            qa(
                "What kind of support could help your transition to university?",
                "Time management coaching would help. I can work hard, but I still underestimate how long assignments take.",
            ),
        ],
        interview_text=clean(
            """
            I speak more clearly than I write, so I want to explain one thing. I like technology because it can remove small barriers for students, for example when online homework systems are confusing. At my school I was the person teachers asked when younger students could not submit assignments. It was not advanced programming, but I enjoyed solving practical problems and teaching others patiently. At inVision U I want to build stronger technical foundations and also learn how to design tools that are actually usable.
            """
        ),
        completion_rate=0.86,
        returned_to_edit=True,
        skipped_optional_questions=1,
        intended_primary_signals=["stronger oral coherence", "practical problem solving", "peer support behavior"],
        intended_primary_risks=["essay organization weakness", "modest achievement record"],
        noise_profile=["essay_structure_noise", "informal_written_register"],
        generator_notes="Interview carries clarity that the written materials do not capture.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_007",
        intended_slice="essay_weak_interview_stronger",
        intended_ambiguity="hard",
        text_length="long",
        english_type="school classes",
        english_score=67.0,
        school_type="Lyceum certificate",
        school_score=85.0,
        motivation_letter_text=clean(
            """
            Writing this motivation letter is difficult for me because when I try to explain my plan, I often circle around the same point. I know this is not ideal, but I prefer to be honest about it. I am applying to inVision U because I want to study environmental systems and because I need a stronger learning environment than what I can build alone.

            In school I have average to good results, especially in science, but my essays are usually the weaker part. Teachers told me my thinking is better than my structure. I agree with this feedback. Sometimes I know what I want to say but I do not organize it in the best order. I am improving slowly by writing outlines, but I still repeat ideas.

            What I can show from experience is mostly small actions. I helped in a weekend clean up team near our neighborhood river and later joined a class project measuring simple water quality indicators. We did not produce big results, but it changed how I look at local environmental issues. I started paying attention to things I ignored before, like waste sorting and drainage problems after rain.

            Another point that matters to me is learning with peers who challenge me in a constructive way. I work better when people ask clear questions and expect clear answers. I think inVision U can give this kind of environment. I am not trying to present myself as exceptional. I am applying as a student who has solid interest, uneven communication, and real willingness to improve.

            If admitted, I want to build stronger writing and project planning habits in the first year. My long term direction is still open between environmental analysis and public planning, but I am ready to do the hard foundational work.
            """
        ),
        motivation_questions=[
            qa(
                "What local issue do you care about most right now?",
                "Water quality and waste flow in my area. I noticed people discuss it a lot, but practical monitoring and data sharing are limited.",
            ),
            qa(
                "Tell us about a project you started but could not fully finish.",
                "I started a school awareness mini campaign about river littering, but after exams our team stopped meeting. I learned I should have planned around exam periods.",
            ),
            qa(
                "How would you contribute to classmates?",
                "I can contribute consistency in field tasks and data collection. I am usually careful with repeated measurements and records.",
            ),
        ],
        interview_text=clean(
            """
            In speaking, I can explain my priorities better. I care about environmental systems because I saw local problems that look small but affect daily life, like drainage and waste around school routes. I joined a water quality class project and learned that I like practical measurement work. I know my writing sounds repetitive, but I am good at following through when a task is clear. At inVision U I want to strengthen academic writing while keeping a practical project focus.
            """
        ),
        completion_rate=0.91,
        returned_to_edit=True,
        skipped_optional_questions=1,
        intended_primary_signals=["oral clarity", "practical engagement", "follow through"],
        intended_primary_risks=["rambling writing style", "uncertain specialization"],
        noise_profile=["repetition_in_letter", "structure_drift"],
        generator_notes="Long essay intentionally uneven; interview gives a cleaner competence signal.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_008",
        intended_slice="essay_weak_interview_stronger",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school + self-study",
        english_score=73.0,
        school_type="Regional school certificate",
        school_score=79.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U because I want to study psychology, maybe with education topics also. I think university can give me better direction because now I am interested in many things and it is difficult to make one plan.

            At school I had normal results. I am not high award student. I help friends with class notes and sometimes help teacher organize materials. My weakness is that in writing I become too general, and this letter maybe shows that.

            I still believe inVision U is good fit for me because I need strong classes and good peer environment where people discuss ideas seriously.
            """
        ),
        motivation_questions=[
            qa(
                "How do you handle uncertainty in your plans?",
                "I try to keep one core direction, like psychology, and test options through small projects instead of deciding everything at once.",
            ),
            qa(
                "What support might you need?",
                "Academic writing support and regular advisor meetings would help me stay focused.",
            ),
            qa(
                "How would you work with classmates different from you?",
                "I listen first and ask what each person wants as an outcome. That helps avoid unnecessary arguments.",
            ),
        ],
        interview_text=clean(
            """
            If I explain orally, my motivation is simple. I want to study psychology because I am interested in how students lose or gain motivation. In school I often helped classmates prepare notes before tests, and I noticed confidence matters as much as content. I do not claim big impact, but I like this kind of work. I need support with writing structure, and I am ready for that. I think inVision U fits because it combines academic standards with collaborative learning.
            """
        ),
        completion_rate=0.82,
        returned_to_edit=False,
        skipped_optional_questions=2,
        intended_primary_signals=["interview groundedness", "peer assistance", "coachability"],
        intended_primary_risks=["weak essay precision", "broad focus"],
        noise_profile=["generic_written_phrases", "medium_evidence_density"],
        generator_notes="Writing underperforms relative to oral communication quality.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_009",
        intended_slice="essay_stronger_interview_weaker",
        intended_ambiguity="hard",
        text_length="short",
        english_type="private tutoring",
        english_score=88.0,
        school_type="City school diploma",
        school_score=82.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U to study sociology with a focus on youth transitions from school to university. In my final school year I led a small peer survey on study habits and presented findings to our class advisor. The work was modest but confirmed that I enjoy evidence based questions. I value the university's collaborative seminars and would contribute careful group preparation and respectful discussion.
            """
        ),
        motivation_questions=[
            qa(
                "What do you hope to study first?",
                "Foundational sociology and introductory research methods, especially courses where we learn to design and interpret small studies.",
            ),
            qa(
                "What did you learn from the peer survey you ran?",
                "I learned that students often know what to do but struggle with consistency and confidence, which made me interested in behavior and learning context.",
            ),
            qa(
                "How would you contribute to peers?",
                "I can contribute clear preparation, note sharing, and respectful discussion norms in group projects.",
            ),
        ],
        interview_text=clean(
            """
            I am sorry, I get nervous in interviews. I wrote my plan before, but now I cannot explain it well. I still think sociology is right for me. I do not know what else to add right now.
            """
        ),
        completion_rate=0.9,
        returned_to_edit=True,
        skipped_optional_questions=0,
        intended_primary_signals=["clear written intent", "structured thinking", "basic evidence use"],
        intended_primary_risks=["interview performance drop", "confidence fragility"],
        noise_profile=["speech_anxiety_signal", "short_interview"],
        generator_notes="Essay appears stronger than oral responses, creating committee uncertainty.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_010",
        intended_slice="essay_stronger_interview_weaker",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school + self-study",
        english_score=84.0,
        school_type="Lyceum certificate",
        school_score=87.0,
        motivation_letter_text=clean(
            """
            During the last two years I became interested in urban studies after noticing how different neighborhoods in my city offer very different opportunities for students. I started with simple observations and then read introductory articles on transport access, public space, and school commuting patterns.

            My school activities are not unusual, but they are consistent. I joined a student council logistics team, helped prepare discussion sessions, and collected feedback from younger students about after school study spaces. These experiences showed me that planning decisions can support or block learning.

            I am applying to inVision U because I want formal training that combines analytical methods with social context. I expect to begin with public policy, data literacy, and sociology courses, then choose a narrower focus.

            I can contribute careful organization and calm group communication. I still need to improve spontaneous speaking, especially when I am asked unexpected questions.
            """
        ),
        motivation_questions=[
            qa(
                "What local issue would you like to explore through study?",
                "I want to explore study space and transport access for students in districts that are far from central resources.",
            ),
            qa(
                "Tell us about difficult feedback and what you changed.",
                "A mentor told me my project notes were too descriptive and not analytical. I started adding comparison criteria and basic metrics.",
            ),
            qa(
                "How do you work with different people?",
                "I try to set common goals early and check assumptions openly so that everyone understands why tasks are assigned a certain way.",
            ),
        ],
        interview_text=clean(
            """
            I think my main answer is in my letter. I am interested in urban studies, yes. In interview I am less clear sometimes. I can work hard. I am not sure what extra detail to add now. Maybe I repeat what I wrote.
            """
        ),
        completion_rate=0.92,
        returned_to_edit=False,
        skipped_optional_questions=1,
        intended_primary_signals=["well formed essay", "applied observation", "stable academics"],
        intended_primary_risks=["thin oral elaboration", "interview hesitation"],
        noise_profile=["interview_repetition", "spoken_thinness"],
        generator_notes="Written case appears stronger than live communication signal.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_011",
        intended_slice="essay_stronger_interview_weaker",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="language center",
        english_score=81.0,
        school_type="City school diploma",
        school_score=90.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U to study mathematics in a setting where collaboration is taken seriously. I enjoy abstract problem solving, but I also value explaining methods to classmates, because teaching reveals gaps in my own understanding.

            Over the past year I co organized a small after school problem group for students preparing for final exams. The group was informal, but it improved my discipline and communication. I learned that clear examples and patient pacing often matter more than speed.

            At inVision U I want to build stronger foundations in pure and applied mathematics while exploring how quantitative skills can support social and educational questions. I am open to discovering a better direction through coursework.

            I can contribute persistence and thoughtful peer support. A weakness I am actively addressing is overthinking in high pressure conversations.
            """
        ),
        motivation_questions=[
            qa(
                "What do you hope university changes in how you think?",
                "I want to become more comfortable with ambiguity and develop the habit of testing assumptions instead of waiting for certainty.",
            ),
            qa(
                "How have you helped peers over time?",
                "I regularly helped classmates with exam preparation in math, especially by creating step by step solution guides.",
            ),
            qa(
                "What support might be useful for you?",
                "I might need support in oral presentation practice. I can explain ideas in writing well, but speaking under pressure is still difficult.",
            ),
        ],
        interview_text=clean(
            """
            Interview is harder for me than writing. I know I want math, but right now my answers are not coming out well. I usually think before speaking, and in interview I become too slow. I am trying to improve this.
            """
        ),
        completion_rate=0.88,
        returned_to_edit=True,
        skipped_optional_questions=1,
        intended_primary_signals=["strong written motivation", "peer tutoring evidence", "academic readiness"],
        intended_primary_risks=["oral confidence gap", "possible over controlled communication"],
        noise_profile=["short_hesitant_interview", "performance_anxiety"],
        generator_notes="Essay confidence contrasts with weaker live interview delivery.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_012",
        intended_slice="essay_stronger_interview_weaker",
        intended_ambiguity="hard",
        text_length="long",
        english_type="private tutoring",
        english_score=89.0,
        school_type="Lyceum certificate",
        school_score=93.0,
        motivation_letter_text=clean(
            """
            My interest in public health started from ordinary observations rather than one dramatic event. In my neighborhood, families often delay preventive care because basic information is confusing, schedules are hard to coordinate, and many people do not know where to ask practical questions. I began noticing these patterns when helping a relative with appointment logistics and forms.

            At school I tried to understand this issue more systematically. I joined a health awareness club, helped prepare short information sessions, and worked with classmates on a small survey about barriers to student health consultations. The scale was limited, but the process taught me to move from assumptions to evidence.

            Academically, I am strongest in biology and chemistry, with stable results in social science subjects. I also value writing and reflection because health questions are not only technical. They involve trust, communication, and culture. For that reason, I am interested in interdisciplinary study rather than a narrow track from day one.

            inVision U attracts me because it offers a learning environment where research practice, discussion, and community oriented projects can be combined. I want to build strong scientific foundations, but I also want to learn how to translate knowledge into understandable communication for non specialists.

            I can contribute consistency, careful preparation, and respect for different perspectives in group tasks. My limitation is that I can sound overly formal in writing and then less confident in unscripted speaking. I am aware of this gap and ready to work on it.

            In the longer term, I hope to contribute to preventive health initiatives that are realistic for local communities. I do not claim to have a full solution now. I am applying because I want rigorous education and the habit of testing ideas responsibly.
            """
        ),
        motivation_questions=[
            qa(
                "What local issue do you care about and how would study help?",
                "I care about preventive health communication gaps. Study can help me combine evidence based reasoning with community level implementation.",
            ),
            qa(
                "Tell us about a time you changed your approach after feedback.",
                "A mentor said our survey questions were leading participants. I rewrote them to be neutral, and the responses became more informative.",
            ),
            qa(
                "What kind of environment helps you learn best?",
                "An environment with clear expectations, research practice, and open discussion helps me learn best.",
            ),
            qa(
                "What support might you need at university?",
                "I may need coaching for spontaneous speaking because my prepared writing can be stronger than my live answers.",
            ),
        ],
        interview_text=clean(
            """
            I prepared this topic in writing, but in interview I feel less organized. I still want public health direction, yes. I am sorry if my answers are short. I can explain details better when I have time to structure them.
            """
        ),
        completion_rate=0.96,
        returned_to_edit=True,
        skipped_optional_questions=0,
        intended_primary_signals=["high quality essay evidence", "method awareness", "academic strength"],
        intended_primary_risks=["live communication weakness", "possible over polishing in writing"],
        noise_profile=["essay_interview_gap", "interview_underperformance"],
        generator_notes="Long polished writing contrasts with thin oral signal for ambiguity.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_013",
        intended_slice="support_needed_but_not_hidden_star",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school classes",
        english_score=69.0,
        school_type="Standard school diploma",
        school_score=83.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U because I want a structured learning environment where I can build confidence gradually. My school record is decent, especially in social studies, but I need more support in academic writing and speaking in English.

            I do not have a big leadership story. Most of my contribution has been practical and small scale: helping classmates prepare summaries, organizing shared notes, and checking in with peers who missed classes. These are simple actions, but they taught me responsibility.

            I am interested in education and community development topics. I want to understand how universities can support first year students who are motivated but unsure.

            What I can offer is steady effort, respectful teamwork, and willingness to ask for help when needed. I am not a hidden genius case. I am a student with potential that needs a supportive but demanding environment.
            """
        ),
        motivation_questions=[
            qa(
                "What support do you think will matter most in your first year?",
                "Writing mentorship and regular advisor check ins would matter most for me.",
            ),
            qa(
                "Tell us about a time you helped someone over a period of time.",
                "I helped two classmates with summary notes for several months when they had attendance gaps. It was not formal tutoring, but it helped them catch up.",
            ),
            qa(
                "How would you contribute to peers at inVision U?",
                "I can contribute consistency in group tasks and make sure people who are quieter still have a role in the discussion.",
            ),
        ],
        interview_text=clean(
            """
            I want to be honest about support needs. I can handle coursework, but I need guidance with academic writing style and confidence in speaking English during seminars. I am ready to do the work. I am not expecting special treatment, only clear expectations and feedback.
            """
        ),
        completion_rate=0.8,
        returned_to_edit=True,
        skipped_optional_questions=2,
        intended_primary_signals=["coachable mindset", "steady responsibility", "realistic self view"],
        intended_primary_risks=["transition support need", "limited high impact evidence"],
        noise_profile=["modest_self_presentation", "support_need_explicit"],
        generator_notes="Promising but clearly support-dependent profile without hidden-star framing.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_014",
        intended_slice="support_needed_but_not_hidden_star",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school + apps",
        english_score=65.0,
        school_type="Regional school certificate",
        school_score=79.0,
        motivation_letter_text=clean(
            """
            My motivation for inVision U is connected to growth, not only grades. I want to study business fundamentals and social project design, but I know I need support in formal writing and presentation.

            In school I usually complete tasks on time and keep acceptable results. My initiative examples are moderate: I helped organize one local donation drive and later supported planning for a small school event. These were useful, but not exceptional.

            I am applying because I believe inVision U can give me the structure, peer environment, and accountability that I need for the next stage. I learn best when expectations are clear and feedback is practical.

            I can contribute reliability and teamwork. I also know I need to build stronger confidence when speaking in front of unfamiliar groups.
            """
        ),
        motivation_questions=[
            qa(
                "What do you want professors to understand about you?",
                "I am motivated and responsible, but I improve faster when feedback is concrete rather than only general.",
            ),
            qa(
                "Describe a time you were unsure but still continued.",
                "I agreed to coordinate part of a school event even though I had not done it before. I asked older students for advice and completed my part.",
            ),
            qa(
                "What kind of peer environment helps you learn?",
                "A respectful environment where students share methods and not only final answers helps me learn best.",
            ),
        ],
        interview_text="",
        completion_rate=0.72,
        returned_to_edit=False,
        skipped_optional_questions=3,
        intended_primary_signals=["responsible baseline", "realistic goals", "willingness to improve"],
        intended_primary_risks=["confidence gap", "limited depth of initiative"],
        noise_profile=["modest_evidence", "non_dramatic_profile"],
        generator_notes="Kept intentionally ordinary and support-needing without hidden upside narrative.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_015",
        intended_slice="support_needed_but_not_hidden_star",
        intended_ambiguity="hard",
        text_length="medium",
        english_type="language center",
        english_score=74.0,
        school_type="City school diploma",
        school_score=82.0,
        motivation_letter_text=clean(
            """
            I want to study at inVision U because I am interested in psychology and peer learning, and I think a university community can help me grow in a balanced way. I am not applying as a top performer, but as someone who works steadily and improves with feedback.

            My school record is mostly consistent. I had one difficult term where my writing scores dropped, and I recovered after meeting teachers more regularly. That period showed me I need structure and accountability.

            I helped classmates prepare for two exam periods by making short revision sheets. This was simple work, but it taught me that small support can make a real difference.

            At inVision U I hope to strengthen academic writing, speaking confidence, and research basics. I can contribute patience and reliability in group tasks.
            """
        ),
        motivation_questions=[
            qa(
                "What changed after difficult feedback?",
                "A teacher said my writing was too vague. I began using a basic paragraph plan and asked for line by line comments.",
            ),
            qa(
                "What support do you expect to need?",
                "I expect to need writing center support and occasional coaching for seminar participation.",
            ),
            qa(
                "How do you contribute when a teammate is struggling?",
                "I usually help them split the task into smaller steps and check progress at agreed points.",
            ),
        ],
        interview_text=clean(
            """
            I think my profile is not dramatic, but it is honest. I can work hard and I usually follow through, but I do better when expectations are explicit. If I join inVision U, I will use support resources early instead of waiting until problems become big. I want to study psychology with practical learning applications, and I am open to refining that focus over time.
            """
        ),
        completion_rate=0.87,
        returned_to_edit=True,
        skipped_optional_questions=1,
        intended_primary_signals=["improvement after feedback", "steady effort", "peer support habit"],
        intended_primary_risks=["needs structured support", "moderate confidence"],
        noise_profile=["modest_claims", "support_dependency_signal"],
        generator_notes="Support-needed candidate with plausible promise but no hidden-star cues.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_016",
        intended_slice="support_needed_but_not_hidden_star",
        intended_ambiguity="hard",
        text_length="long",
        english_type="school + self-study",
        english_score=78.0,
        school_type="Lyceum certificate",
        school_score=86.0,
        motivation_letter_text=clean(
            """
            My decision to apply to inVision U comes from a realistic understanding of what I can already do and what I still need. I have stable grades and clear motivation, but I also know I will need support in academic writing, especially during the first semester.

            I am interested in social policy and education pathways because many students in my district are capable but uncertain about how to navigate university expectations. I started paying attention to this when I helped classmates with application timelines and document checklists. The work was ordinary, but it made me realize that process barriers can limit potential.

            I do not have one big achievement to present. My profile is made of smaller, repeated actions: helping with study plans before exams, sharing notes with students who missed classes, and coordinating basic logistics for class presentations. These tasks taught me planning, reliability, and patience.

            At the same time, I should be honest about weaknesses. When assignments are open ended, I sometimes spend too much time deciding how to start. I improved this by using templates and asking for early feedback. I still need practice in this area.

            I am applying to inVision U because I want an environment that is demanding but also supportive, where students are encouraged to ask for help and then take responsibility for progress. I can contribute careful teamwork and consistency. I expect to use mentoring and writing support from the beginning, not as a last step.

            My goal is not to present myself as exceptional. My goal is to become better prepared, more independent, and more useful to others through serious university study.
            """
        ),
        motivation_questions=[
            qa(
                "What support would be most useful in your first year?",
                "Writing center support and advisor check ins would be most useful, especially for open ended assignments.",
            ),
            qa(
                "Tell us about a time you adapted your approach after struggling.",
                "I started using simple assignment templates and seeking feedback earlier instead of rewriting everything at the end.",
            ),
            qa(
                "How do you contribute to people around you?",
                "I contribute practical organization, shared notes, and follow up messages to classmates who may be falling behind.",
            ),
            qa(
                "What kind of learning environment helps you most?",
                "A demanding environment with clear guidance helps me the most because I respond well to structured feedback loops.",
            ),
        ],
        interview_text=clean(
            """
            I am motivated, but I do not want to hide that I need support in writing style and confidence at the beginning. I have learned that asking for help early is more responsible than waiting. At school I was often the person who kept practical tasks moving, like schedules and shared notes, and I can bring that same reliability to inVision U. I am not a hidden genius case. I am a solid student who can improve meaningfully in the right environment.
            """
        ),
        completion_rate=0.93,
        returned_to_edit=True,
        skipped_optional_questions=0,
        intended_primary_signals=["realistic self assessment", "reliable contribution", "coachable growth trajectory"],
        intended_primary_risks=["high initial support need", "limited standout evidence"],
        noise_profile=["long_modest_reflection", "support_need_explicit"],
        generator_notes="Long form support-needed profile kept intentionally modest and non-cinematic.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_017",
        intended_slice="academically_ok_low_initiative_low_evidence",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school classes",
        english_score=76.0,
        school_type="City school diploma",
        school_score=85.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U because I want to continue my studies in a structured environment with clear academic standards. My school results are generally good, especially in core subjects.

            Outside class I have not done many major projects. I usually focus on assigned work and preparing for exams. I sometimes help classmates with notes, but I have not led any long term activity.

            I am interested in business and management topics and would like to build a strong foundation before deciding on a specific path. I value inVision U for its academic environment and peer learning opportunities.

            I can contribute reliability and consistent attendance. I need to improve initiative and independent project planning.
            """
        ),
        motivation_questions=[
            qa(
                "What do you want to explore academically in year one?",
                "I want to explore foundational business courses and basic data analysis, then decide a specific direction later.",
            ),
            qa(
                "Describe something you started but did not finish.",
                "I planned to join a school entrepreneurship club but stopped attending during exam preparation and did not return.",
            ),
            qa(
                "What would you like to improve about your study habits?",
                "I want to improve initiative in independent work and not wait for detailed instructions every time.",
            ),
        ],
        interview_text="",
        completion_rate=0.79,
        returned_to_edit=False,
        skipped_optional_questions=2,
        intended_primary_signals=["acceptable academics", "consistency", "self awareness of limits"],
        intended_primary_risks=["low initiative", "thin evidence of ownership"],
        noise_profile=["generic_content", "limited_examples"],
        generator_notes="Academically fine candidate with low initiative density by design.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_018",
        intended_slice="academically_ok_low_initiative_low_evidence",
        intended_ambiguity="borderline",
        text_length="short",
        english_type="private tutoring",
        english_score=82.0,
        school_type="Lyceum certificate",
        school_score=88.0,
        motivation_letter_text=clean(
            """
            I want to study at inVision U because I like economics and want a strong academic base. My grades are good and I complete school tasks responsibly. I have not done many activities outside required coursework. I hope university will help me become more active and independent in projects, not only in exam preparation.
            """
        ),
        motivation_questions=[
            qa(
                "How do you stay motivated when tasks are difficult or boring?",
                "I usually follow a schedule and finish the required task first. I am disciplined, but sometimes not very creative in approach.",
            ),
            qa(
                "What do you think you should improve before university?",
                "I should improve initiative and participation in optional activities.",
            ),
        ],
        interview_text=clean(
            """
            I can keep grades stable, but I know my profile is mostly academic and not very active outside class. I am hoping university will push me to take more initiative. Right now I usually do what is required and not much more.
            """
        ),
        completion_rate=0.7,
        returned_to_edit=False,
        skipped_optional_questions=3,
        intended_primary_signals=["academic discipline", "honest self report"],
        intended_primary_risks=["passive stance", "low initiative evidence"],
        noise_profile=["short_form", "low_evidence_density"],
        generator_notes="Compact profile intentionally low on ownership and extra initiative.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_019",
        intended_slice="academically_ok_low_initiative_low_evidence",
        intended_ambiguity="hard",
        text_length="medium",
        english_type="language center",
        english_score=73.0,
        school_type="Standard school diploma",
        school_score=81.0,
        motivation_letter_text=clean(
            """
            My reason for applying to inVision U is to continue education in a stable academic setting. I am interested in information systems, and my school results in math and informatics are acceptable.

            Most of my current evidence is from regular coursework. I have not built personal projects beyond school assignments. I usually wait for clear instructions before starting tasks, which is something I need to improve.

            I value inVision U because it seems organized and student focused. I think I can adapt to university expectations if I build stronger independent habits.

            I can contribute punctuality and cooperative behavior in group tasks. I need to grow in initiative and ownership.
            """
        ),
        motivation_questions=[
            qa(
                "What kind of environment helps you learn effectively?",
                "I learn best when instructions are clear and deadlines are visible. Open ended tasks are harder for me right now.",
            ),
            qa(
                "Tell us about difficult feedback and your response.",
                "A teacher said I rely too much on examples and do not propose my own approach. I now try to suggest one idea before asking for a model answer.",
            ),
            qa(
                "What might your peers rely on you for?",
                "They can rely on me for punctual delivery and basic coordination of deadlines.",
            ),
        ],
        interview_text="",
        completion_rate=0.84,
        returned_to_edit=False,
        skipped_optional_questions=2,
        intended_primary_signals=["academic baseline", "structured work habits"],
        intended_primary_risks=["limited initiative ownership", "thin extracurricular evidence"],
        noise_profile=["generic_wording", "conservative_claims"],
        generator_notes="Hard-to-sort ordinary case: capable but low ownership signal.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_020",
        intended_slice="academically_ok_low_initiative_low_evidence",
        intended_ambiguity="borderline",
        text_length="long",
        english_type="school + self-study",
        english_score=80.0,
        school_type="City school diploma",
        school_score=89.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U because I want to continue to a serious university program with clear academic standards and good peer environment. My school results are generally solid, and I usually complete tasks on time.

            If I describe my profile honestly, most of my strengths are related to consistency rather than initiative. I do assigned work, I prepare for exams, and I maintain stable performance. What is missing is stronger evidence of independent projects or self started activities.

            I had a few chances to do more. For example, I considered joining a student data project and a local volunteering group, but in both cases I prioritized exam preparation and did not continue. I cannot present these as achievements. They are examples of where my initiative level was lower than it could be.

            My current academic interests are in economics and analytics. I think I can succeed in these areas if I build better ownership habits and become more proactive outside mandatory coursework. I do not expect this to change automatically; I will need to make deliberate decisions about how I use my time.

            I value inVision U because it appears to combine discipline with collaboration. I believe that environment can push me to move beyond minimum requirements. I can contribute reliability and respectful teamwork from day one. At the same time, I need to improve in taking initiative, proposing ideas earlier, and following through on optional opportunities.

            My application is not built around dramatic stories. It is an ordinary profile with acceptable academics and clear room for growth in initiative.
            """
        ),
        motivation_questions=[
            qa(
                "What have you started but not finished, and why?",
                "I started exploring a student data project and then dropped it during exam season. I did not return after exams, which showed weak follow through.",
            ),
            qa(
                "How do you plan to improve initiative at university?",
                "I plan to choose one optional activity each semester and commit to it with defined milestones.",
            ),
            qa(
                "What could you contribute to your peers?",
                "I can contribute reliable execution, deadline discipline, and calm teamwork in structured tasks.",
            ),
            qa(
                "What support might help this growth?",
                "Periodic advisor accountability could help me maintain initiative goals, not only academic grades.",
            ),
        ],
        interview_text=clean(
            """
            I think my profile is straightforward. I can do academic work consistently, but I have not shown much independent initiative yet. I do not want to overstate this. At inVision U I would like to build that part of my profile with better habits and accountability.
            """
        ),
        completion_rate=0.9,
        returned_to_edit=True,
        skipped_optional_questions=1,
        intended_primary_signals=["solid academics", "reliability", "honest self diagnosis"],
        intended_primary_risks=["low initiative history", "limited ownership evidence"],
        noise_profile=["long_generic_sections", "evidence_thinness"],
        generator_notes="Long but still low-initiative profile to cover important borderline zone.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_021",
        intended_slice="values_aligned_but_vague_and_thin",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="school + apps",
        english_score=77.0,
        school_type="City school diploma",
        school_score=84.0,
        motivation_letter_text=clean(
            """
            I am motivated to join inVision U because I care about learning, respect, and positive contribution to community. I believe education should help students become thoughtful and responsible people.

            I want to study in an environment where people from different backgrounds can cooperate and learn from each other. I value teamwork, empathy, and open communication, and I try to bring these values in school interactions.

            At university I hope to develop strong knowledge and practical skills while also growing as a person. I am interested in social sciences and community related work, though my exact specialization is still open.

            I believe I can contribute a supportive attitude and willingness to collaborate. I am ready to work hard and learn continuously.
            """
        ),
        motivation_questions=[
            qa(
                "How would you contribute to classmates and campus life?",
                "I would contribute positive communication, respectful teamwork, and readiness to support peers when needed.",
            ),
            qa(
                "What kind of environment helps you learn best?",
                "A respectful and inclusive environment with clear expectations helps me learn best.",
            ),
            qa(
                "What do you want university to change in you?",
                "I want university to make me more mature, more independent, and better at cooperating with people who have different perspectives.",
            ),
        ],
        interview_text="",
        completion_rate=0.85,
        returned_to_edit=True,
        skipped_optional_questions=1,
        intended_primary_signals=["values alignment", "pro social orientation"],
        intended_primary_risks=["vague evidence", "unclear execution depth"],
        noise_profile=["socially_desirable_language", "thin_specificity"],
        generator_notes="Purposefully values-aligned but low on concrete grounding.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_022",
        intended_slice="values_aligned_but_vague_and_thin",
        intended_ambiguity="borderline",
        text_length="medium",
        english_type="language center",
        english_score=75.0,
        school_type="Standard school diploma",
        school_score=80.0,
        motivation_letter_text=clean(
            """
            My motivation for inVision U is based on the idea that universities should shape both knowledge and character. I respect collaborative learning and ethical responsibility, and I want to be part of a community with these principles.

            I am interested in studying management and social development topics. I care about teamwork and constructive dialogue, and I believe these are important for student life and future work.

            In school I have tried to stay responsible and cooperative. I may not have many remarkable examples, but I try to act with consistency and respect in group tasks.

            At inVision U I hope to develop stronger skills, wider perspective, and the ability to contribute positively to society.
            """
        ),
        motivation_questions=[
            qa(
                "What values are most important to you in a university community?",
                "Respect, collaboration, responsibility, and openness to different viewpoints are most important to me.",
            ),
            qa(
                "How do you handle disagreements in group work?",
                "I try to stay calm, listen carefully, and find a solution that protects both fairness and group progress.",
            ),
            qa(
                "What support might you need as a first year student?",
                "I might need guidance in planning longer assignments and adapting to university level workload.",
            ),
        ],
        interview_text="",
        completion_rate=0.78,
        returned_to_edit=False,
        skipped_optional_questions=2,
        intended_primary_signals=["normative fit language", "cooperative framing"],
        intended_primary_risks=["limited concrete examples", "generalized claims"],
        noise_profile=["value_heavy_text", "evidence_light"],
        generator_notes="Realistic nice-sounding profile with thin specificity.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_023",
        intended_slice="values_aligned_but_vague_and_thin",
        intended_ambiguity="hard",
        text_length="short",
        english_type="private tutoring",
        english_score=83.0,
        school_type="Lyceum certificate",
        school_score=86.0,
        motivation_letter_text=clean(
            """
            I want to join inVision U because I believe education should create responsible people who help others. I value teamwork, respect, and continuous growth. I am ready to study seriously and contribute to a positive learning culture. My goal is to develop knowledge, character, and practical skills that can be useful for my community.
            """
        ),
        motivation_questions=[
            qa(
                "How would you contribute to your peers?",
                "I would contribute a supportive attitude, respectful communication, and willingness to cooperate in shared tasks.",
            ),
            qa(
                "What do you want to learn about yourself at university?",
                "I want to learn how to become more independent, disciplined, and constructive in diverse teams.",
            ),
        ],
        interview_text="",
        completion_rate=0.67,
        returned_to_edit=False,
        skipped_optional_questions=4,
        intended_primary_signals=["pro social language", "motivation to grow"],
        intended_primary_risks=["very thin grounding", "low evidence density"],
        noise_profile=["short_generic_statement", "high_ambiguity"],
        generator_notes="Deliberately concise values language with minimal concrete proof.",
    ),
    CandidateSpec(
        candidate_id="syn_ord_v6_024",
        intended_slice="values_aligned_but_vague_and_thin",
        intended_ambiguity="hard",
        text_length="long",
        english_type="school + self-study",
        english_score=79.0,
        school_type="City school diploma",
        school_score=87.0,
        motivation_letter_text=clean(
            """
            I am applying to inVision U because I strongly believe higher education should combine academic development with values based growth. For me, university is not only about professional preparation. It is also about learning how to collaborate responsibly, communicate respectfully, and contribute to society with integrity.

            In school I tried to maintain a positive and cooperative attitude in group settings. I value inclusion, fairness, and mutual respect. I believe these principles matter in classrooms as much as in wider community life. My goal is to continue building these qualities while strengthening my academic foundation.

            I am interested in social sciences and management related fields because they involve working with people, understanding systems, and making decisions that affect others. At this stage, my focus is broad. I want to use the first period of university to refine my direction through courses and interaction with peers and mentors.

            What attracts me to inVision U is the learning culture that emphasizes collaboration, reflection, and development. I want to be part of a student community where people support each other and engage in meaningful dialogue. I think this type of environment helps students become not only skilled, but also responsible.

            I can contribute commitment, respectful communication, and readiness to participate in shared activities. I am prepared to learn continuously and improve through feedback. I know I still need to develop more practical examples of leadership and project ownership, and I see university as the right place to build that experience step by step.
            """
        ),
        motivation_questions=[
            qa(
                "Why does inVision U fit how you want to learn?",
                "I value environments that combine strong academics with collaborative and reflective learning, which is how I understand inVision U.",
            ),
            qa(
                "What do you hope to contribute to peers?",
                "I hope to contribute respectful communication, consistent participation, and a constructive mindset in team settings.",
            ),
            qa(
                "What is one area you need to strengthen?",
                "I need to strengthen concrete project ownership and move from general intentions to clearer execution evidence.",
            ),
            qa(
                "How do you respond when you receive difficult feedback?",
                "I try to reflect first, then turn feedback into specific next steps instead of defending myself.",
            ),
        ],
        interview_text=clean(
            """
            I can summarize my motivation simply. I care about academic growth and values based collaboration, and I want an environment where both are expected. I know my examples are not very strong yet in terms of large projects. I am hoping to build that at inVision U through steady participation and mentorship.
            """
        ),
        completion_rate=0.94,
        returned_to_edit=True,
        skipped_optional_questions=0,
        intended_primary_signals=["strong fit language", "reflective tone", "cooperative orientation"],
        intended_primary_risks=["vague supporting evidence", "limited concrete ownership"],
        noise_profile=["polished_but_thin", "values_forward_framing"],
        generator_notes="Long polished values profile with intentionally thin specifics.",
    ),
]


def letter_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))


def classify_length(text: str) -> str:
    words = letter_word_count(text)
    if words <= 90:
        return "short"
    if words <= 220:
        return "medium"
    return "long"


def build_records() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    base_time = datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc)
    raw_records: list[dict[str, object]] = []
    sanitized_records: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []

    for idx, spec in enumerate(SPECS):
        submitted_at = (base_time + timedelta(hours=6 * idx)).isoformat().replace("+00:00", "Z")
        raw = {
            "candidate_id": spec.candidate_id,
            "structured_data": {
                "education": {
                    "english_proficiency": {
                        "type": spec.english_type,
                        "score": spec.english_score,
                    },
                    "school_certificate": {
                        "type": spec.school_type,
                        "score": spec.school_score,
                    },
                }
            },
            "text_inputs": {
                "motivation_letter_text": spec.motivation_letter_text,
                "motivation_questions": spec.motivation_questions,
                "interview_text": spec.interview_text,
            },
            "behavioral_signals": {
                "completion_rate": spec.completion_rate,
                "returned_to_edit": spec.returned_to_edit,
                "skipped_optional_questions": spec.skipped_optional_questions,
            },
            "metadata": {
                "source": "ordinary_batch_v6",
                "submitted_at": submitted_at,
                "scoring_version": None,
            },
        }
        sanitized = {
            "candidate_id": raw["candidate_id"],
            "structured_data": raw["structured_data"],
            "text_inputs": raw["text_inputs"],
            "behavioral_signals": raw["behavioral_signals"],
        }

        raw_records.append(raw)
        sanitized_records.append(sanitized)
        manifest_records.append(
            {
                "candidate_id": spec.candidate_id,
                "intended_slice": spec.intended_slice,
                "intended_ambiguity": spec.intended_ambiguity,
                "intended_primary_signals": list(spec.intended_primary_signals),
                "intended_primary_risks": list(spec.intended_primary_risks),
                "noise_profile": list(spec.noise_profile),
                "generator_notes": spec.generator_notes,
            }
        )

    return raw_records, sanitized_records, manifest_records


def reviewer_table_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        education = ((record.get("structured_data") or {}).get("education") or {})
        english = education.get("english_proficiency") or {}
        school = education.get("school_certificate") or {}
        text_inputs = record.get("text_inputs") or {}
        behavioral = record.get("behavioral_signals") or {}

        letter = text_inputs.get("motivation_letter_text") or ""
        questions = text_inputs.get("motivation_questions") or []
        interview = text_inputs.get("interview_text") or ""

        rows.append(
            {
                "candidate_id": record.get("candidate_id"),
                "english_proficiency_type": english.get("type"),
                "english_proficiency_score": english.get("score"),
                "school_cert_type": school.get("type"),
                "school_cert_score": school.get("score"),
                "motivation_letter_length": len(letter),
                "num_questions": len(questions),
                "has_interview": int(bool(str(interview).strip())),
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
        parts: list[str] = [str(text_inputs.get("motivation_letter_text") or "")]
        for item in text_inputs.get("motivation_questions") or []:
            if isinstance(item, dict):
                parts.append(str(item.get("question") or ""))
                parts.append(str(item.get("answer") or ""))
        parts.append(str(text_inputs.get("interview_text") or ""))
        merged = re.sub(r"\s+", " ", " ".join(parts).strip().lower())
        assembled.append((str(record.get("candidate_id")), merged))

    pairs: list[list[str | float]] = []
    for i in range(len(assembled)):
        for j in range(i + 1, len(assembled)):
            left_id, left_text = assembled[i]
            right_id, right_text = assembled[j]
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= 0.92:
                pairs.append([left_id, right_id, round(ratio, 4)])
    return pairs


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    raw_records, sanitized_records, manifest_records = build_records()

    expected_ids = [f"syn_ord_v6_{idx:03d}" for idx in range(1, 25)]
    ids = [str(record.get("candidate_id")) for record in raw_records]
    raw_to_sanitized_ids = [str(record.get("candidate_id")) for record in sanitized_records]

    slice_counts: dict[str, int] = {}
    ambiguity_counts: dict[str, int] = {}
    text_length_counts = {"short": 0, "medium": 0, "long": 0}
    text_length_mismatches: list[str] = []

    for spec, record in zip(SPECS, raw_records):
        slice_counts[spec.intended_slice] = slice_counts.get(spec.intended_slice, 0) + 1
        ambiguity_counts[spec.intended_ambiguity] = ambiguity_counts.get(spec.intended_ambiguity, 0) + 1

        actual_length = classify_length((record.get("text_inputs") or {}).get("motivation_letter_text") or "")
        text_length_counts[actual_length] += 1
        if actual_length != spec.text_length:
            text_length_mismatches.append(f"{spec.candidate_id}:{spec.text_length}->{actual_length}")

    interview_with = sum(
        1
        for record in raw_records
        if str(((record.get("text_inputs") or {}).get("interview_text") or "")).strip()
    )
    interview_counts = {
        "with_interview": interview_with,
        "without_interview": len(raw_records) - interview_with,
    }

    completion_values = [
        float((record.get("behavioral_signals") or {}).get("completion_rate") or 0.0)
        for record in raw_records
    ]
    skipped_values = [
        int((record.get("behavioral_signals") or {}).get("skipped_optional_questions") or 0)
        for record in raw_records
    ]
    returned_to_edit_count = sum(
        1
        for record in raw_records
        if bool((record.get("behavioral_signals") or {}).get("returned_to_edit"))
    )

    near_duplicate_pairs = find_near_duplicates(sanitized_records)

    leakage_terms = [
        "ordinary_borderline_university_applicant",
        "essay_weak_interview_stronger",
        "essay_stronger_interview_weaker",
        "support_needed_but_not_hidden_star",
        "academically_ok_low_initiative_low_evidence",
        "values_aligned_but_vague_and_thin",
        "intended_slice",
        "generator_notes",
        "noise_profile",
    ]
    sanitized_payload_text = json.dumps(sanitized_records, ensure_ascii=False).lower()
    leakage_hits = [term for term in leakage_terms if term.lower() in sanitized_payload_text]

    raw_allowed_fields = {
        "candidate_id",
        "structured_data",
        "text_inputs",
        "behavioral_signals",
        "metadata",
    }
    sanitized_allowed_fields = {
        "candidate_id",
        "structured_data",
        "text_inputs",
        "behavioral_signals",
    }

    slice_targets = {
        "ordinary_borderline_university_applicant": 4,
        "essay_weak_interview_stronger": 4,
        "essay_stronger_interview_weaker": 4,
        "support_needed_but_not_hidden_star": 4,
        "academically_ok_low_initiative_low_evidence": 4,
        "values_aligned_but_vague_and_thin": 4,
    }

    validation_checks = {
        "all_ids_correct_and_in_order": ids == expected_ids,
        "all_ids_unique": len(ids) == len(set(ids)),
        "raw_sanitized_one_to_one": ids == raw_to_sanitized_ids,
        "raw_fields_match_required_schema": all(set(record.keys()) == raw_allowed_fields for record in raw_records),
        "sanitized_fields_only_allowed": all(
            set(record.keys()) == sanitized_allowed_fields for record in sanitized_records
        ),
        "sanitized_has_no_metadata": all("metadata" not in record for record in sanitized_records),
        "slice_counts_target_met": slice_counts == slice_targets,
        "text_length_target_met": text_length_counts == {"short": 4, "medium": 14, "long": 6},
        "interview_mix_target_met": interview_counts == {"with_interview": 16, "without_interview": 8},
        "text_length_tag_consistency": len(text_length_mismatches) == 0,
        "hidden_metadata_leakage_absent": len(leakage_hits) == 0,
        "no_near_duplicates": len(near_duplicate_pairs) == 0,
    }

    failed = [name for name, ok in validation_checks.items() if not ok]
    if failed:
        raise ValueError(
            "Validation failed for ordinary_batch_v6: "
            + ", ".join(failed)
            + f" | text_length_mismatches={text_length_mismatches}"
            + f" | leakage_hits={leakage_hits}"
            + f" | near_duplicate_pairs={near_duplicate_pairs}"
        )

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    summary = {
        "batch_name": "ordinary_batch_v6",
        "total_candidates": len(raw_records),
        "generated_at": now_iso,
        "distributions": {
            "slices": slice_counts,
            "ambiguity": ambiguity_counts,
            "text_lengths": text_length_counts,
            "interview_presence": interview_counts,
        },
        "behavioral_signal_stats": {
            "completion_rate_min": round(min(completion_values), 3),
            "completion_rate_max": round(max(completion_values), 3),
            "completion_rate_avg": round(mean(completion_values), 3),
            "returned_to_edit_count": returned_to_edit_count,
            "skipped_optional_min": min(skipped_values),
            "skipped_optional_max": max(skipped_values),
            "skipped_optional_avg": round(mean(skipped_values), 3),
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

    table_rows = reviewer_table_rows(sanitized_records)
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
        writer.writerows(table_rows)

    pack_manifest = {
        "pack_name": "ordinary_batch_v6",
        "batch_source": "ordinary_batch_v6_api_input.jsonl",
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


if __name__ == "__main__":
    main()
