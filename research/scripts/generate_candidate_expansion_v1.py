from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


QUESTION_BANK = {
    "q1": "Tell us about the most difficult period in your life - what happened, how did you cope, and what changed in you afterwards?",
    "q4": "Describe a project, idea, or solution you came up with entirely on your own initiative - not as an assignment. Where did the idea come from, and what was the outcome?",
    "q8": "Describe a situation, idea, book, person, or event that significantly shifted your worldview. What exactly changed in how you see things now?",
    "q11": "Tell us about a person or community you helped - not because you had to, but because you genuinely wanted to. What did that experience give you?",
    "q13": "Have you ever organized, launched, or promoted something - an event, an initiative, a small business, a club? What worked, what did not, and what would you do differently?",
    "q14": "How have you earned money or created something of value before applying to university? Walk us through the idea, the process, and the result.",
}


def paras(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def qlist(answer_map: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"id": qid, "question": QUESTION_BANK[qid], "answer": answer_map[qid].strip()}
        for qid in ["q1", "q4", "q8", "q11", "q13", "q14"]
    ]


def application_materials(
    *,
    documents: list[str] | None = None,
    attachments: list[str] | None = None,
    portfolio_links: list[str] | None = None,
    video_presentation_link: str | None = None,
) -> dict:
    return {
        "documents": documents or [],
        "attachments": attachments or [],
        "portfolio_links": portfolio_links or [],
        "video_presentation_link": video_presentation_link,
    }


def mk_candidate(
    *,
    candidate_id: str,
    source_type: str,
    derived_from_candidate_id: str | None,
    expansion_reason: str,
    language_profile: str,
    modality_profile: str,
    scenario_meta: dict,
    content_profile: dict,
    education: dict,
    motivation_letter_text: str,
    motivation_questions: dict[str, str],
    interview_text: str | None = None,
    video_interview_transcript_text: str | None = None,
    video_presentation_transcript_text: str | None = None,
    application_materials_payload: dict | None = None,
    behavioral_signals: dict | None = None,
    submitted_at: str = "2026-03-31T08:00:00Z",
) -> dict:
    structured_data = {"education": education}
    if application_materials_payload is not None:
        structured_data["application_materials"] = application_materials_payload

    text_inputs = {
        "motivation_letter_text": motivation_letter_text.strip(),
        "motivation_questions": qlist(motivation_questions),
    }
    if interview_text is not None:
        text_inputs["interview_text"] = interview_text.strip()
    if video_interview_transcript_text is not None:
        text_inputs["video_interview_transcript_text"] = video_interview_transcript_text.strip()
    if video_presentation_transcript_text is not None:
        text_inputs["video_presentation_transcript_text"] = video_presentation_transcript_text.strip()

    metadata = {
        "source": "synthetic_controlled_expansion_v1",
        "submitted_at": submitted_at,
        "scoring_version": "v1",
        "source_type": source_type,
        "derived_from_candidate_id": derived_from_candidate_id,
        "expansion_reason": expansion_reason,
        "language_profile": language_profile,
        "modality_profile": modality_profile,
    }

    return {
        "candidate_id": candidate_id,
        "scenario_meta": scenario_meta,
        "content_profile": content_profile,
        "structured_data": structured_data,
        "text_inputs": text_inputs,
        "behavioral_signals": behavioral_signals
        or {"completion_rate": 1.0, "returned_to_edit": False, "skipped_optional_questions": 0},
        "metadata": metadata,
    }


def build_candidates() -> list[dict]:
    cands: list[dict] = []

    cands.append(
        mk_candidate(
            candidate_id="cand_053",
            source_type="counterfactual",
            derived_from_candidate_id="cand_003",
            expansion_reason="same local ecology applicant with stronger self-presentation and multimodal proof",
            language_profile="en",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "high_potential_weak_self_presentation",
                "difficulty": "hard",
                "notes": "Grounded environmental tinkerer from Taraz with clearer self-advocacy, logs, and rough multimodal evidence.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Taraz",
                "primary_domain": "environmental science",
                "leadership_mode": "experimental, informal",
                "adversity_pattern": "limited local resources, family health concerns",
                "essay_style": "clearer and more assertive than peers from same motif",
                "impact_pattern": "neighborhood ecology and practical repair",
            },
            education={
                "english_proficiency": {"type": "Duolingo practice tests", "score": 112},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 86},
            },
            application_materials_payload=application_materials(
                documents=["canal_water_observation_log.pdf", "cleanup_attendance_sheet.pdf"],
                attachments=["filter_build_photos.zip"],
                portfolio_links=["https://example.org/field-notes/cand_053"],
                video_presentation_link="https://example.org/video/cand_053",
            ),
            motivation_letter_text=paras(
                "I am applying to inVision U because I have already learned that I work best when a problem is visible, local, and slightly messy. I grew up in Taraz near an irrigation canal that people complain about every year. Most adults talk about dirty water as something normal. I could not leave it alone, so I started taking small samples, comparing smell, color, and simple strip results, and writing them down in a cheap notebook. Later I borrowed a microscope from my biology teacher twice and realized that even a tiny observation habit changes how you think.",
                "My projects are still small. I built bottle filters, joined two weekend cleanups, and sometimes fixed broken chargers for neighbors to pay for gloves, bags, and test strips. But the important part is that I do not only collect information. I try something, see where it fails, adjust it, and then explain it to other people in plain language. Last autumn I convinced four friends to stop treating cleanup as a one-time photo event. We made a simple route and repeated it three Saturdays in a row. The canal did not become clean, but people on our street stopped saying teenagers never finish anything.",
                "I want inVision U because I need stronger tools, not a prettier story. I want to learn how to test environmental ideas properly, how to build low-cost community projects that continue after the first week, and how to speak with evidence even when I am younger than everyone in the room. I am not applying because I think I already solved something. I am applying because I know how I behave when a real problem appears: I start working, I keep records, and I pull others in.",
            ),
            motivation_questions={
                "q1": "The hardest period was when my younger sister's allergies got worse and my mother started blaming the damp flat, the canal smell, and the dust from the road all at once. We did not have money to move. I coped by doing what I could measure: cleaning mold, sealing cracks, airing the rooms, and reading about triggers instead of just worrying. It changed me because I stopped waiting for an adult with authority to explain everything.",
                "q4": "I built a simple filter from bottles, sand, charcoal, and cloth after comparing several online diagrams. The first version only made the water look cleaner, which taught me not to confuse appearance with quality. I rebuilt it three times and used it mainly as a learning tool at our cleanup days so younger kids could see what filtration can and cannot do.",
                "q8": "My worldview shifted when I realized that evidence can start small. Before that I thought science belonged to people in labs with expensive equipment. After I began logging canal observations and comparing results over time, I understood that disciplined local noticing is also a form of responsibility.",
                "q11": "I often help an elderly neighbor with groceries, but the more meaningful help was explaining the canal tests to children in our yard. They kept asking if the water was 'poison' or 'safe'. I showed them what the strips actually measure and why one test is not the full answer. That experience taught me that helping also means reducing confusion.",
                "q13": "I helped turn a one-day cleanup into a small repeated neighborhood routine. What worked was choosing the same short route each time and writing down who brought supplies. What did not work was expecting adults to join because they complained the loudest. Next time I would start with fewer promises and a clearer division of small tasks.",
                "q14": "I earned small money by repairing chargers, headphones, and one electric kettle for neighbors. The process was basic: diagnose with YouTube and trial and error, tell the person honestly if I was unsure, then try the repair. The result was not a business, but it paid for materials and showed me that practical skill becomes useful only when someone trusts you with a real object.",
            },
            interview_text=paras(
                "I speak better when I can point to something I made. If I only say 'I care about the environment', that sounds empty. But if I show my notebook or explain why one filter failed, then I feel more honest.",
                "I do not want a university where people only perform confidence. I want one where repeating a test, checking a result, or improving a rough design counts as real work.",
            ),
            video_presentation_transcript_text=paras(
                "Hi, I am recording this next to the canal, so sorry, there is wind. This notebook is what I used for almost a year. The first pages are messy because I did not know what mattered yet. Later I started writing date, weather, smell, and whether the water changed after rain.",
                "I also brought the bottle filter model. It looks childish, maybe, but it helped me explain to schoolchildren why cleaner-looking water is not automatically safe water. That difference matters.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_054",
            source_type="counterfactual",
            derived_from_candidate_id="cand_003",
            expansion_reason="same local ecology applicant with thinner evidence and no transcript support",
            language_profile="en",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "high_potential_weak_self_presentation",
                "difficulty": "hard",
                "notes": "Same environmental concern as cand_003 but with lighter proof and more scattered self-presentation.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Taraz",
                "primary_domain": "environmental science",
                "leadership_mode": "experimental, informal",
                "adversity_pattern": "limited materials, self-taught exploration",
                "essay_style": "awkward, diffuse, under-evidenced",
                "impact_pattern": "family and street level",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 79},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 83},
            },
            motivation_letter_text=paras(
                "I want to study at inVision U because I spend a lot of time thinking about why our canal water looks worse every year. I am not saying I have some big project. Mostly I read, test small things when I can, and ask questions that people around me do not really want to discuss. It makes me feel strange sometimes, because I cannot prove everything strongly yet.",
                "I did a few small experiments with water strips from a pharmacy and one bottle filter made from charcoal and sand. It was not scientific enough for real conclusions, but it made me more serious about learning. Also I joined cleanups with friends, but it was irregular. Sometimes people came, sometimes they forgot, so I cannot say it became a stable initiative.",
                "The reason I still apply is simple: when I notice a problem, I do not forget it fast. Even without good tools I keep coming back. I think that matters. I need stronger methods, better language, and people who do not laugh when an unfinished idea is still worth testing.",
            ),
            motivation_questions={
                "q1": "A difficult period was when I felt I cared about local pollution more than anyone around me and started doubting whether I was just inventing a problem. I coped by reading more and trying small tests instead of arguing all the time. It changed me because I learned that uncertainty is not the same as uselessness.",
                "q4": "I tried making a small bottle filter after watching a video and reading a guide. The outcome was mixed because the water looked better but I could not prove it was much safer. Still, the process made me want to learn proper testing instead of stopping at appearance.",
                "q8": "I changed after a biology teacher told me that being curious is only the first half and documenting is the second half. Before that I thought ideas alone were enough. Now I understand that if I do not keep records, I forget what actually happened.",
                "q11": "I helped a younger neighbor with a science fair poster about waste because she was nervous and did not know how to explain her topic. We drew a very basic diagram together. It showed me that people listen more when the idea becomes visible.",
                "q13": "I invited friends to two canal cleanup mornings. What worked was choosing a place everybody knew and keeping it short. What did not work was depending on motivation only. I did not bring enough bags once and I had no plan for repeating it.",
                "q14": "I sometimes repaired headphones or cables for classmates for very small money. The result was inconsistent because I am still learning. But even these tiny jobs made me feel that practical curiosity can become value for other people.",
            },
            behavioral_signals={"completion_rate": 0.96, "returned_to_edit": False, "skipped_optional_questions": 1},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_055",
            source_type="counterfactual",
            derived_from_candidate_id="cand_016",
            expansion_reason="same rural agriculture applicant with stronger self-presentation and clearer village-impact framing",
            language_profile="mixed",
            modality_profile="text_plus_interview",
            scenario_meta={
                "archetype": "high_potential_weak_self_presentation",
                "difficulty": "hard",
                "notes": "Rural doer still modest, but now explains adaptation and village-level ambition with more structure.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "near Saryagash, Turkistan region",
                "primary_domain": "agriculture / rural development",
                "leadership_mode": "quiet doer",
                "adversity_pattern": "rural constraints, limited equipment, family work pressure",
                "essay_style": "mixed-language but organized and grounded",
                "impact_pattern": "family farm and small school circle",
            },
            education={
                "english_proficiency": {"type": "school classes + self-study", "score": 81},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 84},
            },
            motivation_letter_text=paras(
                "Меня зовут Данияр, я из села рядом с Сарыагашем. I am applying not because village life is romantic, but because rural problems punish weak thinking very quickly. If irrigation fails, if soil is tired, if transport is late, the effect is visible the same week. This made me practical very early.",
                "On our family's tomato field I tested compost ratios and later a very cheap drip setup made from reused containers and tubing. The result was not magic, but two rows held moisture longer and my father stopped saying that online ideas are only for YouTube. At school I also started a small after-class group where we read about farming methods in other countries and tried to translate the useful parts into local conditions.",
                "I want inVision U because I need stronger scientific thinking, not distance from the village. I am ready to learn elsewhere, but my goal is to return with methods that can survive low budgets and uneven infrastructure. If I sound calm, it is because most of my work happened quietly. But quiet work can still move a system when it keeps going season after season.",
            ),
            motivation_questions={
                "q1": "The hardest period was one dry season when our family spent money on seeds and part of the crop still underperformed. I felt guilty because I kept suggesting experiments while my parents needed stability. I coped by making smaller trials instead of changing everything at once. After that, I learned that responsibility means designing experiments people can survive.",
                "q4": "I created a homemade compost and moisture comparison for different rows of tomatoes after noticing that we argued about fertilizer without any evidence. I marked the rows, wrote down watering, and compared plant condition every week. The outcome was modest but real: one method clearly kept the soil better during hot days.",
                "q8": "My worldview changed when I read about farmers in India using tiny, low-cost adaptations instead of waiting for large state programs. Before that I thought progress came mainly from big institutions. Now I think durable change often begins with small methods that ordinary people can repeat.",
                "q11": "I help my younger sister with homework almost every night, but the more important help was tutoring two village children in math when they started falling behind. Their parents were busy and embarrassed to ask teachers again. That taught me that support is often logistics plus patience, not inspiration alone.",
                "q13": "I started a school discussion group about farming ideas and local problems. What worked was linking each meeting to a real issue people knew, like water waste or storage loss. What did not work was holding it too late in the evening during harvest season. Next time I would combine shorter meetings with one small demonstration plot.",
                "q14": "I earned a small amount by tutoring local children in math and by helping neighbors sort produce for sale. The value I created was not only the money. I prepared worksheets using market examples and harvest calculations, which made the lessons more understandable and more trusted by parents.",
            },
            interview_text=paras(
                "I do not want to leave the village just to say I escaped it. I want to become useful enough that when I come back, people see method, not just ambition.",
                "My strongest habit is repeating a small experiment carefully. In farming, a sloppy trial gives you a comforting story but not a usable answer.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_056",
            source_type="counterfactual",
            derived_from_candidate_id="cand_016",
            expansion_reason="same rural agriculture applicant without interview but with rough video transcript and clearer evidence channel",
            language_profile="mixed",
            modality_profile="text_plus_video",
            scenario_meta={
                "archetype": "high_potential_weak_self_presentation",
                "difficulty": "hard",
                "notes": "Village candidate still understated, but video gives more direct evidence than the written self-presentation.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Turkistan region",
                "primary_domain": "agriculture / rural development",
                "leadership_mode": "quiet doer",
                "adversity_pattern": "farm chores, weak infrastructure, language limitations",
                "essay_style": "modest and uneven in text, more concrete in speech",
                "impact_pattern": "family and village level",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 74},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 82},
            },
            application_materials_payload=application_materials(
                attachments=["tomato_rows_before_after.jpg"],
                portfolio_links=[],
                video_presentation_link="https://example.org/video/cand_056",
            ),
            motivation_letter_text=paras(
                "Я подаюсь в inVision U, потому что мне важно понять, как превращать бытовой опыт сельской жизни в нормальные, проверяемые решения. У нас многие вещи делаются по привычке: так поливали всегда, так хранили всегда, так продавали всегда. Но когда сезон идет плохо, привычка уже не помогает.",
                "Last year I tested compost and a simple watering routine on a few rows instead of arguing with adults abstractly. It did not solve all our farm problems, but it showed me that even a small comparison changes family decisions. I also helped a teacher gather articles about drought-resistant practices for a school corner, because younger students mostly hear that agriculture is only hard work, not learning.",
                "I am not very polished in writing, and maybe that is visible. Still, I know how to observe patiently, keep going when results are slow, and work without applause. I want an education that respects those habits and sharpens them.",
            ),
            motivation_questions={
                "q1": "The hardest period was when my father got frustrated after a weak season and started saying experiments are for people who can afford mistakes. I understood his fear because our family budget is tight. I responded by making the next trial much smaller and writing down every step. That changed me because I learned to earn trust gradually.",
                "q4": "I designed a homemade drip experiment from reused containers and tubing because water loss in summer was obvious. The first setup leaked badly, so I adjusted the holes and distance. The result was only partial, but one section held moisture better and convinced my family to keep observing instead of dismissing the idea.",
                "q8": "My worldview shifted after I stopped seeing the village as only a place people leave. Once I compared our problems to examples from other countries, I understood that rural life can be a place of design and adaptation, not only shortage.",
                "q11": "I helped a younger cousin prepare for math exams because he wanted to quit trying after two bad tests. We used examples from crop counting and market prices so the numbers felt less abstract. That experience showed me that explanation works best when it comes from familiar life.",
                "q13": "I helped a teacher start a small shelf of printed materials about farming methods and invited classmates to discuss them. What worked was connecting each article to something our families already do. What did not work was expecting steady attendance during busy periods. Next time I would attach one practical task to each discussion.",
                "q14": "I earned small money by sorting produce for neighbors and tutoring two local children. The process was simple but disciplined: show up on time, finish even when tired, and explain clearly. That taught me that reliability is a form of value people remember.",
            },
            video_presentation_transcript_text=paras(
                "Ассалаумагалейкум. I am standing near our tomato rows. I wanted to show the place because in text it looks too neat. Here you can see the part where we tried different compost thickness and also this line, this one, where the water stayed longer after heat.",
                "I am not saying I invented something big. I am saying I stopped accepting 'we always do it like this' as a full answer. That is the main reason I want to study more seriously.",
            ),
            behavioral_signals={"completion_rate": 0.98, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_057",
            source_type="counterfactual",
            derived_from_candidate_id="cand_023",
            expansion_reason="same school leadership candidate with weaker self-presentation and flatter articulation of growth",
            language_profile="ru",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "strong_growth_trajectory",
                "difficulty": "medium",
                "notes": "Real growth remains visible, but the applicant underexplains outcomes and sounds less confident than the original.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Kostanay",
                "primary_domain": "STEM / school leadership",
                "leadership_mode": "peer motivator",
                "adversity_pattern": "academic setback and confidence loss",
                "essay_style": "honest but plain and slightly underpowered",
                "impact_pattern": "school-level peer support",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 77},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 87},
            },
            motivation_letter_text=paras(
                "Я хочу поступить в inVision U, потому что за последние два года сильно изменился, хотя мне до сих пор трудно об этом уверенно говорить. Раньше я был закрытым человеком и учился неровно. После нескольких провалов, особенно по математике, стало понятно, что если самому не менять привычки, то ничего не изменится.",
                "Постепенно я начал больше работать, помогать другим ребятам с задачами и предлагать небольшие идеи в школе. Например, мы сделали стену отзывов в библиотеке и позже я пытался помогать учителям с простыми сайтами. Это не были большие проекты, но они дали мне ощущение, что я могу приносить пользу не только себе.",
                "Мне нужен университет, где рост оценивают не только по прошлым ошибкам. Я не считаю себя готовым лидером, но уже знаю, что умею вытаскивать себя из слабой позиции и потом поддерживать других, когда они застревают в похожем состоянии.",
            ),
            motivation_questions={
                "q1": "Самый трудный период был после неудачного итогового теста по математике. Тогда мне казалось, что я просто слабый ученик и ничего не исправлю. Я справился не быстро: начал заниматься регулярно, спрашивать объяснения и перестал делать вид, что мне все равно. После этого у меня появилось больше внутренней ответственности.",
                "q4": "Я предложил сделать в школьной библиотеке стену отзывов о книгах, потому что заметил, что многие читают по обязанности и не обсуждают свои впечатления. Мы распечатали карточки и прикрепили их возле полок. Результат был скромный, но библиотека стала живее, и люди начали советовать книги друг другу.",
                "q8": "Мировоззрение поменялось после провала в учебе. До этого я думал, что если не мешать и ждать, все само наладится. Потом понял, что ожидание только закрепляет слабость. С тех пор я стал больше действовать сам, даже если шаг маленький.",
                "q11": "Я помогал девочке из параллели с математикой, потому что видел, как она теряет уверенность после каждой контрольной. Мы разбирали задачи после уроков и искали более простые объяснения. Мне это дало понимание, что чужой прогресс тоже может поддерживать тебя.",
                "q13": "Кроме стены отзывов, я пытался запустить маленький онлайн-клуб по программированию. Не получилось из-за плохого времени и слабого приглашения. Я понял, что идея сама по себе никого не собирает, если не продумать формат и ритм.",
                "q14": "Я делал простые сайты для учителей за небольшую оплату или бесплатно. Обычно это были страницы с материалами и контактами. Денег было мало, но я научился уточнять запрос и доводить работу до понятного результата, даже если заказчик сначала сам не знает, чего хочет.",
            },
            behavioral_signals={"completion_rate": 0.95, "returned_to_edit": True, "skipped_optional_questions": 1},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_058",
            source_type="counterfactual",
            derived_from_candidate_id="cand_023",
            expansion_reason="same growth-trajectory candidate with more grounded evidence, transcripts, and practical proof",
            language_profile="ru",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "strong_growth_trajectory",
                "difficulty": "medium",
                "notes": "Growth trajectory remains central, but candidate now provides clearer school and client-facing evidence.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Kostanay",
                "primary_domain": "STEM / school leadership",
                "leadership_mode": "peer motivator",
                "adversity_pattern": "past academic slump, self-recovery",
                "essay_style": "balanced, reflective, concrete",
                "impact_pattern": "school-level systems and peer tutoring",
            },
            education={
                "english_proficiency": {"type": "IELTS practice tests", "score": 6.0},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 90},
            },
            application_materials_payload=application_materials(
                documents=["teacher_site_examples.pdf", "library_feedback_wall_summary.pdf"],
                attachments=["peer_tutoring_schedule.png"],
                portfolio_links=["https://example.org/portfolio/cand_058"],
                video_presentation_link="https://example.org/video/cand_058",
            ),
            motivation_letter_text=paras(
                "Я подаюсь в inVision U не потому, что у меня всегда все получалось, а потому что мой главный ресурс появился именно после провалов. В девятом классе я почти перестал верить в себя как в ученика. Потом пришлось шаг за шагом восстанавливать дисциплину, спрашивать помощь, признавать пробелы и, что неожиданно, объяснять темы другим. Этот процесс оказался сильнее любого разового успеха.",
                "За последний год я сделал несколько вещей, которыми могу подтвердить рост. Мы с библиотекарем запустили стену отзывов и мини-викторину, чтобы чтение перестало быть формальностью. Я собрал для трех учителей простые сайты с учебными материалами и расписанием, а еще регулярно помогал ученице из параллели подтягивать математику. Это не звучит как большие титулы, но для меня это пример перехода от пассивности к полезности.",
                "Мне интересен inVision U, потому что там можно учиться через реальные задачи и обратную связь. Я хорошо понимаю, как выглядит человек до внутреннего сдвига, и как постепенно меняется его поведение после него. Я хочу дальше расти именно в среде, где изменения подтверждаются делом.",
            ),
            motivation_questions={
                "q1": "Тяжелее всего было пережить момент, когда я провалил важный тест и понял, что мое прежнее представление о себе уже не работает. Сначала я закрывался и избегал разговоров. Потом начал составлять простой план занятий и просить объяснять то, чего не понимаю. После этого у меня появилось больше смелости признавать слабые места заранее.",
                "q4": "Я предложил сделать в библиотеке стену отзывов и рекомендаций, потому что видел, как школьники проходят мимо книг без интереса. Мы сами распечатали карточки, придумали категории и потом собрали короткие реакции. В результате люди стали чаще брать книги по советам друг друга, а библиотекарь попросила продолжить формат.",
                "q8": "Больше всего мировоззрение изменил именно провал по математике, потому что он разрушил удобную иллюзию, будто прогресс происходит сам. Я понял, что рост начинается с неудобной честности. После этого мне стало легче уважать чужие усилия, а не только результаты.",
                "q11": "Я помогал девочке из параллели, у которой были сильные трудности с математикой и уверенность уже почти пропала. Мы занимались после уроков и делили сложные задачи на очень маленькие шаги. Этот опыт дал мне чувство, что поддержка может быть конкретной, а не просто ободряющей.",
                "q13": "Я организовывал стену отзывов и пробовал делать онлайн-клуб по программированию. Первое сработало, потому что было простым, видимым и не требовало долгих объяснений. Второе не сработало из-за слабой рекламы и неудобного времени. Теперь я лучше понимаю разницу между хорошей идеей и хорошим запуском.",
                "q14": "Я делал сайты для местных учителей, иногда бесплатно, иногда за небольшую оплату. Обычно сначала выяснялось, что человек сам плохо понимает, какие разделы ему нужны, поэтому я учился задавать вопросы и предлагать простую структуру. В результате получались рабочие страницы, которыми реально пользовались ученики и родители.",
            },
            interview_text=paras(
                "Если честно, моя история не про быстрый талант. Она про то, что после серии слабых результатов я перестал прятаться и начал перестраивать себя через небольшие регулярные действия.",
                "Сейчас мне уже легче вести других, но я всегда помню, как выглядит растерянность изнутри. Наверное, поэтому мне удается объяснять без высокомерия.",
            ),
            video_presentation_transcript_text=paras(
                "Здравствуйте. Я хотел показать не только себя, но и пару примеров работ. Вот, например, распечатка школьной стены отзывов и скриншоты сайта учителя физики. Это не сложная разработка, но раньше я вообще боялся брать ответственность за что-то видимое.",
                "Для меня рост - это когда ты можешь назвать свой прошлый провал, не стыдясь, и показать, что из него вышло что-то полезное для других.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_059",
            source_type="counterfactual",
            derived_from_candidate_id="cand_034",
            expansion_reason="same hands-on engineering applicant with weaker self-presentation and minimal supporting detail",
            language_profile="mixed",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "high_potential_weak_self_presentation",
                "difficulty": "hard",
                "notes": "Real technical inclination remains but examples are shorter and less well documented.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Karaganda",
                "primary_domain": "STEM / engineering",
                "leadership_mode": "hands-on, technical",
                "adversity_pattern": "test anxiety, low formal recognition",
                "essay_style": "brief, self-effacing, uneven",
                "impact_pattern": "family and neighbor repairs",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 73},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 85},
            },
            motivation_letter_text=paras(
                "My name is Samat and I am from Karaganda. I want to study at inVision U because I am better at fixing and building than at talking about it. In school I am not usually the student who looks strongest on paper. But when something breaks, I become focused and patient in a way that does not happen during exams.",
                "I have repaired small things for relatives and neighbors, made one power bank from old batteries, and helped with an old computer in our school library. None of this is very impressive if written shortly, and maybe that is part of my problem. I often move to the next task before I explain the previous one properly.",
                "Still, I know that practical work is how I think. I want to learn engineering in a place where building, troubleshooting, and improving rough prototypes counts as real intelligence, not as a side hobby for people who are not good at presentations.",
            ),
            motivation_questions={
                "q1": "A hard period for me was when I felt I was failing at school tests even though I knew I could solve real technical problems at home. I coped by giving myself small repair tasks and finishing them fully instead of only feeling frustrated. It changed me because I stopped using grades as the only way to judge my ability.",
                "q4": "I made a basic power bank for my younger brother from old phone batteries because he kept losing cheap ones and I wanted to try building something instead of buying again. It was not neat and I had to redo the wiring, but it worked in the end and taught me more than a chapter in a textbook.",
                "q8": "My worldview shifted after a summer electronics camp where people treated building as a normal language. Before that I thought practical skill was secondary to formal answers. After that I understood that solving a real problem is also a serious form of knowledge.",
                "q11": "I helped an elderly neighbor with faulty sockets because he was embarrassed to keep asking for help. I am not a certified electrician, so I stayed careful and only did what I understood. That experience made me realize trust is a responsibility, not just a compliment.",
                "q13": "I suggested repairing an old library computer with two classmates. What worked was that we combined spare parts and found an operating system that could run. What did not work was relying on equal motivation from everyone. I ended up doing more alone than I planned.",
                "q14": "I earned small money by fixing two old devices and reselling one laptop after cleaning it, replacing a drive, and reinstalling the system. The process was slow because I had to search for drivers and cheap parts. The result was useful, but I still need better organization and record-keeping.",
            },
            behavioral_signals={"completion_rate": 0.97, "returned_to_edit": False, "skipped_optional_questions": 1},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_060",
            source_type="counterfactual",
            derived_from_candidate_id="cand_034",
            expansion_reason="same engineering applicant with more grounded evidence, repair records, and multimodal proof",
            language_profile="mixed",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "high_potential_weak_self_presentation",
                "difficulty": "hard",
                "notes": "Technical repair candidate remains modest but provides enough multimodal proof to reduce ambiguity.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Karaganda",
                "primary_domain": "STEM / engineering",
                "leadership_mode": "hands-on, technical",
                "adversity_pattern": "nervous in formal evaluation, stronger in direct problem-solving",
                "essay_style": "grounded and concrete",
                "impact_pattern": "real technical outcomes for family, school, and neighbors",
            },
            education={
                "english_proficiency": {"type": "IELTS practice tests", "score": 5.5},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 87},
            },
            application_materials_payload=application_materials(
                documents=["repair_log_laptops.pdf", "library_pc_parts_list.pdf"],
                attachments=["powerbank_build_photos.zip", "soldering_notes.jpg"],
                portfolio_links=["https://example.org/repairs/cand_060"],
                video_presentation_link="https://example.org/video/cand_060",
            ),
            motivation_letter_text=paras(
                "My strongest motivation for inVision U is that I finally found a place where practical thinking is treated seriously. I have never been the smoothest speaker, and in exams I sometimes underperform. But if a device is dead, a cable is unstable, or an old computer looks beyond repair, I can stay with the problem for hours until it becomes understandable.",
                "I built a power bank from old phone batteries, repaired sockets for a neighbor under adult supervision, helped restore a school library computer, and revived two old laptops that I later sold cheaply. None of these projects were perfect. I burned one board while learning to solder and lost time on driver errors. But each failure made the next repair faster and more disciplined. I started keeping short records because otherwise I repeated the same mistake twice.",
                "I want to study where technical work is not hidden behind polished language. At the same time, I know I need stronger theory, safety habits, and project discipline. I am ready for that. My evidence is simple: if something real is broken, I move toward it.",
            ),
            motivation_questions={
                "q1": "The hardest period was when I felt trapped between average school results and the clear feeling that my real strengths were outside standard tests. I coped by taking on concrete repairs and writing down what I changed in each device. That gave me a different kind of confidence, built on repeatable results instead of mood.",
                "q4": "I built a power bank for my younger brother because he kept borrowing and breaking cheap ones. I collected batteries from old phones, learned basic safety rules, and rebuilt the casing after the first version overheated. The result was useful and also pushed me to stop improvising carelessly.",
                "q8": "A summer electronics camp changed my worldview because I met students who treated trial, error, and repair logs as normal intellectual work. Before that I thought practical students were always seen as secondary. After that I understood that making and fixing can be a serious path, not just a hobby.",
                "q11": "I repaired an old library computer at school because teachers had almost stopped trying to use it. That help mattered because students actually needed access to digital materials. It taught me that technical work becomes leadership when other people's routine starts depending on it.",
                "q13": "I organized a small three-person effort to repair the school library computer. What worked was dividing parts search, software installation, and cleaning. What did not work was assuming everybody would stay engaged once the easy part ended. Next time I would assign exact responsibilities from the first day.",
                "q14": "I found two old laptops at an electronics market, estimated what could be saved, and repaired them using spare RAM and new storage. After testing and reinstalling the system, I sold them for modest profit. The value was not only money. I learned pricing, risk, and the cost of weak documentation.",
            },
            interview_text=paras(
                "I am still not a naturally polished person. If someone asks me to sell myself in a beautiful way, I slow down. But if someone asks me what failed in the first power-bank version, I can answer very clearly.",
                "That is why I added logs and photos. They are closer to how I actually think than a perfect speech would be.",
            ),
            video_presentation_transcript_text=paras(
                "This is one of the laptops I repaired. The outside still looks old, but it boots and runs normally now. I wanted to show it because sometimes a project sounds bigger in writing than it really is. Here it is, just a real machine on my table.",
                "I also kept a page where I wrote which parts I changed and which mistake cost me extra time. I started doing that only recently, and it already improved my work a lot.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_061",
            source_type="counterfactual",
            derived_from_candidate_id="cand_045",
            expansion_reason="same socially active urban applicant with weaker self-presentation and thinner supporting detail",
            language_profile="ru",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "incomplete_but_promising",
                "difficulty": "medium",
                "notes": "Promising but uneven applicant whose social initiatives remain plausible while the written case becomes less developed.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Karaganda",
                "primary_domain": "community activity",
                "leadership_mode": "informal small-group leader",
                "adversity_pattern": "financial pressure, early work responsibility",
                "essay_style": "energetic but abbreviated",
                "impact_pattern": "local school and neighborhood initiatives",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 70},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 81},
            },
            motivation_letter_text=paras(
                "Я хочу учиться в inVision U, потому что мне близка идея реальных проектов, а не только экзаменов. Я рано начал подрабатывать и понял, что полезность нельзя откладывать до какого-то идеального будущего. Иногда это просто маленькая акция, помощь соседям или попытка собрать людей вокруг понятной задачи.",
                "У меня были такие вещи: сбор одежды в школе, мини-ярмарка для помощи приюту, подработка с младшими школьниками. Это не очень большие достижения, и иногда мне не хватало времени или хорошей организации. Но я понял, что мне легче включаться в действие, чем оставаться наблюдателем.",
                "Мне нужен университет, где этот импульс можно превратить в более устойчивые навыки. Сейчас у меня есть энергия и опыт маленьких запусков, но пока мало системы.",
            ),
            motivation_questions={
                "q1": "Сложнее всего было, когда отец потерял работу, а я начал совмещать учебу с подработками и постоянно чувствовал усталость. Я справлялся тем, что разбивал задачи на маленькие части и не ждал идеального настроения. После этого я стал меньше жаловаться и больше считать, что делать дальше.",
                "q4": "Я придумал школьную акцию по сбору хорошей, но ненужной одежды после того, как заметил, сколько вещей просто лежит без дела. Мы собрали часть вещей и передали их в соседний район. Это был полезный опыт, хотя по масштабу все осталось довольно маленьким.",
                "q8": "Мировоззрение поменялось после волонтерства и общения с людьми, которые попали в тяжелые обстоятельства. Я понял, что чужие проблемы обычно сложнее, чем выглядят со стороны. После этого мне стало важнее действовать без быстрого осуждения.",
                "q11": "Я помогал соседке-пенсионерке в огороде и с тяжелыми делами по дому. Это дало мне ощущение, что польза не обязана быть громкой. Иногда регулярная маленькая помощь ценнее красивых слов.",
                "q13": "Я участвовал в организации мини-ярмарки для помощи приюту животных. Работало то, что людям была понятна цель. Не сработало то, что мы поздно сделали рекламу и плохо распределили обязанности. Я понял, что доброй идеи мало без подготовки.",
                "q14": "Я зарабатывал, помогая младшим школьникам с домашними заданиями и иногда работая курьером. Деньги были небольшими, но это научило меня ответственности за время и обещания. Когда тебе платят даже символически, уже нельзя работать как попало.",
            },
            behavioral_signals={"completion_rate": 0.94, "returned_to_edit": False, "skipped_optional_questions": 1},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_062",
            source_type="counterfactual",
            derived_from_candidate_id="cand_045",
            expansion_reason="same socially active urban applicant with stronger grounded evidence and multimodal proof",
            language_profile="ru",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "incomplete_but_promising",
                "difficulty": "medium",
                "notes": "Peripheral urban applicant remains energetic, but now the record of concrete actions is better documented.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Karaganda",
                "primary_domain": "community activity",
                "leadership_mode": "informal small-group leader",
                "adversity_pattern": "family financial stress and work-study balance",
                "essay_style": "direct and more concrete",
                "impact_pattern": "school and neighborhood support initiatives",
            },
            education={
                "english_proficiency": {"type": "school classes + YouTube practice", "score": 76},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 83},
            },
            application_materials_payload=application_materials(
                documents=["clothing_drive_count_sheet.pdf", "animal_shelter_fair_budget.xlsx"],
                attachments=["fair_poster.jpg"],
                portfolio_links=["https://example.org/community/cand_062"],
                video_presentation_link="https://example.org/video/cand_062",
            ),
            motivation_letter_text=paras(
                "Я подаюсь в inVision U, потому что уже видел, как маленькая инициатива может менять поведение людей вокруг, если довести ее до конца. После финансовых проблем в семье я рано начал подрабатывать и, честно говоря, стал быстрее взрослеть. Это не сделало меня идеальным организатором, но научило не бояться неприятной работы и контакта с реальными ограничениями.",
                "За последний год у меня было несколько небольших, но понятных проектов: акция по сбору одежды, школьная ярмарка в пользу приюта, помощь младшим школьникам за символическую оплату. У каждой вещи был свой провал. Где-то мы поздно сделали рекламу, где-то недооценили сортировку вещей, где-то я слишком многое тянул сам. Но именно поэтому я начал вести простые списки, считать расходы и заранее распределять задачи.",
                "Мне нужен университет, где инициативность не превращают в абстрактный лозунг. Я хочу научиться строить проекты так, чтобы они были не разовым всплеском энергии, а устойчивой помощью. Опыт маленьких стартов у меня уже есть. Сейчас мне нужна система и окружение, которое требует качества.",
            ),
            motivation_questions={
                "q1": "Самым трудным был период после увольнения отца, когда мне пришлось совмещать учебу, вечернюю подработку и домашние обязанности. Я часто приходил домой выжатым и злился на всех подряд. Справился тем, что начал заранее планировать неделю и честно отказываться от того, что не вытяну. Это сделало меня менее хаотичным.",
                "q4": "Я придумал акцию 'День теплообмена' в школе, когда увидел, сколько хорошей одежды просто лежит без дела. Мы собрали вещи, отсортировали их и отвезли в соседний район через знакомых волонтеров. Результат был не огромный, но реальный, и я понял, как важно заранее продумать логистику, а не только идею.",
                "q8": "Мировоззрение сильно поменяло волонтерство и работа с людьми, которые оказались в трудной ситуации. Я перестал делить мир на 'старается' и 'сам виноват'. Теперь мне ближе подход, где сначала разбираешься в условиях человека, а потом уже предлагаешь помощь.",
                "q11": "Я помогал пенсионерке-соседке по огороду и дому, потому что видел, что ей реально тяжело. Это дало мне спокойное понимание пользы: не все важное выглядит как проект или событие. Иногда значимость держится на повторяемости.",
                "q13": "Я организовывал школьную ярмарку в помощь приюту животных. Сработало то, что цель была эмоционально понятной и люди охотно приносили выпечку и корм. Не сработало то, что мы поздно начали продвижение и недооценили поток людей. В следующий раз я бы раньше подключил классных руководителей и отдельного человека на учет денег.",
                "q14": "Я зарабатывал как курьер и помогал младшим школьникам с уроками за небольшую плату. Из этого я вынес не только деньги, но и привычку отвечать за срок и качество. Даже символическая оплата быстро убирает романтику и оставляет дисциплину.",
            },
            interview_text=paras(
                "У меня нет ощущения, что я уже умею 'руководить'. Но я точно умею входить в неприятную часть работы, когда нужно считать, носить, договариваться и не бросать после первого энтузиазма.",
                "Наверное, поэтому мне интересно учиться в среде, где проект оценивают по тому, что действительно произошло, а не по красивому началу.",
            ),
            video_presentation_transcript_text=paras(
                "Здесь у меня лист с примерным учетом одежды после школьной акции и фото с ярмарки. Я специально показываю это, потому что часто про инициативы говорят слишком общо. А мне важно, чтобы было видно: вот сколько вещей собрали, вот где ошиблись с сортировкой, вот где денег оказалось меньше, чем ожидали.",
                "Мне кажется, полезность начинается там, где ты не прячешь неудачные детали, а учитываешь их в следующем запуске.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_063",
            source_type="counterfactual",
            derived_from_candidate_id="cand_033",
            expansion_reason="same communication-oriented applicant but with more grounded evidence and smaller claims",
            language_profile="ru",
            modality_profile="text_plus_interview",
            scenario_meta={
                "archetype": "low_potential_strong_presentation",
                "difficulty": "medium",
                "notes": "Communication candidate still articulate, but now anchored in more believable, limited-scale actions.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Shymkent",
                "primary_domain": "communications",
                "leadership_mode": "small-group facilitator",
                "adversity_pattern": "self-doubt hidden behind polished communication",
                "essay_style": "polished but more bounded and specific",
                "impact_pattern": "school peer support, limited scale",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 82},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 85},
            },
            motivation_letter_text=paras(
                "Я хочу учиться в inVision U, потому что поняла: мне действительно близка работа с людьми, но одной уверенной речи для этого недостаточно. Раньше я слишком легко называла себя человеком, который вдохновляет и объединяет других. Потом стала внимательнее смотреть, что реально изменилось после моих инициатив, а что осталось просто хорошим настроением.",
                "Самый честный пример — небольшая встреча для девочек про уверенность и публичные выступления. Это не был большой форум, пришло мало людей, но мы разговаривали откровенно, делали простые упражнения и потом еще несколько участниц попросили меня помочь им с подготовкой к школьным выступлениям. Я также подрабатывала у репетитора и увидела, что реальная коммуникация требует терпения, повторений и конкретной пользы, а не только вдохновляющих слов.",
                "В inVision U мне интересно учиться именно потому, что там идеи можно проверять действием. Я не хочу больше опираться только на впечатление, которое произвожу. Я хочу строить более содержательные инициативы и честно видеть их пределы.",
            ),
            motivation_questions={
                "q1": "Трудным периодом было время, когда я внешне выглядела уверенной, а внутри чувствовала, что мои инициативы часто слишком поверхностны. Я справлялась тем, что стала записывать конкретные результаты после мероприятий и спрашивать у людей честную обратную связь. Это изменило меня, потому что я стала меньше путать красивую подачу с настоящим влиянием.",
                "q4": "Я сама придумала небольшую встречу для девочек о страхе выступлений, когда увидела, как подруга буквально теряет голос перед олимпиадой. Мы собрались после уроков, обсудили свои страхи и сделали пару простых упражнений. Итог был скромный, но несколько участниц потом стали просить у меня помощь с презентациями, и это для меня стало важнее красивого названия события.",
                "q8": "Мировоззрение изменилось после того, как я заметила разницу между вдохновением в моменте и устойчивой поддержкой. Раньше мне казалось, что если людям понравилось мероприятие, то этого достаточно. Теперь я думаю, что ценность есть только там, где меняется чье-то поведение или уверенность хотя бы на маленьком уровне.",
                "q11": "Я помогала младшим школьникам и одной пожилой соседке, но сильнее всего запомнилась работа с девочкой, которая боялась отвечать перед классом. Мы несколько раз репетировали речь в пустом кабинете, и потом она впервые выступила без слез. Это дало мне ощущение, что коммуникация может быть очень прикладной.",
                "q13": "Я организовала небольшую soft-skills встречу для девочек. Сработала теплая и безопасная атмосфера, потому что многие боялись говорить в большом кругу. Не сработал масштаб: я слишком слабо продумала приглашение и не подключила старших учеников. В следующий раз я бы делала серию коротких встреч, а не одну общую.",
                "q14": "Я подрабатывала помощницей у репетитора и помогала младшим школьникам с домашними заданиями. Процесс был несложный, но требовал терпения и способности объяснять разными словами. Из этого я вынесла, что доверие к тебе растет не от красивых формулировок, а от того, что ребенок действительно понял тему.",
            },
            interview_text=paras(
                "Мне кажется, раньше я слишком любила звучать убедительно. Сейчас для меня важнее другой вопрос: стало ли после моего участия кому-то реально легче, понятнее или спокойнее.",
                "Я не хочу потерять сильную речь. Я хочу, чтобы она наконец опиралась на настоящую работу.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_064",
            source_type="counterfactual",
            derived_from_candidate_id="cand_033",
            expansion_reason="same communication-oriented applicant with even more polished tone and lighter evidence as a controlled negative",
            language_profile="ru",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "low_potential_strong_presentation",
                "difficulty": "easy",
                "notes": "Deliberately polished and pleasant but still underpowered on specifics and durable outcomes.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Shymkent",
                "primary_domain": "communications",
                "leadership_mode": "symbolic involvement",
                "adversity_pattern": "limited concrete follow-through",
                "essay_style": "very polished, abstract, motivational",
                "impact_pattern": "claimed peer influence with weak proof",
            },
            education={
                "english_proficiency": {"type": "school classes + Olympiad prep", "score": 88},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 86},
            },
            motivation_letter_text=paras(
                "Для меня inVision U — это пространство, где энергия, ценности и лидерство встречаются в одном месте. Я всегда чувствовала, что моя сила заключается в умении вдохновлять людей, создавать атмосферу доверия и помогать каждому раскрывать внутренний потенциал. В современном мире особенно важно не просто знать, но и уметь объединять, слышать, направлять.",
                "В школе я не раз становилась инициатором теплых разговоров, поддерживающих встреч и маленьких событий, которые давали другим смелость поверить в себя. Я убеждена, что большие изменения начинаются с внутреннего импульса, и именно такой импульс я стараюсь нести в любую среду. Мне близки идеи эмпатии, совместного роста и мягкой силы.",
                "Я хочу учиться там, где можно превратить ценности в масштабное влияние. Верю, что мой голос, моя открытость и моя мотивация помогут мне стать человеком, который будет запускать важные общественные изменения.",
            ),
            motivation_questions={
                "q1": "Сложный период был тогда, когда я почувствовала, что не все вокруг разделяют мою открытость и готовность к диалогу. Это научило меня еще сильнее ценить внутреннюю устойчивость и силу слова. Я стала внимательнее к эмоциональному состоянию других людей.",
                "q4": "Я инициировала встречу для девочек о вере в себя и развитии soft skills. Идея пришла из желания создать безопасное пространство поддержки. Итогом стала очень теплая атмосфера, и я почувствовала, что такие форматы действительно нужны.",
                "q8": "На мое мировоззрение сильно повлияли книги о роли эмпатии и женского лидерства. После них я увидела, как важно не только говорить, но и быть опорой для других. Теперь я гораздо глубже чувствую ценность сообщества.",
                "q11": "Я помогала людям вокруг главным образом через внимание и поддержку. Иногда человеку важнее всего, чтобы его просто услышали. Это дало мне понимание, что присутствие и доброе слово тоже могут менять многое.",
                "q13": "Я проводила мотивирующие встречи и обсуждения. Работало то, что люди чувствовали себя принятыми. В следующий раз я бы хотела сделать подобные инициативы еще более масштабными и системными.",
                "q14": "Я помогала младшим школьникам и иногда ассистировала в учебном процессе, что тоже считаю созданием ценности. Для меня важнее не сумма, а то доверие, которое возникает между людьми, когда ты искренне вкладываешься.",
            },
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_065",
            source_type="counterfactual",
            derived_from_candidate_id="cand_037",
            expansion_reason="same grassroots organizer with less evidence density and fewer concrete outcomes",
            language_profile="en",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "academically_average_high_initiative",
                "difficulty": "medium",
                "notes": "Initiative remains plausible, but evidence channels are thinner and less outcome-rich than the original.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Taraz",
                "primary_domain": "community development",
                "leadership_mode": "grassroots organizer",
                "adversity_pattern": "limited local opportunities and average academics",
                "essay_style": "warm but somewhat generic",
                "impact_pattern": "neighborhood support, lightly documented",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 80},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 82},
            },
            motivation_letter_text=paras(
                "I am applying to inVision U because I learned that I naturally notice gaps in everyday community life and want to respond instead of only complaining. My grades are fine but not special. What gives me energy is organizing small things that help other people feel less stuck or less alone.",
                "I once started a small reading circle for children in my apartment complex and later helped with a neighborhood clothing-swap idea. These were not large projects and I did not keep strong records. Still, they showed me that I like practical community work and that even small coordination can change the mood of a place.",
                "I want a university where this instinct can become more disciplined. I am not applying with a dramatic success story. I am applying because when there is a missing piece in a local system, I usually try to fill it.",
            ),
            motivation_questions={
                "q1": "A difficult period for me was feeling average in school and not sure where I stood compared to more visibly successful students. I coped by putting my energy into small community activities where I could actually contribute. That changed me because I stopped seeing impact and grades as the same thing.",
                "q4": "I started a reading circle for younger children in my apartment complex during summer because they looked bored and many did not have anyone helping with books. We used donated books and met outside. The outcome was positive, though informal and not very well tracked.",
                "q8": "My worldview changed when I realized that local change does not need permission from a big organization. Watching older kids help younger kids at the reading circle made me see how small actions can multiply if people feel included.",
                "q11": "I help elderly neighbors with groceries or simple errands when needed. Those moments made me less impatient and more aware that care is often logistical and repetitive, not heroic.",
                "q13": "I helped organize a neighborhood clothing-swap event. What worked was posting in local chats and keeping the purpose simple. What did not work was doing everything quickly with too few helpers. I would plan volunteer roles earlier next time.",
                "q14": "I earned small money by tutoring younger students in literature and Russian after parents from the reading circle asked me. The process taught me how much preparation is needed even for basic lessons. It was modest income but real responsibility.",
            },
            behavioral_signals={"completion_rate": 0.95, "returned_to_edit": False, "skipped_optional_questions": 1},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_066",
            source_type="counterfactual",
            derived_from_candidate_id="cand_037",
            expansion_reason="same grassroots organizer with stronger evidence logs, transcripts, and multimodal support",
            language_profile="en",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "academically_average_high_initiative",
                "difficulty": "medium",
                "notes": "Grassroots organizer remains average academically but now shows repeat activity and practical record-keeping.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Taraz",
                "primary_domain": "community development",
                "leadership_mode": "grassroots organizer",
                "adversity_pattern": "few local opportunities, self-built structure",
                "essay_style": "grounded and steady",
                "impact_pattern": "neighborhood children and local mutual aid",
            },
            education={
                "english_proficiency": {"type": "school classes + online practice", "score": 88},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 84},
            },
            application_materials_payload=application_materials(
                documents=["reading_club_attendance_log.pdf", "clothing_swap_sorting_sheet.pdf"],
                attachments=["parent_group_messages.png"],
                portfolio_links=["https://example.org/community/cand_066"],
                video_presentation_link="https://example.org/video/cand_066",
            ),
            motivation_letter_text=paras(
                "I am applying to inVision U because I have already tested, on a small scale, what happens when you build useful structure where none existed before. I am not a top academic student, but I am someone who notices friction in daily life and starts organizing around it. In my neighborhood that usually meant younger children with no study support, families with extra clothing and no exchange system, or parents willing to help but not knowing how to coordinate.",
                "The reading club I started last summer became more serious than I first expected. We tracked attendance, collected old books, made a rotation of older helpers, and later some parents asked me to tutor their children. A clothing-swap event taught me the opposite lesson: good intentions create chaos if sorting, timing, and volunteer roles are weak. These experiences made me more interested in design, logistics, and community systems than in abstract leadership language.",
                "I want inVision U because I need harder training around project design and stronger peers who will question my assumptions. I already know I can start. Now I want to get better at scale, continuity, and honest evaluation.",
            ),
            motivation_questions={
                "q1": "The hardest period was realizing that many younger kids around me were quietly falling behind and I could not help everyone at once. I started feeling guilty and overcommitted. I coped by making a schedule and inviting older students to help instead of trying to become a one-person solution. That changed me because I began to think in systems, not only in personal effort.",
                "q4": "I started a reading club for children in my apartment complex because I saw them spending entire summer days outside with little structure. I asked parents for unused books, set a weekly rhythm, and created simple reading tasks. The outcome was modest but real: attendance stayed steady enough that some parents later requested tutoring.",
                "q8": "My worldview shifted when the reading club showed me that leadership is often infrastructure. Before that I imagined leadership as speaking well or being chosen by others. Now I think it also means arranging books, messages, timing, and follow-up so that participation becomes easy for people.",
                "q11": "I regularly help elderly neighbors with shopping and winter errands, but the most meaningful help was organizing older children to support younger readers. It gave me the feeling that a community becomes stronger when help is designed to continue without one central person.",
                "q13": "I organized the reading club and helped lead a clothing-swap event. The reading club worked because the rhythm was predictable and parents trusted the setup. The clothing swap partly failed because we rushed sorting and did not assign enough volunteers. I learned to respect the invisible work behind a simple event.",
                "q14": "I earned money by tutoring children who first came through the reading club. I made lesson plans based on their weak points and communicated progress to parents. The income was modest, but it taught me that once people trust your initiative, they also expect professionalism from it.",
            },
            interview_text=paras(
                "I used to describe myself as 'someone who likes helping'. That is true, but incomplete. What I actually like is building a small structure that makes help easier for many people, not just for one afternoon.",
                "I still make mistakes when I underestimate logistics. That is one reason I want a more demanding environment.",
            ),
            video_presentation_transcript_text=paras(
                "These are some of the books from our reading club, and this printed sheet is the attendance list. I wanted to include it because community work can sound soft and abstract unless you show the practical side.",
                "The clothing swap also taught me a lot. We did good work, but the first hour was messy because we had not separated sizes well. I keep that in mind now whenever I think something will be 'simple to organize'.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_067",
            source_type="counterfactual",
            derived_from_candidate_id="cand_040",
            expansion_reason="same polished communication applicant with stronger execution and bounded claims",
            language_profile="en",
            modality_profile="text_plus_interview",
            scenario_meta={
                "archetype": "low_potential_strong_presentation",
                "difficulty": "medium",
                "notes": "Communication-first candidate becomes more credible by adding repeated workshops and clearer limits.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Pavlodar",
                "primary_domain": "communications",
                "leadership_mode": "facilitator",
                "adversity_pattern": "strong speaking skills can outrun substance",
                "essay_style": "polished but now more evidence-based",
                "impact_pattern": "peer confidence-building at school scale",
            },
            education={
                "english_proficiency": {"type": "school classes + self-study", "score": 91},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 84},
            },
            motivation_letter_text=paras(
                "I am applying to inVision U because communication became meaningful to me only when I stopped treating it as inspiration and started treating it as facilitation. I speak easily, but for a while I overestimated what that meant. Recently I began running small peer workshops around presentation anxiety and discussion skills, and that experience forced me to think about format, repetition, and outcomes rather than pure enthusiasm.",
                "The most useful thing I organized was a short series of after-school practice sessions for students who were nervous about speaking in front of class. We met in a classroom, rehearsed introductions, timed answers, and gave each other simple feedback. Attendance was not huge, but the sessions repeated, which mattered more than one impressive event. I still like positive energy, but I trust it more when it is attached to structure.",
                "At inVision U I want to keep the strengths of communication while becoming stricter about evidence, design, and real usefulness. I do not want to be the person who can describe impact better than create it.",
            ),
            motivation_questions={
                "q1": "A difficult period for me was realizing that some people saw me as 'the motivational one' while I privately wondered whether I was mostly producing atmosphere. I dealt with that by running smaller, repeatable workshops and asking what participants actually found useful. It changed me because I became less attached to sounding impressive.",
                "q4": "I started a series of short speaking-practice sessions for classmates who got anxious before presentations. The idea came after one girl cried before her history talk. We practiced opening lines, pauses, and posture. The outcome was not dramatic, but several participants later told me they felt less panicked in class.",
                "q8": "Reading about active listening and then trying to facilitate real discussions changed my worldview. I understood that communication is not only expression. It is also designing a space where another person can think more clearly.",
                "q11": "I helped a classmate prepare for public speaking because she was avoiding classes with oral presentations. We practiced after school in small steps. It gave me the insight that confidence work becomes real only when someone returns the next day and tries again.",
                "q13": "I organized encouragement campaigns before, but the stronger project was the repeated workshop series. What worked was making the exercises concrete and short. What did not work was assuming everyone wanted emotional sharing; some students only wanted technique. I would separate those formats more clearly next time.",
                "q14": "I tutored younger students in English and literature, which helped me earn small money and test whether my communication was actually useful. I had to prepare differently for each student instead of relying on general warmth. That made the work more grounded.",
            },
            interview_text=paras(
                "The main correction I made in myself is this: I no longer think that being articulate is the same as being effective. It helps, but only if something keeps happening after the conversation ends.",
                "That is why the workshops matter to me. They were small, but they repeated, and repetition is where real responsibility begins.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_068",
            source_type="counterfactual",
            derived_from_candidate_id="cand_040",
            expansion_reason="same communication applicant without interview but with rough video modality and slightly stronger proof",
            language_profile="en",
            modality_profile="text_plus_video",
            scenario_meta={
                "archetype": "low_potential_strong_presentation",
                "difficulty": "medium",
                "notes": "Polished communicator still verbal, but a rough video sample adds a more realistic evidence channel.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Pavlodar",
                "primary_domain": "communications",
                "leadership_mode": "motivational facilitator",
                "adversity_pattern": "risk of overclaiming soft impact",
                "essay_style": "polished and upbeat",
                "impact_pattern": "small school workshops and peer coaching",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 89},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 83},
            },
            application_materials_payload=application_materials(
                attachments=["peer_workshop_outline.pdf"],
                video_presentation_link="https://example.org/video/cand_068",
            ),
            motivation_letter_text=paras(
                "I am applying to inVision U because communication is the area where I have already started to create small but visible value. In school I often became the person who encouraged others, but that description used to be too vague. More recently I tried to turn it into repeated action through short peer sessions on speaking, confidence, and discussion skills.",
                "What I learned is that good atmosphere is not enough. Some students needed emotional support; others needed concrete practice and feedback. Once I started separating those needs, the sessions became better. They were still small and far from perfect, but more real than one-time campaigns with nice slogans.",
                "I want to study in a place where I can keep the strengths of expression while becoming more rigorous about design, evidence, and practical outcomes. I am interested in the point where communication stops being decoration and starts becoming infrastructure.",
            ),
            motivation_questions={
                "q1": "The difficult period was when I felt that people liked my energy but I was not sure whether I was helping in a durable way. I coped by moving from one-off encouragement to repeated sessions with simple feedback forms. That changed me because I started valuing process over impression.",
                "q4": "I came up with a short workshop format for classmates who were afraid of speaking in public. We practiced answering a simple question in under one minute, then repeated it after feedback. The outcome was small but practical: some students told me they finally understood what exactly to improve.",
                "q8": "My worldview shifted after attending a youth forum and noticing how much stronger some sessions were simply because the facilitators listened well. Before that I thought impact in communication mostly came from charisma. Now I think structure and listening matter more.",
                "q11": "I helped one classmate prepare for a public speech because she was close to refusing the assignment. We practiced in an empty classroom three times. That taught me that confidence is often built through repetition in a low-pressure environment, not through one strong pep talk.",
                "q13": "I organized discussion circles and later a more practical peer-speaking session. What worked was keeping groups small and exercises simple. What did not work was relying on general invitations. I would choose clearer target groups next time.",
                "q14": "I earned small money by tutoring younger students in English and literature. The process forced me to move from broad motivational language to concrete explanations and preparation. That made my communication more honest.",
            },
            video_presentation_transcript_text=paras(
                "Hi, I am showing a page from the workshop outline because I wanted this video to be less abstract. The first part is just a breathing exercise and a thirty-second intro round. It sounds basic, but it actually helped the group settle.",
                "I still have a tendency to speak in big words, so I am trying to train myself to show the structure, the worksheet, the repeated exercise. That is a better test of whether the idea is real.",
            ),
            behavioral_signals={"completion_rate": 0.99, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_069",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill sustained builder/founder gap with repair-cooperative style candidate and multimodal proof",
            language_profile="mixed",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "sustained_builder_grounded",
                "difficulty": "hard",
                "notes": "Longer-horizon builder who turned recurring repair work into a small neighborhood system.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Aktobe",
                "primary_domain": "repair / local entrepreneurship",
                "leadership_mode": "builder-founder",
                "adversity_pattern": "financial limits, reuse culture, uneven adult trust",
                "essay_style": "grounded, practical, slightly rough",
                "impact_pattern": "neighborhood device repair and shared tools",
            },
            education={
                "english_proficiency": {"type": "self-study + school classes", "score": 78},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 80},
            },
            application_materials_payload=application_materials(
                documents=["repair_club_intake_log.pdf", "shared_tool_inventory.xlsx"],
                attachments=["before_after_repairs.zip"],
                portfolio_links=["https://example.org/repairs/cand_069"],
                video_presentation_link="https://example.org/video/cand_069",
            ),
            motivation_letter_text=paras(
                "I am applying to inVision U because I already know the kind of work that makes sense to me: build something useful, make it repeatable, and lower the cost for the next person. In Aktobe many families around us do not throw broken things away immediately because buying new ones is expensive. I started by repairing phones and kettles for neighbors, then realized that people also lacked tools, chargers, and someone who could explain what was worth fixing and what was not.",
                "During the last year I turned that habit into a small neighborhood repair club on Saturdays in my uncle's garage. It is not an official business and I am careful with anything dangerous, but we created a routine: people bring small devices, we inspect them, note whether repair is realistic, and sometimes show younger students basic tool safety. I also started a shared shelf of screwdrivers, solder, and spare cables because the same missing items slowed everyone down.",
                "I do not want university as an escape from practical life. I want it as a way to become more competent at systems, engineering, and low-cost product design. My evidence is not polished, but it is sustained. I kept coming back to the same local problem until a rough structure existed.",
            ),
            motivation_questions={
                "q1": "The hardest period was when I accidentally damaged a neighbor's radio while trying to repair it and felt I had lost people's trust. I responded by slowing down, refusing tasks beyond my current skill, and creating a simple intake sheet with risks written clearly. That changed me because I understood that a builder must also create safety and expectation, not only action.",
                "q4": "I created a Saturday repair club in my uncle's garage because individual repairs kept repeating the same chaos: no parts list, no tools, no record of what was tried. I set up a table, a notebook, and a basic sorting system for cables and screws. The outcome was a small but stable routine that made repairs faster and more teachable.",
                "q8": "My worldview shifted when I realized that entrepreneurship does not always begin with a big new product. Sometimes it begins with making broken everyday systems less wasteful. That idea changed how I look at local problems.",
                "q11": "I helped younger boys from our street learn safe tool habits because they kept opening devices carelessly after watching videos online. We practiced identifying screws, batteries, and risky parts first. It gave me patience and made me take responsibility for how skill is passed on.",
                "q13": "The repair club is the clearest thing I have organized. What worked was a fixed Saturday time, a shared tool shelf, and honest limits about what we could not repair. What did not work was storing parts without labeling them in the beginning. I now understand how quickly a useful space can become messy and unusable.",
                "q14": "I earned money by repairing small devices, reselling two restored phones, and charging symbolic fees when the fix required real time or purchased parts. The result was not high income, but it funded better tools and made the work more sustainable. More importantly, it taught me pricing, trust, and repeat customers.",
            },
            interview_text=paras(
                "What I enjoy most is when a system gets one step less fragile. A repaired kettle matters, but a notebook, a tool shelf, and a repeat process matter more because they make the next repair easier too.",
                "I think that is why I see myself less as a lone fixer and more as someone who likes building durable local infrastructure.",
            ),
            video_presentation_transcript_text=paras(
                "This is the garage table where we work on Saturdays. It is not fancy. The useful part is the organization: intake notebook here, common tools here, unsafe items on the side where younger students do not touch them.",
                "I wanted to show that because if I only say 'repair club', it can sound bigger than it is. In reality it is one table, repeated many times, and that repetition is exactly the point.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_070",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill sustained builder/founder gap with rural logistics and family-business process builder",
            language_profile="mixed",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "sustained_builder_grounded",
                "difficulty": "hard",
                "notes": "Rural candidate built lightweight order and delivery processes around family produce sales.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Kyzylorda region",
                "primary_domain": "small business / logistics",
                "leadership_mode": "builder-founder",
                "adversity_pattern": "family work pressure, transport unpredictability",
                "essay_style": "plain and operational",
                "impact_pattern": "family business and nearby households",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 71},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 79},
            },
            application_materials_payload=application_materials(
                documents=["weekly_order_sheet.pdf", "delivery_route_notes.pdf"],
                attachments=["whatsapp_order_screenshots.png"],
                portfolio_links=["https://example.org/processes/cand_070"],
                video_presentation_link="https://example.org/video/cand_070",
            ),
            motivation_letter_text=paras(
                "Меня интересует inVision U, потому что я уже вижу себя человеком, который умеет собирать разрозненные бытовые процессы в работающую систему. Моя семья выращивает овощи, и раньше продажи постоянно зависели от случайных звонков, путаницы с заказами и поздних доставок. Я не называл это предпринимательством, пока не понял, что фактически занимаюсь именно этим: упрощаю цепочку от поля до покупателя.",
                "Я сделал простую систему заказов через WhatsApp, разделил постоянных клиентов по дням, начал вести лист с остатками и маршрутом доставки. Это не технологический стартап и не что-то очень красивое, зато сразу повлияло на количество путаницы и возвратов. Позже я помог еще одной семье из соседнего аула использовать похожий способ учета для продажи зелени.",
                "Я хочу учиться там, где из такого опыта можно сделать более серьезное понимание логистики, операций и цифровых инструментов. Мне близок путь не через громкие идеи, а через системное уменьшение хаоса.",
            ),
            motivation_questions={
                "q1": "Самый трудный период был в сезон, когда у нас пропало несколько заказов из-за путаницы и семья сильно поссорилась на фоне усталости. Я понял, что хаос в коммуникации тоже стоит денег. Я справился тем, что сам начал записывать заявки и маршруты, даже если поначалу никто не воспринимал это серьезно. После этого я стал больше доверять простым системам.",
                "q4": "Я сам придумал разделять заказы по дням, адресам и типу продукции в одном листе, а не держать все в голове и в переписках. Идея пришла после нескольких ошибок с доставкой. В результате мы стали реже забывать клиентов и проще планировать поездки.",
                "q8": "Мировоззрение изменилось, когда я увидел, что 'бизнес' — это не только офис или приложение, а иногда обычная семья, которая учится считать, упаковывать и обещать вовремя. После этого мне стало интереснее изучать операции и процессы, а не только мечтать о чем-то абстрактном.",
                "q11": "Я помог соседней семье настроить похожий учет заказов, потому что у них повторялись те же ошибки, что раньше у нас. Это дало мне ощущение, что даже простая организационная идея может переноситься и приносить пользу другим.",
                "q13": "Фактически я запустил внутри семейной продажи новый порядок учета и доставки. Сработало то, что система была очень простой и все могли ей пользоваться. Не сработало то, что сначала я сделал слишком сложные пометки, и мама путалась. Я понял, что хорошая система должна быть удобной не только для ее автора.",
                "q14": "Я участвовал в продаже овощей, вел учет заказов и получал часть денег за доставку и сортировку. Ценность я создавал не только руками, но и организацией: меньше потерь, меньше недоразумений, больше повторных заказов от тех же клиентов.",
            },
            interview_text=paras(
                "Мне нравится видеть, как после одной таблицы или одного понятного правила у людей становится меньше нервов. Это очень простая вещь, но она меня по-настоящему мотивирует.",
                "Если честно, я не люблю громкие слова про лидерство. Мне ближе слово 'собрал' или 'наладил'.",
            ),
            video_presentation_transcript_text=paras(
                "Here are the paper sheets we used first, before I moved most orders to WhatsApp notes. I am showing them because the process started very manually. Nothing glamorous.",
                "But when we began grouping orders by delivery day, the arguments at home became fewer. For me that was the first proof that operations can change daily life, not only business profit.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_071",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill strong-evidence but weak-motivation gap with academically capable lab-oriented candidate",
            language_profile="en",
            modality_profile="text_plus_interview",
            scenario_meta={
                "archetype": "strong_evidence_weak_motivation",
                "difficulty": "hard",
                "notes": "Capable science student with concrete evidence but a noticeably thin, externally framed reason for applying.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Semey",
                "primary_domain": "chemistry / lab work",
                "leadership_mode": "individual contributor",
                "adversity_pattern": "narrow motivation, externally guided decisions",
                "essay_style": "competent but emotionally flat",
                "impact_pattern": "school laboratory and competition outputs",
            },
            education={
                "english_proficiency": {"type": "IELTS", "score": 6.5},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 93},
            },
            motivation_letter_text=paras(
                "I am applying to inVision U because I have strong preparation in chemistry and I think the university would give me wider opportunities than a standard local program. I have done well in school laboratory work, science competitions, and independent practice. My teachers suggested that I apply to institutions with a stronger project environment, and this is one reason I am interested.",
                "In the last two years I worked on a school project about water hardness testing, assisted a teacher during practical sessions for younger students, and placed well in regional science events. I also spent time learning safer lab procedures and documenting results more carefully after an early mistake with contaminated samples.",
                "My motivation is probably less expressive than some applicants. I am not applying because of one dramatic personal story. I am applying because I want access to stronger equipment, more demanding peers, and a path into serious scientific work.",
            ),
            motivation_questions={
                "q1": "The hardest period was when a project sample got contaminated because I rushed a preparation step and then had to admit the results were unusable. I coped by repeating the experiment from the beginning and tightening my process notes. It changed me because I became more careful and less attached to appearing efficient.",
                "q4": "I designed a school project to compare water hardness in samples from different neighborhoods after noticing that many families complained about scale in kettles. I built the procedure with help from a teacher, then adjusted it after the first measurements were inconsistent. The outcome was a stronger report and a clearer understanding of why method matters.",
                "q8": "My worldview changed after I realized how much scientific confidence depends on small procedural discipline. Before that I was more focused on getting the result quickly. Now I respect the process more than the appearance of being advanced.",
                "q11": "I helped younger students during school lab sessions because they were often nervous about chemicals and equipment. I showed them the setup step by step and checked their labels. That experience gave me patience, though I still prefer lab work itself over mentoring.",
                "q13": "I helped organize a chemistry club practical session and later a small demonstration for lower grades. What worked was preparing materials in advance and limiting the number of experiments. What did not work was trying to explain too much theory during the activity. I learned that a practical event needs tighter focus.",
                "q14": "I created value mainly through competition projects and lab assistance rather than paid work. Once I prepared simple tutoring sheets for a younger student in chemistry for a small fee. The money was minor; the main value was reliable technical help.",
            },
            interview_text=paras(
                "I know my motivation may sound plain. The honest answer is that I am applying because I want stronger scientific training and a better environment. I am still figuring out the broader social meaning of that path.",
                "What I can state with confidence is that I work seriously when the task is technical, repeatable, and demanding.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_072",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill strong-evidence but weak-motivation gap with vocational event-operations candidate",
            language_profile="ru",
            modality_profile="text_plus_interview",
            scenario_meta={
                "archetype": "strong_evidence_weak_motivation",
                "difficulty": "hard",
                "notes": "Execution-heavy stage and event operations candidate with concrete evidence but weak narrative motivation.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Ust-Kamenogorsk",
                "primary_domain": "event operations / technical stage work",
                "leadership_mode": "operations executor",
                "adversity_pattern": "limited reflection, externally motivated path",
                "essay_style": "dry and practical",
                "impact_pattern": "school and city event execution",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 68},
                "school_certificate": {"type": "college prep diploma", "score": 85},
            },
            motivation_letter_text=paras(
                "Я подаюсь в inVision U, потому что умею работать на мероприятиях и хочу перейти от школьного уровня к более серьезным проектам. У меня нет очень вдохновляющей истории про мечту с детства. Просто за последние два года я понял, что мне подходит техническая и организационная часть событий: сцена, свет, тайминг, сборка, контроль мелких сбоев.",
                "Я несколько раз отвечал за подготовку аппаратуры на школьных мероприятиях и помогал в городском молодежном центре, где мы собирали площадку, проверяли звук и вели технический план. Именно там я увидел, что мне нравится быть не на первом плане, а в точке, где от точности и спокойствия зависит весь ход события.",
                "Почему inVision U? Потому что это место, где практическая работа ценится не меньше, чем красивое выступление. Я пока не умею ярко объяснять свою мотивацию, но умею держать процесс в порядке и быстро исправлять сбои.",
            ),
            motivation_questions={
                "q1": "Труднее всего было на одном городском мероприятии, когда за час до начала возникли проблемы со звуком и часть команды начала паниковать. Я справлялся тем, что пошел по списку: кабели, питание, микшер, микрофоны. После этого я понял, что в стрессовых ситуациях мне помогает порядок, а не эмоции.",
                "q4": "Я сам предложил сделать для школьных концертов единый чек-лист подготовки оборудования, потому что каждый раз мы забывали разные мелочи. В итоге стало меньше хаоса перед началом и проще распределять обязанности.",
                "q8": "Мировоззрение изменилось, когда я впервые оказался за кулисами крупного события и увидел, что видимый успех держится на незаметной работе многих людей. После этого я стал иначе смотреть на роль исполнителя и операционного мышления.",
                "q11": "Я помогал младшим ребятам из школьного медиа-клуба разбираться с базовым оборудованием и таймингом, чтобы они не терялись перед мероприятием. Это дало мне понимание, что даже технические знания лучше закрепляются, когда ты объясняешь их кому-то другому.",
                "q13": "Я не столько организовывал события с нуля, сколько выстраивал их техническую часть. Сработало то, что я заранее готовил чек-листы и схемы. Не сработало — иногда я слишком поздно проверял часть оборудования, надеясь, что все как в прошлый раз. Теперь я меньше полагаюсь на память.",
                "q14": "Я получал небольшую оплату за помощь на местных мероприятиях: сборка, разборка, проверка техники. Деньги были не главными. Главным было то, что меня стали звать повторно, потому что я не пропадал и не создавал лишней паники.",
            },
            interview_text=paras(
                "У меня правда нет особенно красивой мотивационной истории. Я просто знаю, что на мероприятиях чувствую себя на своем месте, когда нужно собрать систему и удержать ее от сбоев.",
                "Если говорить совсем честно, я пока больше уверен в своей исполнительской ценности, чем в большой миссии. Но эта ценность у меня реальная.",
            ),
            behavioral_signals={"completion_rate": 0.99, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_073",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill non-STEM arts-first gap with documentary and oral-history execution plus multimodal proof",
            language_profile="mixed",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "arts_first_execution",
                "difficulty": "hard",
                "notes": "Arts-first candidate with grounded documentary practice and community archive work rather than formal STEM signaling.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Atyrau",
                "primary_domain": "arts / documentary audio",
                "leadership_mode": "project-based collaborator",
                "adversity_pattern": "limited arts infrastructure and equipment access",
                "essay_style": "reflective but concrete",
                "impact_pattern": "community memory and local storytelling",
            },
            education={
                "english_proficiency": {"type": "school classes + subtitles practice", "score": 75},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 84},
            },
            application_materials_payload=application_materials(
                documents=["oral_history_project_outline.pdf"],
                attachments=["audio_edit_waveforms.png", "photo_stills.zip"],
                portfolio_links=["https://example.org/archive/cand_073"],
                video_presentation_link="https://example.org/video/cand_073",
            ),
            motivation_letter_text=paras(
                "I am applying to inVision U because my strongest work has come not from competitions or grades, but from listening carefully and turning local memory into something others can access. In Atyrau, many older people around me carry stories about labor, migration, floods, and family life that younger people hear only in fragments. I became interested in recording these stories after helping a teacher with a school media project and realizing how quickly spoken memory disappears.",
                "Since then I have been learning basic audio recording and editing with limited equipment. I interviewed relatives, neighbors, and one retired river worker, then made short edited pieces with notes so that other students could understand the context. Later I ran a small after-school session where younger students learned how to ask respectful questions and label recordings properly. This is not traditional STEM work, but it requires discipline, ethics, and production skill.",
                "I want inVision U because I need a place where creative execution is taken seriously as a way of building knowledge. I am interested in documentary work, community archives, and the design of media that keeps ordinary people's experience from being lost or simplified.",
            ),
            motivation_questions={
                "q1": "The hardest period was when one interview with an elderly woman went badly because I asked questions too quickly and she shut down. I felt ashamed because I realized I had treated a person like a source. I coped by learning more about oral-history practice and by apologizing. It changed how I understand responsibility in creative work.",
                "q4": "I started a small oral-history recording project after hearing family stories disappear between generations. I borrowed a recorder, learned basic editing, and created short pieces with context notes. The outcome was a small archive that my school media teacher later used as an example for other students.",
                "q8": "My worldview changed when I understood that archives are not only for famous people. Ordinary voices also shape what a place remembers about itself. That idea made me see creative work as a form of civic responsibility.",
                "q11": "I helped younger students learn how to record interviews respectfully because they were excited about media but did not think about consent or labeling. That experience gave me patience and a stronger sense that method matters even in art.",
                "q13": "I organized a small after-school oral-history session and later coordinated two peers on a short documentary audio piece. What worked was giving everyone clear roles: interviewer, note-taker, editor. What did not work was underestimating how long cleaning audio would take. I learned that art projects also need operational planning.",
                "q14": "I created value by editing audio pieces for a local youth center and once received a small payment for assembling a simple event recap with captions. The income was minor, but it showed me that careful storytelling can have practical use outside school too.",
            },
            interview_text=paras(
                "People sometimes assume arts work is vague or purely expressive. For me it is almost the opposite. A good interview, a correctly labeled file, a fair edit decision - these are all disciplined actions.",
                "I want to grow as someone who can build trustworthy cultural records, not just 'creative content'.",
            ),
            video_presentation_transcript_text=paras(
                "I brought my recorder and a printout of the log sheet. Each file has the date, location, speaker, and one sentence about the topic. I started doing this after I lost track of an early interview and understood how fast material becomes unusable without structure.",
                "The clip behind me is from a river worker talking about flood seasons. I am not showing the whole thing, only enough to explain the method and why these voices matter.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_074",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill vocational execution gap with electrician-maintenance candidate and rough video proof",
            language_profile="ru",
            modality_profile="text_plus_video",
            scenario_meta={
                "archetype": "vocational_execution_grounded",
                "difficulty": "hard",
                "notes": "Vocationally oriented maintenance candidate with practical evidence and limited polish.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Ekibastuz",
                "primary_domain": "electrical maintenance",
                "leadership_mode": "reliable executor",
                "adversity_pattern": "family work pressure, safety responsibility",
                "essay_style": "plain and practical",
                "impact_pattern": "household and small-shop maintenance",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 60},
                "school_certificate": {"type": "vocational college entry diploma", "score": 78},
            },
            application_materials_payload=application_materials(
                attachments=["maintenance_checklist_photo.jpg"],
                video_presentation_link="https://example.org/video/cand_074",
            ),
            motivation_letter_text=paras(
                "Я хочу учиться в inVision U, потому что лучше всего понимаю мир через работу руками, схемами и ответственностью за безопасность. В Экибастузе у многих знакомых мелкие бытовые проблемы с проводкой, розетками, освещением, и я рано начал помогать отцу и мастерам на подработках. Это не звучит как очень 'академичная' история, но именно там я понял цену аккуратности и последовательности.",
                "Сейчас я умею делать базовую диагностику, составлять простой список материалов, менять часть бытовых элементов и проверять, что после ремонта все работает стабильно. Я стараюсь не лезть туда, где нет достаточной компетенции, и считаю это не слабостью, а нормальной профессиональной границей. Меня привлекает обучение, где практическая работа не считается чем-то второстепенным.",
                "Если мне дать задачу, инструкцию и ответственность, я не пугаюсь. Я хочу расти из надежного исполнителя в человека, который понимает системы глубже и может организовывать более сложные процессы, не теряя осторожности.",
            ),
            motivation_questions={
                "q1": "Самым трудным был случай, когда после одного ремонта свет снова начал мигать, и я очень испугался, что пропустил что-то опасное. Я вернулся, заново проверил все соединения вместе со старшим мастером и понял, где именно ошибся. После этого я стал гораздо внимательнее относиться к двойной проверке и не стесняться просить контроль.",
                "q4": "Я сам придумал делать маленький чек-лист перед любым бытовым ремонтом, потому что видел, как в спешке забываются очевидные вещи. Благодаря этому стало меньше суеты и повторных выездов.",
                "q8": "Мировоззрение поменялось, когда я понял, что надежность — это не скучная черта, а реальная ценность. До этого казалось, что главное — быстро уметь много. Сейчас я думаю наоборот: важнее всего делать то, за что можешь отвечать.",
                "q11": "Я помогал пожилым соседям и маленькому магазину рядом с домом с простыми проблемами по свету и проводке, потому что им было сложно быстро найти мастера на мелкий заказ. Это дало мне уважение к незаметной работе, от которой зависит чужой обычный день.",
                "q13": "Я не запускал клуб или большую инициативу, но организовал у себя систему подготовки к выездам: инструменты, расходники, порядок проверки. Сработало то, что стало меньше забытых мелочей. Не сработало — сначала я не вел нормальный учет материалов. Теперь стараюсь исправлять это.",
                "q14": "Я зарабатывал на мелких ремонтных работах и помощи мастерам. Деньги были разными, но важнее было другое: если тебя зовут повторно, значит, ты сделал работу без лишнего риска и нервов. Для меня это самый честный показатель ценности.",
            },
            video_presentation_transcript_text=paras(
                "Это мой обычный набор на мелкий выезд: тестер, изолента, запасные клеммы, список проверки. Я специально показываю не красивую сторону, а рабочую. Потому что моя сильная сторона не в презентации, а в том, что я не люблю халтуру.",
                "Я знаю, что пока мой уровень ограничен. Но я уже понимаю важную вещь: хороший исполнитель не скрывает границы своей компетенции.",
            ),
            behavioral_signals={"completion_rate": 0.98, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_075",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill support-needing but high-potential gap with arts facilitation candidate who lacks polish but shows initiative",
            language_profile="mixed",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "support_needing_high_potential",
                "difficulty": "hard",
                "notes": "Promising arts-first facilitator with uneven language and reflection but credible initiative signals.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Kokshetau",
                "primary_domain": "theatre / youth facilitation",
                "leadership_mode": "small-group facilitator",
                "adversity_pattern": "language insecurity, caregiving load, low polish",
                "essay_style": "mixed, imperfect, sincere",
                "impact_pattern": "school theatre and shy younger students",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 65},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 79},
            },
            motivation_letter_text=paras(
                "I want to study at inVision U because I feel most alive when people who are usually silent start speaking and moving with confidence. My way into this was not formal debate or competitions. It was school theatre, warm-up games, and helping younger students stop freezing when everyone is looking at them.",
                "My writing is not very polished. Sometimes I think in Russian and speak in English badly, or the opposite. But in practice I know how to hold a room, calm nervous students, and turn a shy rehearsal into something real. Last year I helped run a small theatre circle for children whose parents worked late. We used simple scenes, voice exercises, and everyday stories instead of complicated scripts.",
                "I need support in language and probably in academic writing. But I do not think that means low potential. I am applying because I already know how to build trust in a group and how to keep showing up for work that is emotional, repetitive, and human.",
            ),
            motivation_questions={
                "q1": "The hardest period was when my mother was ill for several months and I had to help more at home while trying not to disappear from school activities. I was tired and sometimes ashamed that my work looked small compared to others. I coped by doing fewer things but doing them steadily. That changed me because I learned continuity matters more than looking impressive.",
                "q4": "I started a tiny after-school theatre circle for younger students who were too shy to join bigger events. We used short scenes from daily life and body warm-ups instead of formal plays. The outcome was that several children who first refused to speak later performed small roles in front of parents.",
                "q8": "My worldview changed when I saw that confidence is often built in very ordinary rooms, not on big stages. Before that I thought art mattered only when it looked impressive. Now I think art also matters when it helps a person feel visible without fear.",
                "q11": "I helped younger students who were scared to read aloud or speak during rehearsals. We practiced quietly first, sometimes only one sentence at a time. This gave me patience and also showed me that support can be structured, not only emotional.",
                "q13": "I helped run the small theatre circle. What worked was keeping the group small and using simple exercises. What did not work was trying one longer script too early. I learned that progression matters, especially for shy students.",
                "q14": "I did not earn much money, but I sometimes helped at children's events and received a small payment for assisting with games and setup. More important than the payment was learning that facilitation is real work and needs preparation, not just enthusiasm.",
            },
            behavioral_signals={"completion_rate": 0.93, "returned_to_edit": True, "skipped_optional_questions": 1},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_076",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill polished low-evidence negative-control gap with highly articulate but shallow global-affairs style profile",
            language_profile="en",
            modality_profile="text_plus_interview",
            scenario_meta={
                "archetype": "low_potential_strong_presentation",
                "difficulty": "easy",
                "notes": "Strong polish and high-level framing without commensurate local evidence or durable execution.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Astana",
                "primary_domain": "international relations / policy interest",
                "leadership_mode": "symbolic involvement",
                "adversity_pattern": "thin evidence beneath polished framing",
                "essay_style": "highly polished and abstract",
                "impact_pattern": "mostly aspirational",
            },
            education={
                "english_proficiency": {"type": "IELTS", "score": 7.0},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 91},
            },
            motivation_letter_text=paras(
                "I am applying to inVision U because I believe our generation must learn to navigate complexity with empathy, agility, and a truly global perspective. In a world defined by accelerating change, interdisciplinary collaboration is no longer optional; it is the condition for meaningful impact. I see education as a platform for shaping responsible, future-oriented leaders who can bridge cultures, sectors, and ideas.",
                "Throughout school I have consistently positioned myself in spaces of dialogue, reflection, and intellectual openness. I enjoy discussing social trends, public policy, and the role of youth in building more resilient communities. These experiences strengthened my conviction that transformational leadership begins with vision and an ability to communicate across differences.",
                "What draws me to inVision U is its ecosystem of innovation, diversity, and applied learning. I want to refine my voice, deepen my strategic thinking, and contribute to a community of ambitious peers committed to reimagining the future.",
            ),
            motivation_questions={
                "q1": "A difficult period for me was realizing that many people around me did not share my long-term orientation toward systems thinking and public impact. I responded by becoming more reflective and intentional in how I position my goals. This made me more resilient and internally driven.",
                "q4": "I independently initiated a small discussion gathering about youth perspectives on global challenges. The idea came from noticing that many students had opinions but not enough spaces for exchange. The outcome was an engaging conversation and a stronger sense that dialogue itself can be catalytic.",
                "q8": "My worldview was deeply shaped by reading about global leadership, diplomacy, and civic innovation. These ideas helped me understand that meaningful change requires both vision and adaptability. Since then I have tried to approach challenges more systemically.",
                "q11": "I often support peers by listening to them, encouraging them, and helping them articulate their goals. These interactions have shown me the importance of emotional intelligence in leadership. They also reinforced my belief in collaborative growth.",
                "q13": "I have promoted conversations and participated in several student-facing initiatives. What worked was creating a positive atmosphere of openness. In the future I would like to make such initiatives more scalable and strategically structured.",
                "q14": "I have occasionally supported younger students academically and informally, which I consider a form of value creation. For me, value is not always monetary; it can also be relational and intellectual.",
            },
            interview_text=paras(
                "I think what matters most today is preparing to lead in ambiguity. My interest is very much at the intersection of policy, communication, and social innovation.",
                "If I am honest, my experience is still more exploratory than deep. What I do have is strong orientation toward the future and an ability to synthesize ideas quickly.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_077",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill polished low-evidence negative-control gap with media-flavored profile that sounds stronger than actions",
            language_profile="ru",
            modality_profile="text_only",
            scenario_meta={
                "archetype": "low_potential_strong_presentation",
                "difficulty": "easy",
                "notes": "Media-oriented profile with fluent self-branding and weak factual grounding.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Almaty region",
                "primary_domain": "media / PR interest",
                "leadership_mode": "self-branding communicator",
                "adversity_pattern": "evidence-light profile",
                "essay_style": "very polished and slogan-heavy",
                "impact_pattern": "mostly claimed influence",
            },
            education={
                "english_proficiency": {"type": "school classes + courses", "score": 84},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 88},
            },
            motivation_letter_text=paras(
                "Я вижу inVision U как место, где формируются новые лидеры мнений, способные работать на стыке креатива, общественного влияния и смыслов. С ранних лет меня вдохновляла сила коммуникации: одно точное сообщение, одна правильная история могут менять отношение людей к себе, к будущему, к целым сообществам.",
                "В школе я всегда тяготела к инициативам, связанным с медиа, представлением идей и созданием позитивной атмосферы. Мне близка работа с аудиторией, продвижение важных тем и построение узнаваемого личного голоса. Я убеждена, что в современном мире именно тот, кто умеет формировать нарратив, получает возможность создавать реальные изменения.",
                "Поступление в inVision U для меня — это шаг к тому, чтобы превратить потенциал в масштабное воздействие. Я хочу учиться среди людей, которые тоже думают смело, видят шире привычных рамок и готовы создавать проекты с сильным общественным резонансом.",
            ),
            motivation_questions={
                "q1": "Сложным периодом было осознание, что не все готовы воспринимать новые идеи и современный стиль мышления. Это научило меня сохранять внутреннюю опору и не терять веру в силу коммуникации. Я стала еще осознаннее относиться к тому, как доносить свои мысли.",
                "q4": "Я предлагала идеи для медийных активностей в школе и старалась создавать вокруг них интерес. Для меня важно было не просто организовать событие, а задать эмоциональный тон и вовлечь людей в обсуждение. Это укрепило мое желание развиваться в коммуникационной сфере.",
                "q8": "На мое мировоззрение повлияли книги и интервью о брендинге, медиа и лидерстве. Я поняла, что истории управляют вниманием и формируют общественные ожидания. После этого я стала еще больше ценить силу смыслов.",
                "q11": "Я поддерживала одноклассников, когда им нужно было выступать или заявлять о себе. Мне всегда казалось важным помогать другим проявляться ярче. Этот опыт подтвердил, что коммуникация — это и про поддержку тоже.",
                "q13": "Я участвовала в школьных инициативах, связанных с подачей идей и атмосферой мероприятий. Работало то, что я умею вдохновлять и задавать тон. В будущем хотела бы делать такие форматы более крупными и заметными.",
                "q14": "Иногда я помогала младшим школьникам или знакомым с текстами и подачей материалов. Для меня это было созданием ценности через смысл и структуру, даже если речь шла не о больших деньгах.",
            },
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": False, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_078",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill support-needing but high-potential gap with caregiving and logistics-oriented applicant",
            language_profile="mixed",
            modality_profile="text_plus_interview",
            scenario_meta={
                "archetype": "support_needing_high_potential",
                "difficulty": "hard",
                "notes": "High-responsibility caregiving candidate with real logistics skill but uneven academic polish and support needs.",
            },
            content_profile={
                "language_profile": "mixed",
                "country": "Kazakhstan",
                "city_or_region": "Petropavl",
                "primary_domain": "care logistics / service coordination",
                "leadership_mode": "quiet coordinator",
                "adversity_pattern": "caregiving load, limited confidence in formal settings",
                "essay_style": "mixed, functional, sincere",
                "impact_pattern": "family and neighborhood coordination",
            },
            education={
                "english_proficiency": {"type": "school classes", "score": 67},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 80},
            },
            motivation_letter_text=paras(
                "I am applying to inVision U because most of my useful skills came from situations that do not look impressive in a normal application. My mother works long shifts, my grandmother needs regular medical visits, and for the last two years I became the person who keeps many daily things from collapsing. I manage appointments, calls, transport, shopping, and the small timing decisions that decide whether a week becomes chaos.",
                "At first I saw this only as family duty. Later I noticed that I had actually become good at coordination. I built shared notes for medicine times, arranged neighbors to help when I had exams, and even supported two other families in our building when they needed to organize clinic visits for older relatives. I am not polished in English and I do not always speak confidently, but I can keep a fragile routine working under pressure.",
                "I want inVision U because I do not want these skills to stay invisible. I need stronger academic tools and support, but I also think institutions underestimate people who learned responsibility through care work. I want to grow from survival coordination into project and service design.",
            ),
            motivation_questions={
                "q1": "The hardest period was when my grandmother's health worsened and my mother's work schedule became unpredictable at the same time. I felt like every day was a chain of small emergencies. I coped by making calendars, medicine reminders, and backup plans with neighbors. It changed me because I stopped underestimating planning as a real skill.",
                "q4": "I created a shared weekly care schedule for my family because we kept forgetting appointments, medicine times, and who could accompany my grandmother. Later I adapted the same idea for another family in our building. The outcome was simple but important: fewer missed visits and less conflict.",
                "q8": "My worldview changed when I realized that care work is not only emotional. It is logistics, communication, and system design under stress. That made me respect invisible work much more deeply.",
                "q11": "I helped an elderly neighbor and later another family in our building organize clinic visits and medicine reminders because they were overwhelmed. This gave me confidence that the routines I built at home could also help others, not only my own family.",
                "q13": "I did not launch a formal club, but I coordinated a building-level volunteer rotation for escorting older residents to the clinic during one difficult month. What worked was keeping responsibilities small and clear. What did not work was relying on verbal promises. After that I always wrote down names and times.",
                "q14": "I earned some money through part-time help at a local pharmacy and by doing delivery errands. The money mattered, but the bigger lesson was learning how much people value reliability when health or timing is involved.",
            },
            interview_text=paras(
                "Sometimes I worry that my application looks ordinary because much of my work happened at home or in my building. But those situations taught me decision-making under pressure more than any school event did.",
                "I know I need support in language and probably in academic writing. I do not need someone to teach me responsibility from zero.",
            ),
            behavioral_signals={"completion_rate": 0.97, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_079",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill support-needing but high-potential gap with service worker and self-taught coder candidate plus video evidence",
            language_profile="ru",
            modality_profile="text_plus_video",
            scenario_meta={
                "archetype": "support_needing_high_potential",
                "difficulty": "hard",
                "notes": "Service-economy candidate with uneven polish but credible coding persistence and work discipline.",
            },
            content_profile={
                "language_profile": "russian",
                "country": "Kazakhstan",
                "city_or_region": "Aktau",
                "primary_domain": "service work / self-taught coding",
                "leadership_mode": "quiet builder",
                "adversity_pattern": "long work hours, fatigue, patchy formal prep",
                "essay_style": "direct, low polish, concrete",
                "impact_pattern": "workplace and family level",
            },
            education={
                "english_proficiency": {"type": "school classes + coding tutorials", "score": 69},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 77},
            },
            application_materials_payload=application_materials(
                attachments=["shift_scheduler_mockup.png"],
                portfolio_links=["https://example.org/code/cand_079"],
                video_presentation_link="https://example.org/video/cand_079",
            ),
            motivation_letter_text=paras(
                "Я хочу поступить в inVision U, потому что последние два года живу в режиме работы и самообучения одновременно. Днем или вечером у меня смены в кафе, а ночью или утром я учу базовый Python и пробую делать простые вещи, которые решают мои же рабочие проблемы. Это не выглядит как классический школьный профиль, и академически я не самый сильный. Но если мне нужно что-то освоить, я умею долго идти маленькими шагами.",
                "Самый полезный мой проект пока очень простой: я сделал черновой вариант расписания смен и учета подмен, потому что в кафе постоянно была путаница в чате. Сначала это был обычный лист, потом я начал пробовать автоматизировать часть через код. Проект еще сырой, но даже черновой вариант уже уменьшил количество недоразумений. Это показало мне, что программирование для меня — не абстрактная мечта, а способ разгребать реальный беспорядок.",
                "Мне понадобится поддержка по языку, математике и, возможно, по тому, как вообще учиться в более сильной среде. Но я не прихожу пустым. Я приношу выносливость, привычку работать уставшим, и опыт превращать раздражающую бытовую проблему в рабочий инструмент.",
            ),
            motivation_questions={
                "q1": "Самый трудный период был, когда я совмещал длинные смены в кафе с подготовкой к экзаменам и почти перестал нормально спать. Я понимал, что без работы не могу, но и без учебы двигаться дальше тоже нельзя. Справлялся тем, что делал очень короткие, но регулярные учебные блоки и перестал ждать идеальных условий. После этого я стал меньше драматизировать усталость и больше ценить ритм.",
                "q4": "Я сам придумал сделать более понятное расписание смен и подмен для нашего кафе, потому что из-за путаницы в чате люди пропускали или путали часы. Сначала это был просто аккуратный общий лист, потом я начал пробовать добавить в это код. Результат пока промежуточный, но даже так стало меньше конфликтов.",
                "q8": "Мировоззрение поменялось, когда я понял, что код — это не только для олимпиадников и людей с идеальной подготовкой. Если ты можешь описать проблему как последовательность шагов, уже можно начинать. Это сильно снизило мой страх перед технической сферой.",
                "q11": "Я часто помогал новым сотрудникам в кафе разбираться со сменами и заказами, потому что сам помнил, как тяжело входить в ритм без нормального объяснения. Это научило меня делать инструкции проще и понятнее.",
                "q13": "Полноценного клуба я не запускал, но фактически внедрил в кафе новый способ учета смен. Сработало то, что всем была понятна боль и решение сразу экономило нервы. Не сработало — я сначала сделал слишком неудобный формат на телефоне. Пришлось упростить.",
                "q14": "Я зарабатывал на сменах в кафе и иногда на мелких цифровых задачах для знакомых, например, делал простые таблицы или помогал оформить заказы. Для меня это ценно, потому что я вижу: даже на базовом уровне цифровой навык уже может создавать практическую пользу.",
            },
            video_presentation_transcript_text=paras(
                "Это не какой-то красивый pet project. Here on the screen is the shift sheet I made first, and here is the rough code after that. It is still ugly. But before this, our manager and workers were always scrolling through chat to find who replaces whom.",
                "I wanted to show the ugly version on purpose. My point is not that I am advanced. My point is that when I meet chaos every day, I now try to structure it, not only complain.",
            ),
            behavioral_signals={"completion_rate": 0.98, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    cands.append(
        mk_candidate(
            candidate_id="cand_080",
            source_type="new_archetype",
            derived_from_candidate_id=None,
            expansion_reason="fill stronger multimodal civic-design gap with accessibility mapping candidate and portfolio-style proof",
            language_profile="en",
            modality_profile="full_multimodal",
            scenario_meta={
                "archetype": "multimodal_civic_builder",
                "difficulty": "hard",
                "notes": "Multimodal civic design candidate with map, field notes, and transcripts supporting a grounded accessibility project.",
            },
            content_profile={
                "language_profile": "english",
                "country": "Kazakhstan",
                "city_or_region": "Almaty",
                "primary_domain": "civic design / accessibility mapping",
                "leadership_mode": "project-based organizer",
                "adversity_pattern": "limited institutional responsiveness",
                "essay_style": "clear, grounded, user-centered",
                "impact_pattern": "city accessibility awareness and practical mapping",
            },
            education={
                "english_proficiency": {"type": "IELTS", "score": 6.5},
                "school_certificate": {"type": "Kazakhstan high school diploma", "score": 90},
            },
            application_materials_payload=application_materials(
                documents=["accessibility_walk_audit.pdf", "user_feedback_summary.pdf"],
                attachments=["ramp_photos.zip", "map_screenshots.png"],
                portfolio_links=["https://example.org/map/cand_080"],
                video_presentation_link="https://example.org/video/cand_080",
            ),
            motivation_letter_text=paras(
                "I am applying to inVision U because I became interested in design not through aesthetics first, but through friction. A close family friend uses a wheelchair, and after walking with her through ordinary parts of Almaty I started noticing how many 'small' barriers shape a whole day: missing curb cuts, ramps that are too steep, doors that require impossible angles, and public information that assumes one kind of body and one kind of movement.",
                "I began by making notes on my phone and taking photos. Later this turned into a small accessibility mapping project with two classmates. We walked several routes, marked problem points, wrote short descriptions, and shared the map with a local youth civic group. The project is not official and it does not fix the infrastructure directly, but it created a clearer basis for conversation than general complaints. We also collected user comments because I did not want the map to become a student interpretation with no lived perspective behind it.",
                "I want inVision U because it sits at the intersection that matters to me: field observation, design thinking, and practical social impact. I do not want to make polished presentations about inclusion without touching streets, doors, gradients, and actual users. I want to learn how to turn lived barriers into better systems.",
            ),
            motivation_questions={
                "q1": "The hardest period was when I realized how slowly institutions respond even when a problem is obvious and documented. I felt discouraged after one district office ignored our first message. I coped by focusing on making the map more useful and collecting stronger examples instead of waiting for a perfect response. That changed me because I became less dependent on immediate recognition.",
                "q4": "I started a small accessibility mapping project after repeatedly seeing the same barriers during walks with a family friend who uses a wheelchair. I designed a simple audit format, tagged locations, and added short notes on why each barrier mattered in practice. The outcome was a shareable map and a stronger evidence base for local discussion.",
                "q8": "My worldview changed when I shifted from thinking of accessibility as a special-interest issue to seeing it as a design truth about how public spaces choose their users. Once I noticed that, I could not unsee it. It changed the way I move through the city and the way I define 'good design'.",
                "q11": "I helped document routes together with a family friend and later shared the method with other students who wanted to contribute. What that gave me was humility: accessibility work should start from users and their actual experience, not from outsiders trying to look helpful.",
                "q13": "I co-organized the accessibility map with two classmates. What worked was combining route audits, photos, and user comments in one place. What did not work was our first version of categories, which was too vague. We revised it after feedback because a barrier needs to be described in a way another person can actually use.",
                "q14": "I created value by building a usable map and later helping a small youth civic group package part of the findings into a public post and meeting notes. There was no real profit, but there was practical value: people could point to exact locations instead of speaking only in general frustrations.",
            },
            interview_text=paras(
                "I do not think design should begin from style and only later remember users. For me it began with a curb that looked minor until I saw how it changed a whole route.",
                "The map is still a student project, but it taught me how observation, language, and documentation can combine into something another person can act on.",
            ),
            video_presentation_transcript_text=paras(
                "Here is one of the route screenshots and a photo of the curb I mention in the essay. On its own the obstacle looks small. In sequence with three other similar barriers, it turns a normal walk into a detour.",
                "That is why I am interested in civic design. The problem is often not hidden. It is just normalized until someone documents it carefully enough that other people can no longer ignore it.",
            ),
            behavioral_signals={"completion_rate": 1.0, "returned_to_edit": True, "skipped_optional_questions": 0},
        )
    )

    return cands


def build_report(new_candidates: list[dict]) -> str:
    source_counts = Counter(c["metadata"]["source_type"] for c in new_candidates)
    modality_counts = Counter(c["metadata"]["modality_profile"] for c in new_candidates)
    language_counts = Counter(c["metadata"]["language_profile"] for c in new_candidates)
    reused_counts = Counter(
        c["metadata"]["derived_from_candidate_id"]
        for c in new_candidates
        if c["metadata"]["derived_from_candidate_id"]
    )

    top_reused = sorted(reused_counts.items(), key=lambda item: (-item[1], item[0]))
    reused_lines = [f"- `{cid}`: {count}" for cid, count in top_reused]

    return "\n".join(
        [
            "# Candidate Expansion Report",
            "",
            "## 1. How Many Candidates Were Added",
            f"- Added candidates: {len(new_candidates)}",
            "",
            "## 2. How Many Counterfactual vs New Archetype",
            f"- counterfactual: {source_counts.get('counterfactual', 0)}",
            f"- new_archetype: {source_counts.get('new_archetype', 0)}",
            "",
            "## 3. How Many By Modality Profile",
            *(f"- {key}: {modality_counts.get(key, 0)}" for key in ["text_only", "text_plus_interview", "text_plus_video", "full_multimodal"]),
            "",
            "## 4. How Many By Language Profile",
            *(f"- {key}: {language_counts.get(key, 0)}" for key in ["ru", "en", "mixed"]),
            "",
            "## 5. Which Source Candidates Were Reused Most Often",
            *reused_lines,
            "",
            "## 6. Which Archetype Gaps Were Filled",
            "- sustained builder/founder profiles with repeat execution and process ownership",
            "- strong evidence but weak motivation cases that separate competence from narrative fit",
            "- non-STEM / arts-first / vocational execution profiles with grounded outputs",
            "- polished but low-evidence negative controls",
            "- support-needing but high-potential applicants whose promise is not driven by polish",
            "- stronger multimodal profiles with transcripts, documents, and portfolio-like proof",
            "",
            "## 7. Major Realism Risks Still Remaining",
            "- Some counterfactual families still share recognizable motifs, so pairwise leakage risk remains if train and eval splits are not source-aware.",
            "- Multimodal proof is still synthetic and cleaner than many real applicant attachments or transcripts would be.",
            "- Community-action and problem-solving narratives remain more common than truly ordinary low-signal applicants.",
            "",
            "## 8. Recommendation For Next Annotation Pass",
            "- Annotate counterfactual families with explicit source-aware split rules so near-neighbors do not leak across train and evaluation sets.",
            "- Prioritize adjudication on the new multimodal and strong-evidence/weak-motivation cases because they expand boundary conditions in the rubric.",
            "- Add one more small batch of mundane, medium-signal profiles to reduce overexposure to unusually proactive narratives.",
            "",
        ]
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates_path = root / "data" / "archive" / "candidates.json"
    out_candidates_path = root / "research" / "data" / "candidates_expanded_v1.json"
    out_report_path = root / "docs" / "archive" / "candidate_expansion_report.md"

    base = json.loads(candidates_path.read_text(encoding="utf-8"))
    base_candidates = base["candidates"]
    new_candidates = build_candidates()

    base_ids = {c["candidate_id"] for c in base_candidates}
    new_ids = [c["candidate_id"] for c in new_candidates]
    if len(new_ids) != len(set(new_ids)):
        raise ValueError("duplicate candidate_id within new candidates")
    overlap = sorted(base_ids.intersection(new_ids))
    if overlap:
        raise ValueError(f"candidate ids already exist: {overlap}")

    expanded = {
        "meta": {
            "source_file": "data/archive/candidates.json",
            "expansion_method": "counterfactual_plus_archetype_gap_fill",
            "version": "candidates_expanded_v1",
        },
        "candidates": base_candidates + new_candidates,
    }

    out_candidates_path.write_text(json.dumps(expanded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_report_path.write_text(build_report(new_candidates), encoding="utf-8")

    print(f"base={len(base_candidates)} new={len(new_candidates)} total={len(expanded['candidates'])}")
    print(Counter(c["metadata"]["source_type"] for c in new_candidates))
    print(Counter(c["metadata"]["modality_profile"] for c in new_candidates))
    print(Counter(c["metadata"]["language_profile"] for c in new_candidates))


if __name__ == "__main__":
    main()
