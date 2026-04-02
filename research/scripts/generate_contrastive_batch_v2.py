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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.input import CandidateInput

RAW_DIR = ROOT / "data" / "ml_workbench" / "raw" / "generated" / "contrastive_batch_v2"
PACK_DIR = ROOT / "data" / "ml_workbench" / "processed" / "annotation_packs" / "contrastive_batch_v2"

RAW_JSONL = RAW_DIR / "contrastive_batch_v2_api_input.jsonl"
GEN_MANIFEST_JSONL = RAW_DIR / "contrastive_batch_v2_generation_manifest.jsonl"
SUMMARY_JSON = RAW_DIR / "contrastive_batch_v2_summary.json"

PACK_JSONL = PACK_DIR / "contrastive_batch_v2_annotation_pack.jsonl"
PACK_JSON = PACK_DIR / "contrastive_batch_v2_annotation_pack.json"
PACK_TABLE_CSV = PACK_DIR / "contrastive_batch_v2_annotation_pack_table.csv"
PACK_MANIFEST_JSON = PACK_DIR / "contrastive_batch_v2_annotation_pack_manifest.json"

QUESTION_BANK = {
    1: "Describe a project, idea, or solution you started on your own. What problem were you trying to solve, and what happened?",
    2: "If you joined inVision U tomorrow, what local issue would you turn into a semester project first, and why that one?",
    3: "Tell us about a period when pressure at home or school changed how you work.",
    4: "What would your future peers gain from having you in a team?",
    5: "Describe a time you worked with people who approached a problem very differently from you.",
    6: "What habit or routine has done the most to improve your results recently?",
    7: "Was there a goal you worked toward for a long time and did not reach? What stayed with you?",
    8: "What kind of environment helps you do your best work, and what part of yourself still needs development?",
    9: "Tell us about something you tried that stayed smaller than you hoped. Why did it stay small?",
    10: "What local problem stays on your mind even when nobody asks you about it?",
    11: "Tell us about a time when feedback made you change your approach.",
}


@dataclass(frozen=True)
class ContrastSpec:
    candidate_id: str
    anchor_candidate_id: str
    anchor_source: str
    contrast_type: str
    dimensions_changed: tuple[str, ...]
    bucket: str
    has_interview: bool
    polish_level: str
    evidence_level: str
    initiative_level: str
    ambiguity_level: str
    city: str
    english_type: str
    english_score: float
    school_type: str
    school_score: float
    opening: str
    interest_area: str
    local_issue: str
    action: str
    outcome: str
    limitation: str
    personal_context: str
    help_example: str
    future_goal: str
    failed_goal: str
    worldview_shift: str
    teamwork_example: str
    peer_contribution: str
    intended_effect: str
    generator_notes: str


SPECS: list[ContrastSpec] = [
    ContrastSpec(
        candidate_id="syn_contrast_v2_001",
        anchor_candidate_id="syn_eng_v1_009",
        anchor_source="synthetic_batch_v1",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength",),
        bucket="long",
        has_interview=True,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Ekibastuz",
        english_type="school + online practice",
        english_score=68,
        school_type="Kazakhstan certificate",
        school_score=4.1,
        opening="Growing up in Ekibastuz made everyday infrastructure feel personal rather than abstract.",
        interest_area="repair culture, everyday electrical reliability, and low-cost tools",
        local_issue="families in older apartment blocks lose time and money when outages damage small appliances and nobody nearby can check what is still repairable",
        action="put together a repair notebook at home, sorted spare plugs and cables, and built one backup desk lamp from reused parts for a neighbor during a winter outage",
        outcome="The lamp worked through exam week and two neighbors later asked me to inspect chargers, but the effort stayed informal and never became a stable repair routine.",
        limitation="Most of the signal is real but still lightly documented and smaller than I want it to be.",
        personal_context="My mother works long pharmacy shifts, so I often handle evening errands and help a younger cousin with homework before I return to my own experiments.",
        help_example="I once cleaned an old family computer and set up a simpler folder system so a younger relative could use it for school assignments.",
        future_goal="learn how to move from one-off fixes toward small tools that families can actually trust",
        failed_goal="I tried to prepare a stronger prototype for a school technology fair, but I only reached a rough demonstration and not a convincing final build.",
        worldview_shift="Seeing how one small lamp changed a neighbor's routine made me stop treating repair as a hobby only.",
        teamwork_example="During a class assignment, I kept pushing the group to test what actually worked instead of adding features that looked impressive but failed during use.",
        peer_contribution="patience with trial and error, calm debugging, and a habit of checking whether a solution is useful outside theory",
        intended_effect="Weaker-grounded neighboring case to a strong repair-and-maker anchor.",
        generator_notes="Keeps the same broad outage-repair domain as the anchor but reduces scale, documentation, and proof.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_002",
        anchor_candidate_id="syn_eng_v1_034",
        anchor_source="synthetic_batch_v1",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength", "outcome_specificity"),
        bucket="medium",
        has_interview=True,
        polish_level="high",
        evidence_level="strong",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Kokshetau",
        english_type="IELTS",
        english_score=7.0,
        school_type="Kazakhstan certificate",
        school_score=4.4,
        opening="I started paying serious attention to educational opportunity gaps in Kokshetau when I realized how often good information arrives too late for ordinary students.",
        interest_area="communication, opportunity access, and practical peer guidance",
        local_issue="students outside specialized schools often miss competitions, grants, and application deadlines because the information lives in scattered chats and private circles",
        action="built a weekly voice-note bulletin with deadlines, opened Saturday question sessions in the library, and kept a shared spreadsheet of local opportunities with one teacher",
        outcome="The bulletin reached students from two schools, seventeen students attended at least one session, and five later submitted applications they had not planned to try.",
        limitation="The work is still local and would become stronger with better delegation and cleaner design.",
        personal_context="I am the first person in my family trying to navigate these systems, so a lot of the learning happened by asking questions and then writing the answers down clearly.",
        help_example="I often review forms with younger students who assume application language is only for more connected families.",
        future_goal="turn information access into systems that feel ordinary and reliable instead of lucky",
        failed_goal="My first version was only a Telegram post series, and I saw that students ignored it unless I added live question time.",
        worldview_shift="Once I watched a classmate skip an application only because the process looked mysterious, I stopped thinking communication work was secondary.",
        teamwork_example="When another volunteer wanted the bulletin to stay selective, I argued for including smaller local programs because those were the ones students could actually reach.",
        peer_contribution="clear explanation, practical synthesis, and steady attention to who is still missing from the room",
        intended_effect="Cleaner, more grounded neighbor to a communication-heavy anchor that had thin execution.",
        generator_notes="Same broad communication domain as the anchor, but with concrete attendance, teacher uptake, and limited yet credible outcomes.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_003",
        anchor_candidate_id="cand_003",
        anchor_source="seed_pack",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength",),
        bucket="medium",
        has_interview=True,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Taraz",
        english_type="school classes",
        english_score=63,
        school_type="Attestat",
        school_score=4.0,
        opening="Life near the canal in Taraz made me curious about environmental problems long before I knew how to study them properly.",
        interest_area="water quality, local ecology, and practical public awareness",
        local_issue="people in my area complain about dirty canal water and waste near the riverbank, but school conversations about it stay theoretical",
        action="kept a notebook of visible pollution spots, took a few jar samples at home, and helped organize one weekend cleanup after speaking with a biology teacher",
        outcome="The cleanup gathered one small group and the notebook helped me make a school presentation, but I have not yet built a lasting system or repeated testing method.",
        limitation="The concern is genuine, yet the proof is still early and much thinner than I would like.",
        personal_context="I help in a family bakery after school, so most of my project time has to fit into short evening windows.",
        help_example="I sometimes walk with a younger neighbor to science club and lend simple materials so the trip feels less intimidating.",
        future_goal="learn how to turn concern about water into methods that are simple, honest, and repeatable",
        failed_goal="I hoped to borrow better testing equipment for a month, but I only managed one short use and did not collect enough data.",
        worldview_shift="The more I looked closely at one local problem, the less I believed that general statements were enough.",
        teamwork_example="When classmates wanted a dramatic environmental poster, I pushed for including even the weak sample notes so the work would stay honest.",
        peer_contribution="honesty about weak evidence, persistence with small steps, and real interest in local environmental problems",
        intended_effect="Thinner-evidence neighbor to a stronger community-and-water anchor.",
        generator_notes="Keeps the water-and-pollution theme but lowers repeatability, resource access, and outcome strength.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_004",
        anchor_candidate_id="cand_017",
        anchor_source="seed_pack",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength",),
        bucket="short",
        has_interview=False,
        polish_level="balanced",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Almaty",
        english_type="school + online practice",
        english_score=78,
        school_type="Kazakhstan certificate",
        school_score=4.4,
        opening="Almaty taught me that city problems often look technical on the surface but become human very quickly.",
        interest_area="civic technology, school coordination, and air-quality awareness",
        local_issue="students talk about air pollution and transport delays every day, but most complaints disappear into informal chats with no shared record",
        action="made a simple spreadsheet where classmates could log smoky stops and bus delays, then used it in one school presentation about commuting stress",
        outcome="The spreadsheet gave one class something concrete to discuss, but the work never moved beyond a small pilot and one presentation.",
        limitation="There is a real starting point here, yet the scale is still modest and unproven.",
        personal_context="After my father spent several months without steady work, I became more deliberate about time and less willing to treat school problems as somebody else's topic.",
        help_example="I helped one younger student learn how to track homework deadlines digitally instead of keeping them in scattered notes.",
        future_goal="connect small civic observations to tools that make public problems harder to ignore",
        failed_goal="I wanted to turn the spreadsheet into a cleaner map interface, but I stayed at the draft stage.",
        worldview_shift="Once I saw how differently classmates described the same route, I realized data only matters if ordinary people can contribute to it.",
        teamwork_example="During the presentation, I had to slow the conversation down because others wanted slogans before basic facts.",
        peer_contribution="structured thinking, grounded curiosity, and a preference for making public problems concrete",
        intended_effect="Committee-confusable civic-tech neighbor to a stronger seed anchor, but with more limited proof.",
        generator_notes="Same air-quality and coordination domain as the seed anchor, but the visible evidence is much smaller and more school-bounded.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_005",
        anchor_candidate_id="syn_eng_v1_023",
        anchor_source="synthetic_batch_v1",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength", "outcome_specificity"),
        bucket="long",
        has_interview=True,
        polish_level="balanced",
        evidence_level="strong",
        initiative_level="acted",
        ambiguity_level="clean",
        city="a village near Petropavl",
        english_type="school classes",
        english_score=65,
        school_type="Village school certificate",
        school_score=4.1,
        opening="After spring flooding in a village near Petropavl, I stopped thinking about resilience as a slogan and started treating it as a daily logistics problem.",
        interest_area="community recovery, school continuity, and practical coordination",
        local_issue="after seasonal flooding, families lose routines faster than outsiders notice, and students are the first to fall behind once materials and transport become unstable",
        action="set up a rotating study room in a dry storage space, built a handwritten needs board with one teacher, and organized a box system for notebooks, flashlights, and borrowed chargers",
        outcome="Two classes used the room for three weeks, a local shop donated basics after seeing the list, and the teacher later kept the box system in place for emergency periods.",
        limitation="I still improvised too much and want stronger planning tools before I call this durable.",
        personal_context="My own family was also moving between relatives during repairs, so I was balancing school with younger siblings and constant uncertainty.",
        help_example="I walked one new classmate home after evening study sessions because the route back felt unfamiliar after the flood.",
        future_goal="build recovery systems that help ordinary schools keep dignity and continuity during disruption",
        failed_goal="My first attempt to coordinate volunteers by chat failed because most people were already overwhelmed and the messages got buried.",
        worldview_shift="Rebuilding one routine at a time made me value systems more than good intentions.",
        teamwork_example="I had to accept that adults wanted a simpler list than the one I first designed, because usefulness mattered more than perfect categories.",
        peer_contribution="steady coordination, calm under pressure, and a habit of turning vague worry into roles and next steps",
        intended_effect="Stronger and more specific evidence neighbor to a hardship-and-recovery anchor.",
        generator_notes="Keeps the flood-recovery theme but makes the visible outputs more concrete, more adopted, and easier to verify.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_006",
        anchor_candidate_id="cand_025",
        anchor_source="seed_pack",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength",),
        bucket="medium",
        has_interview=True,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Karaganda",
        english_type="school + online reading",
        english_score=72,
        school_type="Attestat",
        school_score=4.2,
        opening="In Karaganda, practical science started feeling meaningful to me only when it touched everyday problems at home.",
        interest_area="low-cost science, household water problems, and useful teaching",
        local_issue="many families talk about old pipes and strange water taste, but few people know how to compare simple options or test anything carefully",
        action="built one rough bottle-filter model at home, compared water clarity with simple notes, and showed the idea to two younger students who liked the experiment",
        outcome="The demonstration was encouraging, but I have not yet repeated it enough or moved beyond a home example.",
        limitation="The curiosity is real, but the evidence is still thinner than the ambition behind it.",
        personal_context="I spend a lot of time helping at home because family money has been uneven and younger siblings need support with school.",
        help_example="I often explain science homework to my brother in simpler language so it feels less like memorization.",
        future_goal="connect home experiments to more reliable evidence and broader usefulness",
        failed_goal="I wanted to prepare a better project for a district fair, but family duties kept breaking the study rhythm.",
        worldview_shift="Once I saw younger students react to a real experiment instead of a textbook summary, I started thinking differently about science education.",
        teamwork_example="In one class task, I had to insist that we describe what the filter could not prove, not only what looked encouraging.",
        peer_contribution="practical curiosity, patience with explanation, and a sincere wish to make science feel usable",
        intended_effect="Neighboring case to a strong hardship-plus-science seed anchor, but with much weaker proof.",
        generator_notes="Same science-for-daily-life orientation as the anchor, but the visible project stays home-scale and lightly evidenced.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_007",
        anchor_candidate_id="syn_eng_v1_031",
        anchor_source="synthetic_batch_v1",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength", "outward_orientation"),
        bucket="long",
        has_interview=True,
        polish_level="high",
        evidence_level="strong",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Astana",
        english_type="IELTS",
        english_score=7.5,
        school_type="specialized mathematics high school diploma",
        school_score=96,
        opening="My strongest habit in Astana has always been turning vague academic difficulty into something I can measure and improve.",
        interest_area="statistics, programming, and applied operational systems",
        local_issue="school canteens and student clubs often waste food and supplies because nobody tracks simple patterns closely enough to change behavior",
        action="built a small waste-tracking script with a manual entry form, ran it with a canteen worker for a month, and turned the results into a visual board for one teacher and student council",
        outcome="The canteen adjusted purchase estimates for one menu cycle and the council used the board to argue for better lunch timing.",
        limitation="I still need more practice building with people who are not already comfortable with numbers.",
        personal_context="My academic routine is strong, but I had to learn recently not to keep every useful idea inside a personal spreadsheet.",
        help_example="I help classmates clean up messy data tables and explain patterns in plain language before group projects.",
        future_goal="use rigorous quantitative thinking on messy public problems instead of only competition tasks",
        failed_goal="My first version of the tracker was too technical and nobody wanted to enter data until I simplified the form.",
        worldview_shift="The moment numbers changed a real purchasing decision, statistics stopped feeling closed inside schoolwork.",
        teamwork_example="I had to listen when the canteen worker said my first categories made no sense on a busy day, even though the model looked neat to me.",
        peer_contribution="discipline, evidence-minded reasoning, and a willingness to translate technical work into simple next steps",
        intended_effect="More outward and better-grounded neighbor to a narrow quantitative anchor.",
        generator_notes="Same analytics domain as the anchor, but shifts from self-optimization toward a small real-world pilot with visible uptake.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_008",
        anchor_candidate_id="syn_eng_v1_015",
        anchor_source="synthetic_batch_v1",
        contrast_type="evidence_contrast",
        dimensions_changed=("evidence_strength",),
        bucket="short",
        has_interview=False,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Turkistan",
        english_type="school classes",
        english_score=64,
        school_type="Attestat",
        school_score=3.9,
        opening="I think often about how quickly younger children lose confidence when no welcoming place exists after school.",
        interest_area="reading access, community care, and ordinary belonging",
        local_issue="in my neighborhood, younger children drift between courtyards and shops because after-school spaces feel either expensive or unavailable",
        action="borrowed a few books from a teacher and tried three weekend reading meetings in one courtyard with two friends",
        outcome="A small group of children came back more than once, but the meetings stayed irregular and ended when exam season became busy.",
        limitation="The effort was sincere, yet still far from stable enough to count as a durable space.",
        personal_context="I also spend many evenings helping at home, so I know how easily good intentions disappear without structure.",
        help_example="I sometimes sit with one younger neighbor to practice reading aloud when the house nearby is too noisy.",
        future_goal="build study spaces that feel stable instead of temporary",
        failed_goal="I hoped to find a parent volunteer to keep the meetings going, but I did not manage it.",
        worldview_shift="Trying even a small reading group made me notice how much routine matters to confidence.",
        teamwork_example="When one friend wanted the meetings to become a game day only, I argued for keeping at least a short reading part because that was the main reason children came.",
        peer_contribution="warmth, patience with younger students, and a real instinct for what makes a group feel usable",
        intended_effect="Thinner-proof neighbor to a stronger community-reading anchor.",
        generator_notes="Same broad child-support domain as the anchor, but outcomes and duration are clearly weaker.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_009",
        anchor_candidate_id="cand_012",
        anchor_source="seed_pack",
        contrast_type="polish_contrast",
        dimensions_changed=("presentation_polish", "english_fluency"),
        bucket="medium",
        has_interview=True,
        polish_level="high",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Aktobe",
        english_type="IELTS",
        english_score=7.0,
        school_type="Kazakhstan certificate",
        school_score=4.3,
        opening="What attracts me most to practical technology is not novelty by itself, but the way a well-designed tool can quietly remove friction from daily life.",
        interest_area="small digital tools, device repair, and usable design",
        local_issue="students in ordinary schools often lose time because classroom resources are disorganized and simple digital tasks feel harder than they should",
        action="rebuilt two classroom routers, organized a basic device checkout sheet, and drafted a simple booking page for one shared projector",
        outcome="Teachers used the sheet for a term and fewer lessons lost time to missing cables or overlapping requests.",
        limitation="Most of my work so far is still contained within small settings rather than larger collaborative projects.",
        personal_context="I am usually quieter in crowded rooms, so the technical side of a problem still feels easier to me than the public side.",
        help_example="I also help neighbors set up phones and explain each step slowly so the fix does not stay mysterious.",
        future_goal="combine careful technical thinking with stronger public communication and product judgment",
        failed_goal="I tried to turn the booking page into a fuller school tool, but I kept overbuilding features instead of learning from early users.",
        worldview_shift="I learned that people trust technology much faster when it solves one annoying problem clearly.",
        teamwork_example="During one school task, I had to accept a teacher's simpler workflow instead of the cleaner interface I first preferred.",
        peer_contribution="calm debugging, precise listening, and a bias toward tools that reduce friction",
        intended_effect="Same broad substance as a low-polish technical seed anchor, but with much stronger self-presentation.",
        generator_notes="This is a polish-lift contrast: similar quiet-service substance, cleaner English, and more reflective framing.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_010",
        anchor_candidate_id="syn_eng_v1_039",
        anchor_source="synthetic_batch_v1",
        contrast_type="polish_contrast",
        dimensions_changed=("presentation_polish",),
        bucket="short",
        has_interview=False,
        polish_level="low",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="a village near Pavlodar",
        english_type="school classes",
        english_score=60,
        school_type="Village school certificate",
        school_score=4.0,
        opening="In my village, computers are talked about like they should already work for everyone, but that is not how it feels in real life.",
        interest_area="basic computer access and practical learning",
        local_issue="younger students are expected to type, search, and send files even when the old machines at school barely start",
        action="helped a teacher clean the old computer room, label the machines that still worked, and make one simple practice hour after class",
        outcome="A few younger students stayed to learn typing, and the room felt less locked than before.",
        limitation="I can do useful things in small places, but I still explain them in a very plain way and not always in the strongest order.",
        personal_context="I do chores before school and often study late, so I got used to doing quiet work without much attention.",
        help_example="I sometimes show one younger cousin how to save files properly because that small thing can decide whether schoolwork gets finished.",
        future_goal="learn enough technology and teamwork to make basic access feel normal, not lucky",
        failed_goal="I hoped to keep the practice hour twice a week, but it stayed occasional once exams came closer.",
        worldview_shift="Once I saw younger students wait in line for one working machine, I stopped thinking access problems were small.",
        teamwork_example="I had to work with a teacher who wanted quick results, while I kept saying we should first check what actually worked.",
        peer_contribution="steadiness, patience, and a practical sense of how to make basic tools usable",
        intended_effect="Lower-polish neighboring case to a support-needed tech-access anchor with similar substance.",
        generator_notes="Substance stays close to the anchor, but the voice is blunter and more under-framed.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_011",
        anchor_candidate_id="syn_eng_v1_034",
        anchor_source="synthetic_batch_v1",
        contrast_type="polish_contrast",
        dimensions_changed=("presentation_polish",),
        bucket="medium",
        has_interview=True,
        polish_level="low",
        evidence_level="thin",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Shymkent",
        english_type="school + online videos",
        english_score=69,
        school_type="Attestat",
        school_score=4.1,
        opening="I like when students hear about useful programs early enough to try them, not only later when somebody else already got there.",
        interest_area="student communication and youth opportunity sharing",
        local_issue="good events and programs often stay inside friend groups or school circles that already have more information",
        action="helped make a notice board and one chat channel about events, then spoke in class a few times about trying bigger things",
        outcome="Some classmates read it, but I cannot say it changed much because the work stayed small.",
        limitation="The substance is still limited and I do not want to pretend it already became something large.",
        personal_context="I learned most of this by asking older students direct questions because nobody around me had a ready system for it.",
        help_example="I sometimes forward smaller local opportunities to one younger student who does not have strong school connections.",
        future_goal="learn how to make opportunity-sharing work consistently instead of depending on reminders and luck",
        failed_goal="I wanted the chat channel to stay active through the term, but I did not build enough shared responsibility around it.",
        worldview_shift="It surprised me how much students ignore information until somebody explains why it matters in plain language.",
        teamwork_example="In one planning talk, I had to push back when others wanted only prestigious opportunities on the board because that would have excluded most students again.",
        peer_contribution="plain communication, honesty about limits, and attention to who gets left outside information loops",
        intended_effect="Lower-polish neighboring case to a polished-thin anchor with roughly similar underlying substance.",
        generator_notes="Keeps the same youth-opportunity domain as the anchor but strips away polished framing and leaves the modest scale exposed.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_012",
        anchor_candidate_id="cand_047",
        anchor_source="seed_pack",
        contrast_type="polish_contrast",
        dimensions_changed=("presentation_polish",),
        bucket="long",
        has_interview=True,
        polish_level="high",
        evidence_level="strong",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Pavlodar",
        english_type="TOEFL",
        english_score=97,
        school_type="specialized high school diploma",
        school_score=4.6,
        opening="I have become most engaged by technology when it behaves less like a competition topic and more like a form of public usefulness.",
        interest_area="repair-based design, small devices, and resilient household tools",
        local_issue="older apartment blocks lose time and money on basic breakdowns that are too minor for a service shop but too disruptive to ignore",
        action="rebuilt desk lights from discarded components, repaired small fans and kettles for neighbors, and tested a compact battery lamp for one stairwell during outages",
        outcome="The lamp stayed in use through one outage period and neighbors began bringing small repairs more regularly.",
        limitation="I still need broader collaborative experience and much better documentation than I currently have.",
        personal_context="Most of my practical learning came from working alone for long stretches, which made me good at persistence but slower at asking for feedback.",
        help_example="I like explaining how a repair works instead of handing the device back like a closed trick.",
        future_goal="turn repair-minded experimentation into products and systems that hold up under real use",
        failed_goal="I once designed a more ambitious power-backup prototype, but the circuit became too fragile because I rushed the testing stage.",
        worldview_shift="Useful design started feeling different from hobby tinkering once people depended on the results during real outages.",
        teamwork_example="I learned a lot from working with a classmate who cared more about appearance, because it forced me to explain why reliability mattered before style.",
        peer_contribution="hands-on persistence, careful iteration, and an instinct for making tools feel practical rather than decorative",
        intended_effect="Higher-polish neighboring case to a low-polish strong-maker seed anchor.",
        generator_notes="Same general maker substance as the seed anchor, but the visible self-presentation is much more articulate and reflective.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_013",
        anchor_candidate_id="syn_eng_v1_015",
        anchor_source="synthetic_batch_v1",
        contrast_type="polish_contrast",
        dimensions_changed=("presentation_polish", "english_fluency"),
        bucket="medium",
        has_interview=False,
        polish_level="low",
        evidence_level="strong",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Taraz",
        english_type="school classes",
        english_score=61,
        school_type="Attestat",
        school_score=4.0,
        opening="I am not from a place with many clubs or extra spaces, so when younger children have nowhere to go after school I notice it fast.",
        interest_area="small learning groups and community trust",
        local_issue="many younger children in my area wait outside until parents finish work and do not have a calm place to do homework",
        action="started a table in one courtyard with borrowed books, worksheets, and tea after talking with two parents",
        outcome="Children kept coming two evenings a week for a while and older students sometimes stayed to help.",
        limitation="The work is useful, but I still explain it in a simple way and I know the next step needs more structure than I have now.",
        personal_context="My own routine has a lot of family errands inside it, so anything that lasts has to fit real life and not only good intentions.",
        help_example="I also help one elderly neighbor carry food and sit for a few minutes because loneliness is a real problem too.",
        future_goal="build local study spaces that feel stable and welcoming, not temporary and fragile",
        failed_goal="I wanted the table to run through winter, but I did not yet solve where the group could meet once the weather changed.",
        worldview_shift="It changed me to see that one predictable place can lower fear much faster than one inspirational speech.",
        teamwork_example="I had to negotiate with adults who wanted only the strongest children there, while I kept arguing that the point was openness.",
        peer_contribution="consistency, warmth, and practical care for the people who are easiest to overlook",
        intended_effect="Lower-polish neighboring case to a shortlist-worthy community anchor with similar substance.",
        generator_notes="This is a polish-down contrast: the community work stays real, but the visible framing is plainer and less confident.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_014",
        anchor_candidate_id="syn_eng_v1_031",
        anchor_source="synthetic_batch_v1",
        contrast_type="polish_contrast",
        dimensions_changed=("presentation_polish",),
        bucket="medium",
        has_interview=True,
        polish_level="high",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Astana",
        english_type="IELTS",
        english_score=7.5,
        school_type="NIS diploma",
        school_score=4.6,
        opening="For most of secondary school, my strongest instinct has been to make uncertainty smaller by organizing it carefully.",
        interest_area="quantitative reasoning, structured study systems, and honest feedback",
        local_issue="students who are capable in mathematics often plateau because nobody teaches them how to diagnose their own mistakes systematically",
        action="kept a rigorous error log, turned parts of it into a shared review sheet, and used the sheet while mentoring two younger students before exams",
        outcome="The sheet improved my own preparation and gave the younger students a clearer way to review, but the work stayed academically narrow.",
        limitation="I know the next step is not a harder worksheet only, but a wider context where rigor has to meet ambiguity.",
        personal_context="I have spent more time inside disciplined academic systems than most of my peers, which is both a strength and a limit.",
        help_example="I enjoy helping others see that one repeated mistake often has a more elegant explanation than it first appears.",
        future_goal="test whether quantitative discipline can become more publicly useful without losing rigor",
        failed_goal="I spent months preparing for a national-level competition result that did not arrive, and the disappointment made me think harder about range.",
        worldview_shift="A study method became more interesting to me once I saw it help someone else, not only my own score.",
        teamwork_example="I had to work with a peer who preferred intuition to structure, and the best outcome came when I explained the purpose of the system instead of defending the format.",
        peer_contribution="clarity, disciplined reflection, and a willingness to make complex ideas easier to inspect",
        intended_effect="Higher-polish neighboring case to a narrow-academic anchor with similar underlying substance.",
        generator_notes="Same narrow-academic substance as the anchor, but more fluent self-presentation and reflective framing.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_015",
        anchor_candidate_id="cand_003",
        anchor_source="seed_pack",
        contrast_type="initiative_contrast",
        dimensions_changed=("initiative_level",),
        bucket="long",
        has_interview=True,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="reflective",
        ambiguity_level="clean",
        city="Taraz",
        english_type="school classes",
        english_score=66,
        school_type="Attestat",
        school_score=4.1,
        opening="The canal near my home in Taraz made me care about environmental questions years before I knew what useful action would look like.",
        interest_area="water quality and local environmental responsibility",
        local_issue="people complain about dirty water and litter near the river, but most conversations end before anyone tests, measures, or organizes anything",
        action="read about simple testing methods, kept screenshots and notes, and spoke with a teacher about possible sampling, but I have not yet turned the concern into a steady project",
        outcome="What I have now is understanding and urgency more than a finished action.",
        limitation="The gap between seeing a real problem and organizing a durable response is exactly what I still need to close.",
        personal_context="I help my family after school, so it became easy to tell myself that reflection was enough while more practical steps kept moving further away.",
        help_example="I sometimes join community cleanups, but I know joining something once is not the same as building it.",
        future_goal="learn how to move from environmental concern to concrete, repeatable local action",
        failed_goal="I planned to keep a one-month water diary with photos and sample notes, but I stopped after the first week.",
        worldview_shift="The problem became more serious to me once I saw how quickly people accepted it as normal.",
        teamwork_example="During one class discussion, I realized I was good at naming the issue clearly but still hesitant to volunteer for the unglamorous parts of the response.",
        peer_contribution="honest reflection, seriousness about local problems, and a willingness to admit where action is still missing",
        intended_effect="Initiative-down neighbor to a stronger water-and-community seed anchor.",
        generator_notes="Same environmental concern as the anchor, but keeps the visible case on reflection and planning rather than durable self-started action.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_016",
        anchor_candidate_id="syn_eng_v1_017",
        anchor_source="synthetic_batch_v1",
        contrast_type="initiative_contrast",
        dimensions_changed=("initiative_level",),
        bucket="short",
        has_interview=False,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="reflective",
        ambiguity_level="clean",
        city="Kyzylorda",
        english_type="school classes",
        english_score=63,
        school_type="Attestat",
        school_score=3.9,
        opening="What stays with me most is how quickly younger students decide they are weak when really they are only unsupported.",
        interest_area="after-school learning, confidence, and welcoming study spaces",
        local_issue="students without tutors or older siblings nearby often stop asking questions even when they want help",
        action="keep sketching ideas for a weekend study group and already help one cousin sometimes, but I have not yet started something regular outside home",
        outcome="The gap between caring and organizing is exactly what I want to learn to close.",
        limitation="The concern is real, but so far the initiative is still mostly internal and home-based.",
        personal_context="My mother works shifts, so evenings often move around family needs before anything else can happen.",
        help_example="I know how much one calm explanation can change a younger student's confidence, which is why I keep returning to this idea.",
        future_goal="build a study space that feels normal and reachable for students who are shy about asking for help",
        failed_goal="I tried to pick a start date twice, but I did not yet create a routine strong enough to hold other people around it.",
        worldview_shift="Watching hesitation grow in ordinary classrooms made me take confidence problems more seriously than I did before.",
        teamwork_example="In school projects, I usually notice who has stopped speaking, even when I am still learning how to turn that notice into action.",
        peer_contribution="attention to quieter people and seriousness about making a group feel safe to join",
        intended_effect="Initiative-down neighboring case to a stronger tutoring-and-access anchor.",
        generator_notes="Same broad educational access concern as the anchor, but with only home-scale help and no durable public initiative yet.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_017",
        anchor_candidate_id="cand_012",
        anchor_source="seed_pack",
        contrast_type="initiative_contrast",
        dimensions_changed=("initiative_level",),
        bucket="medium",
        has_interview=True,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="reflective",
        ambiguity_level="clean",
        city="Aktobe",
        english_type="school + online practice",
        english_score=74,
        school_type="Kazakhstan certificate",
        school_score=4.3,
        opening="I am often most motivated by technology when I can picture how it would make one ordinary school day less confusing.",
        interest_area="practical design, digital organization, and quiet technical service",
        local_issue="shared school devices and projectors create unnecessary stress because nobody has a clear system for access or maintenance",
        action="drafted interfaces, watched tutorials, and listed recurring problems for a booking tool, but I have not yet built a version other people can use",
        outcome="So far the work is still at the stage of observation and sketches rather than a tested solution.",
        limitation="I have genuine technical curiosity, but I still need a stronger push from idea into public trial.",
        personal_context="I work best in small groups, so it became easy to keep refining the idea privately instead of releasing an imperfect first version.",
        help_example="I still help classmates fix formatting and setup issues because small technical barriers bother me more than they should.",
        future_goal="learn how to ship useful first versions before everything feels complete",
        failed_goal="I set myself a deadline to build a school booking prototype, but I kept redesigning the screens instead of testing them.",
        worldview_shift="The longer I stayed in planning mode, the more I understood that clarity is not the same thing as action.",
        teamwork_example="When a classmate suggested starting with paper sign-ups first, I resisted at first and later realized that a simpler pilot would have taught me more.",
        peer_contribution="careful problem diagnosis, quiet technical help, and willingness to learn from imperfect first versions",
        intended_effect="Initiative-down neighboring case to a quiet technical seed anchor that already shipped useful work.",
        generator_notes="Same practical-tech interest as the anchor, but the visible evidence stops at sketches and tutorial-led preparation.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_018",
        anchor_candidate_id="syn_eng_v1_034",
        anchor_source="synthetic_batch_v1",
        contrast_type="initiative_contrast",
        dimensions_changed=("initiative_level",),
        bucket="medium",
        has_interview=True,
        polish_level="high",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Almaty",
        english_type="IELTS",
        english_score=7.0,
        school_type="Attestat",
        school_score=4.4,
        opening="I used to think communication work mattered mostly at the level of tone, but I now care much more about whether it changes what people actually do.",
        interest_area="youth communication, feedback systems, and peer coordination",
        local_issue="students talk about weak school communication all the time, yet very few channels let them turn complaints into organized feedback",
        action="launched a monthly student listening circle, summarized patterns in a short memo, and helped one vice-principal test a clearer update format for deadlines and club announcements",
        outcome="Attendance stayed modest but steady, and the update format was later used by two teachers.",
        limitation="The work is still small and school-bounded, but it is more real once it moves beyond commentary alone.",
        personal_context="I became more serious about this after seeing how often capable students missed events simply because the information arrived in the wrong form.",
        help_example="I also review short application drafts for younger students who need a first push to submit anything at all.",
        future_goal="build communication systems that change participation, not only mood",
        failed_goal="My first version was an open chat thread, and it failed because complaints appeared faster than anything usable came out of them.",
        worldview_shift="I stopped valuing good framing by itself once I saw that a clear channel can matter more than a strong speech.",
        teamwork_example="I had to work with school staff who preferred less criticism in writing, so I learned to frame patterns in a way that still protected the substance.",
        peer_contribution="clear public communication, practical follow-through, and energy for moving from discussion into usable structure",
        intended_effect="Initiative-up neighboring case to a polished-thin communication anchor.",
        generator_notes="Keeps the same communication domain as the anchor but adds a modest, recurring structure with teacher uptake.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_019",
        anchor_candidate_id="cand_025",
        anchor_source="seed_pack",
        contrast_type="initiative_contrast",
        dimensions_changed=("initiative_level",),
        bucket="long",
        has_interview=True,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="reflective",
        ambiguity_level="clean",
        city="Karaganda",
        english_type="school classes",
        english_score=62,
        school_type="Attestat",
        school_score=4.0,
        opening="Family pressure made me take science more seriously, but it also showed me how easy it is to stay in reflection when energy is limited.",
        interest_area="practical science, household water questions, and useful learning",
        local_issue="children in my area are curious about experiments, yet most science around them feels disconnected from the problems families actually face",
        action="read, collect examples, and explain ideas to my younger brother, but I have not yet built the simple water-testing project I keep imagining",
        outcome="At this point the interest is real, but the initiative is still more planned than proven.",
        limitation="I do not want effort at home to become an excuse forever, yet that is partly what has happened so far.",
        personal_context="When family routines get heavy, I default toward helping and studying rather than risking a project that might fail publicly.",
        help_example="I still enjoy turning science homework into a hands-on explanation for younger children, even when the bigger idea remains unfinished.",
        future_goal="turn care, curiosity, and local science questions into something I actually build with other people",
        failed_goal="I wrote a materials list for a simple testing project twice and never moved from the list to the first trial.",
        worldview_shift="Seeing how quickly students light up around practical science made me more aware of the cost of hesitation.",
        teamwork_example="In a lab assignment, I noticed that I was happy to help with careful steps but hesitant to claim the messy leadership parts.",
        peer_contribution="sincerity, practical curiosity, and willingness to admit where action still has to catch up with intention",
        intended_effect="Initiative-down neighboring case to a stronger science-and-responsibility seed anchor.",
        generator_notes="Same general science-for-daily-life context as the anchor, but the applicant has not yet turned the concern into a self-started project.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_020",
        anchor_candidate_id="syn_eng_v1_039",
        anchor_source="synthetic_batch_v1",
        contrast_type="initiative_contrast",
        dimensions_changed=("initiative_level",),
        bucket="short",
        has_interview=False,
        polish_level="balanced",
        evidence_level="thin",
        initiative_level="reflective",
        ambiguity_level="clean",
        city="a village near Pavlodar",
        english_type="school classes",
        english_score=59,
        school_type="Village school certificate",
        school_score=3.9,
        opening="In a rural school, the gap between being told to use technology and actually having access to it is obvious every day.",
        interest_area="digital access and basic student confidence",
        local_issue="students are expected to type, search, and submit files even when they barely touch a working computer",
        action="listed broken machines for a teacher and talked about opening the old room after lessons, but I did not manage to turn the idea into a regular practice",
        outcome="What I mostly have so far is a clear picture of the problem and a wish to act more confidently.",
        limitation="The case is more about readiness and awareness than about a finished initiative.",
        personal_context="I know I work carefully, but I also know I wait too long before turning an idea into something visible.",
        help_example="I still show classmates small keyboard shortcuts because even basic confidence with tools changes how school feels.",
        future_goal="become the kind of person who acts before a useful idea becomes only another intention",
        failed_goal="I tried to schedule one weekly computer hour, but I let the idea fade once the first conversation became awkward.",
        worldview_shift="Seeing how many students hide behind embarrassment made me take access problems more seriously than before.",
        teamwork_example="I do better once a group is concrete and calm, which is something I still have to create on purpose instead of waiting for.",
        peer_contribution="careful attention, patience with beginners, and a real sense of where basic systems break down",
        intended_effect="Initiative-down neighboring case to a quiet tech-access anchor with real action.",
        generator_notes="Same rural-tech-access context as the anchor, but the visible evidence stays at discussion and inventory rather than reopened access.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_021",
        anchor_candidate_id="syn_eng_v1_045",
        anchor_source="synthetic_batch_v1",
        contrast_type="ambiguity_manual_review_contrast",
        dimensions_changed=("grounding_consistency", "claim_calibration"),
        bucket="long",
        has_interview=True,
        polish_level="high",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Astana",
        english_type="IELTS",
        english_score=7.5,
        school_type="specialized school diploma",
        school_score=4.5,
        opening="Educational access matters to me most when it stops sounding like a national slogan and turns into something a younger student can use next week.",
        interest_area="youth participation, school feedback, and practical educational access",
        local_issue="students from ordinary schools often do not know how to raise concerns or propose improvements in a way adults will actually read",
        action="ran four feedback circles with younger students, summarized recurring issues about information gaps and club access, and brought one short memo to a counselor and two teachers",
        outcome="One teacher later used the memo to adjust club sign-up instructions and the counselor asked me to repeat the circle with a new group.",
        limitation="The work is still school-scale only, and I prefer to be precise about that while it remains early.",
        personal_context="Online learning pushed me to care more about whether institutions actually hear quieter students, not only whether they praise initiative in general terms.",
        help_example="I often review scholarship forms with one younger student who freezes once official language becomes too formal.",
        future_goal="build honest youth feedback systems that do not need inflated claims to feel meaningful",
        failed_goal="My first instinct was to imagine a citywide version, and I had to scale down before the project became real enough to test.",
        worldview_shift="I stopped admiring big youth language once I saw how much one small, readable memo could already change.",
        teamwork_example="I had to work with adults who wanted the memo shorter and less emotional, and the best compromise still kept the practical problems visible.",
        peer_contribution="clear framing, calibrated ambition, and respect for scale when a project is still early",
        intended_effect="Clean, grounded neighboring case to an over-framed youth-consulting anchor.",
        generator_notes="Same broad youth-access ambition as the anchor, but the scale is explicitly school-level and internally consistent.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_022",
        anchor_candidate_id="syn_eng_v1_047",
        anchor_source="synthetic_batch_v1",
        contrast_type="ambiguity_manual_review_contrast",
        dimensions_changed=("grounding_consistency", "claim_calibration"),
        bucket="medium",
        has_interview=True,
        polish_level="high",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Astana",
        english_type="TOEFL",
        english_score=100,
        school_type="specialized diploma",
        school_score=4.6,
        opening="What interests me most about student tools is not saying they will change everything, but making one repeated frustration disappear.",
        interest_area="student coordination tools and practical product design",
        local_issue="deadlines and room changes still get lost between chat threads, especially for students who are not already inside the strongest circles",
        action="built a simple prototype that combined homework reminders and room updates, tested it with twelve classmates and one teacher, and kept notes on what nobody used",
        outcome="Students used the reminder part more than the extra features, which pushed me to simplify the design instead of exaggerating reach.",
        limitation="The product is early and local, and I think that honesty is part of what makes the experiment useful.",
        personal_context="A competitive school environment made me notice how easy it is to talk like a product is bigger than it really is.",
        help_example="I also help classmates set up smaller tools if the barrier is confusion rather than technical difficulty.",
        future_goal="learn how to turn early prototypes into products that stay honest about stage and scope",
        failed_goal="My first version had too many features, and people ignored most of them until I cut the tool down to the one recurring frustration.",
        worldview_shift="I became more interested in product truthfulness once I saw that a smaller honest pilot teaches more than a grand claim.",
        teamwork_example="I worked with a friend who wanted the app to sound more ambitious, and I kept arguing that accurate testing data mattered more than a louder pitch.",
        peer_contribution="product realism, structured iteration, and willingness to learn from what users ignore",
        intended_effect="Clean, calibrated neighboring case to an overstated student-app anchor.",
        generator_notes="Same student-tool ambition as the anchor, but with explicit scale control and no mismatch between claim and evidence.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_023",
        anchor_candidate_id="cand_024",
        anchor_source="seed_pack",
        contrast_type="ambiguity_manual_review_contrast",
        dimensions_changed=("grounding_consistency",),
        bucket="long",
        has_interview=True,
        polish_level="balanced",
        evidence_level="moderate",
        initiative_level="acted",
        ambiguity_level="clean",
        city="Ust-Kamenogorsk",
        english_type="IELTS",
        english_score=6.8,
        school_type="Kazakhstan certificate",
        school_score=4.3,
        opening="Changing schools after moving from Semey to Ust-Kamenogorsk made me notice how environmental problems and social adjustment can become connected very quickly.",
        interest_area="environmental monitoring, adaptation, and school-community projects",
        local_issue="students talk about smoke, litter, and river pollution, but new students often stay too quiet to join anything that could change those habits",
        action="joined one river cleanup during my first term, later helped build a small air-quality notice board with readings from a borrowed device, and kept a short blog so new students could follow what the project was doing",
        outcome="The board stayed in one corridor and the blog reached only a small group, but the timeline is clear and the work grew step by step after the move.",
        limitation="I am still early and would like much stronger methods, but I can describe the scale and sequence honestly.",
        personal_context="The school move made me cautious for a while, which is partly why environmental work became a way to belong through usefulness.",
        help_example="I still help one newer student navigate small routines because feeling lost socially can silence real ability.",
        future_goal="connect environmental attention with steadier project design so quieter students can join without needing to become loud first",
        failed_goal="I wanted the blog to include more student voices, but I did not yet build a routine that made contributions easy.",
        worldview_shift="I learned that a small project can be more trustworthy once the sequence is clear and the claims stay proportional.",
        teamwork_example="In the cleanup group, I had to accept that consistency mattered more than doing every environmental topic at once.",
        peer_contribution="honest scaling, step-by-step coordination, and sensitivity to how adaptation affects participation",
        intended_effect="Clean, consistent neighboring case to a seed anchor that had real signal but timeline ambiguity.",
        generator_notes="Same broad move-and-environment theme as the anchor, but removes internal timing confusion and keeps scope modest.",
    ),
    ContrastSpec(
        candidate_id="syn_contrast_v2_024",
        anchor_candidate_id="cand_017",
        anchor_source="seed_pack",
        contrast_type="ambiguity_manual_review_contrast",
        dimensions_changed=("grounding_consistency", "claim_calibration"),
        bucket="short",
        has_interview=False,
        polish_level="high",
        evidence_level="thin",
        initiative_level="acted",
        ambiguity_level="borderline",
        city="Almaty",
        english_type="IELTS",
        english_score=7.5,
        school_type="specialized school diploma",
        school_score=4.6,
        opening="I am drawn to civic technology because students in regional cities need platforms that help them move from frustration to organized action.",
        interest_area="civic tech, student coordination, and local air-quality awareness",
        local_issue="many students care about air pollution and transport inequity, but most school discussions stay reactive and fragmented",
        action="described the work as a regional student platform and air-monitoring network, while the visible version so far is a small survey, one prototype page, and a pilot discussion at school",
        outcome="The ambition sounds larger than the current proof, even though the underlying concern itself is plausible.",
        limitation="I can feel the project idea pulling ahead of what I have actually demonstrated.",
        personal_context="A highly ambitious school environment made me more aware of how easily image can outrun evidence if nobody pushes back early.",
        help_example="I still care enough about the issue to keep refining the idea, which is part of why I want harder feedback and not only encouragement.",
        future_goal="build civic tools that are more honest about stage, proof, and real public use",
        failed_goal="I spent months trying to make the platform look region-ready before I had a school-level version people truly used.",
        worldview_shift="The idea stopped feeling purely exciting once I saw how quickly strong framing can hide weak validation.",
        teamwork_example="I had to work with classmates who wanted the pitch to sound city-scale from the beginning, and I did not resist that pressure clearly enough.",
        peer_contribution="big-picture energy, civic motivation, and growing awareness that evidence has to catch up with narrative",
        intended_effect="Borderline ambiguity neighbor to a clean, stronger civic-tech seed anchor.",
        generator_notes="Keeps a similar civic-tech ambition as the seed anchor but introduces claim inflation and grounding tension without turning the case into parody.",
    ),
]


def normalize_space(text: str) -> str:
    text = text.strip().replace("\r\n", "\n")
    paragraphs = [" ".join(part.split()) for part in text.split("\n\n")]
    return "\n\n".join(part for part in paragraphs if part)


def sanitize_visible_text(text: str) -> str:
    return normalize_space(text)


def total_word_count(record: dict[str, object]) -> int:
    text_inputs = record["text_inputs"]
    parts: list[str] = [text_inputs["motivation_letter_text"]]
    for item in text_inputs["motivation_questions"]:
        parts.append(item["answer"])
    parts.append(text_inputs.get("interview_text") or "")
    return len(re.findall(r"\b\w+\b", " ".join(parts)))


def length_bucket_from_words(words: int) -> str:
    if words < 400:
        return "short"
    if words <= 550:
        return "medium"
    return "long"


def contrast_family(spec: ContrastSpec) -> str:
    if spec.contrast_type == "evidence_contrast":
        return "evidence"
    if spec.contrast_type == "polish_contrast":
        return "polish"
    if spec.contrast_type == "initiative_contrast":
        return "initiative"
    return "ambiguity"


def close_sentence(spec: ContrastSpec) -> str:
    family = contrast_family(spec)
    if family == "evidence":
        return "I want a place where evidence, iteration, and public usefulness matter at the same time."
    if family == "polish":
        return "I want critique that tests whether the substance underneath my words is strong enough to last."
    if family == "initiative":
        return "I want a harder environment where concern has to become action and not remain private conviction only."
    if spec.ambiguity_level == "clean":
        return "I would rather build something modest and trustworthy than something that sounds larger than it is."
    return "I want a demanding environment that forces my claims and my proof to meet each other more honestly."


def routine_text(spec: ContrastSpec) -> str:
    family = contrast_family(spec)
    if family in {"evidence", "initiative"} and "technology" in spec.interest_area:
        return "I keep a small notebook of recurring problems and return to it at the end of the week so I do not confuse ideas with evidence."
    if family in {"evidence", "initiative"}:
        return "I try to write down concrete observations the same day I notice them, because otherwise concern becomes general and less useful."
    if family == "polish" and spec.polish_level == "high":
        return "I usually begin the day by picking a few priorities and end it by checking whether I was genuinely useful or only busy."
    if family == "polish":
        return "I work best when I keep one short list and finish the visible part first instead of letting everything stay in my head."
    if spec.ambiguity_level == "clean":
        return "I now review what I can actually prove before I describe a project, because scale is easy to imagine and harder to earn."
    return "I have learned that strong framing can become dangerous if I do not stop and ask what has really happened so far."


def growth_edge(spec: ContrastSpec) -> str:
    if contrast_family(spec) == "initiative":
        return "moving from private concern into visible action earlier"
    if contrast_family(spec) == "polish" and spec.polish_level == "high":
        return "taking more open-ended risks instead of relying on careful framing"
    if contrast_family(spec) == "polish":
        return "speaking about my work with more confidence before everything feels finished"
    if contrast_family(spec) == "evidence":
        return "building cleaner systems and documenting results more consistently"
    if spec.ambiguity_level == "clean":
        return "scaling carefully without losing honesty about what is still early"
    return "keeping ambition in proportion to proof"


def project_learning(spec: ContrastSpec) -> str:
    family = contrast_family(spec)
    if family == "initiative" and spec.initiative_level == "reflective":
        return "What I learned is that insight without visible follow-through still leaves the core problem untouched."
    if family == "ambiguity" and spec.ambiguity_level == "borderline":
        return "The experience made me more aware that strong language can outrun what a project has really demonstrated."
    if contrast_family(spec) == "polish":
        return "That experience made me think harder about the difference between sounding prepared and building something that survives use."
    return "That experience taught me that even a small test can reveal what a bigger idea is missing."


def local_issue_root_text(spec: ContrastSpec) -> str:
    if contrast_family(spec) == "initiative":
        return "the problem keeps waiting for somebody to move first, and too often I have still been standing at the observation stage"
    if contrast_family(spec) == "polish":
        return "students often get more image than structure, so useful ideas look bigger than the systems underneath them"
    if contrast_family(spec) == "ambiguity":
        return "ambition becomes hard to evaluate when language grows faster than evidence"
    return "ordinary friction becomes normal when nobody has time or structure to address it"


def letter_for_spec(spec: ContrastSpec) -> str:
    family = contrast_family(spec)
    if family == "evidence":
        intro = (
            f"{spec.opening} I am drawn to {spec.interest_area} because {spec.local_issue}."
        )
        action = (
            f"I responded by {spec.action}. {spec.outcome} {spec.limitation}"
        )
    elif family == "polish":
        intro = (
            f"{spec.opening} What keeps me engaged is {spec.local_issue}, especially when it touches {spec.interest_area}."
        )
        if spec.polish_level == "high":
            action = f"The clearest example behind this interest is that I {spec.action}. {spec.outcome} {spec.limitation}"
        else:
            action = f"I tried one useful response here. I {spec.action}. {spec.outcome} {spec.limitation}"
    elif family == "initiative":
        intro = (
            f"{spec.opening} The question I keep circling is {spec.local_issue}."
        )
        if spec.initiative_level == "reflective":
            action = f"So far, I {spec.action}. {spec.outcome} {spec.limitation}"
        else:
            action = f"I decided not to leave this only as an idea, so I {spec.action}. {spec.outcome} {spec.limitation}"
    else:
        intro = (
            f"{spec.opening} What keeps me engaged is {spec.local_issue}, and I keep returning to {spec.interest_area} as one possible response."
        )
        if spec.ambiguity_level == "clean":
            action = f"The example I can stand behind most clearly is that I {spec.action}. {spec.outcome} {spec.limitation}"
        else:
            action = f"The project I describe most often is that I {spec.action}. {spec.outcome} {spec.limitation}"

    context = f"{spec.personal_context} {spec.help_example} {spec.worldview_shift}"
    close = f"At inVision U, I want to {spec.future_goal}. {close_sentence(spec)}"

    paragraphs = [intro, action]
    if spec.bucket != "short":
        paragraphs.append(context)
    if spec.bucket == "long":
        paragraphs.append(spec.failed_goal)
    paragraphs.append(close)
    return sanitize_visible_text("\n\n".join(paragraphs))


def question_ids_for_spec(spec: ContrastSpec) -> list[int]:
    family = contrast_family(spec)
    if family == "evidence":
        mapping = {
            "short": [1, 2, 9],
            "medium": [1, 2, 6, 9],
            "long": [1, 2, 3, 5, 9],
        }
    elif family == "polish":
        mapping = {
            "short": [1, 4, 6],
            "medium": [1, 4, 6, 8],
            "long": [1, 4, 5, 6, 8],
        }
    elif family == "initiative":
        mapping = {
            "short": [2, 8, 10],
            "medium": [1, 2, 8, 10],
            "long": [1, 2, 3, 8, 10],
        }
    else:
        mapping = {
            "short": [1, 7, 9],
            "medium": [1, 7, 9, 11],
            "long": [1, 2, 5, 7, 9],
        }
    return mapping[spec.bucket]


def answer_for_question(spec: ContrastSpec, question_id: int) -> str:
    if question_id == 1:
        return sanitize_visible_text(
            f"I kept noticing that {spec.local_issue}. I responded by {spec.action}. {spec.outcome} {project_learning(spec)}"
        )
    if question_id == 2:
        return sanitize_visible_text(
            f"I would start with {spec.local_issue}. I know this issue closely because it appears in daily school and family routines where I live, and I can imagine a first version that is small and testable rather than only impressive."
        )
    if question_id == 3:
        return sanitize_visible_text(
            f"{spec.personal_context} That period made me more deliberate about time, responsibility, and what I can control when circumstances are not calm."
        )
    if question_id == 4:
        return sanitize_visible_text(
            f"In a team, I would contribute {spec.peer_contribution}. I also think I help a group stay honest about what the work is really for."
        )
    if question_id == 5:
        return sanitize_visible_text(
            f"{spec.teamwork_example} What made it difficult was that other people were often optimizing for speed, image, or comfort while I was focused on fit. I learned that disagreement becomes easier once the purpose of the work is concrete."
        )
    if question_id == 6:
        return sanitize_visible_text(
            f"{routine_text(spec)} That routine matters because it keeps me from confusing effort, intention, and proof."
        )
    if question_id == 7:
        return sanitize_visible_text(
            f"{spec.failed_goal} What stayed with me is a better sense of scale, patience, and what kind of preparation actually matters."
        )
    if question_id == 8:
        return sanitize_visible_text(
            f"I do my best work in environments that value substance, honest feedback, and room to iterate. The part of myself that still needs development is {growth_edge(spec)}."
        )
    if question_id == 9:
        return sanitize_visible_text(
            f"A clear example is when I {spec.action}. {spec.outcome} It stayed smaller than I hoped because {spec.limitation.lower()}"
        )
    if question_id == 10:
        return sanitize_visible_text(
            f"What stays on my mind is that {spec.local_issue}. I think the root problem is that {local_issue_root_text(spec)}."
        )
    if question_id == 11:
        return sanitize_visible_text(
            f"Feedback changed me most when {spec.failed_goal.lower()} After that, I became more careful about calibration, proof, and the difference between a promising idea and a validated one."
        )
    raise ValueError(f"Unsupported question id: {question_id}")


def interview_opening(spec: ContrastSpec) -> str:
    family = contrast_family(spec)
    suffix = int(spec.candidate_id.rsplit("_", 1)[-1])
    if family == "evidence":
        variants = [
            "I usually understand a problem best once I can test something against reality instead of discussing it only in abstract terms.",
            "My thinking becomes much clearer when an idea has to survive contact with an actual user, object, or routine.",
            "I trust my own judgment most when a real constraint pushes back against the first version of an idea.",
        ]
        return variants[suffix % len(variants)]
    if family == "polish" and spec.polish_level == "high":
        variants = [
            "I care a lot about whether the substance under an idea is strong enough to survive careful explanation.",
            "Good communication matters to me most when it is strong enough to carry real work rather than decorate it.",
            "I have become more interested in how clearly an idea can be explained without losing the limits that keep it honest.",
        ]
        return variants[suffix % len(variants)]
    if family == "polish":
        return "I do not always sound polished in a new room, but I usually become more useful once the task is concrete."
    if family == "initiative":
        variants = [
            "What I am trying to learn now is how to close the distance between concern and visible action.",
            "The main shift I want in myself is from noticing problems early to acting on them before hesitation takes over.",
        ]
        return variants[suffix % len(variants)]
    if spec.ambiguity_level == "clean":
        variants = [
            "I have become more careful about describing a project at the scale it has actually earned.",
            "I now pay close attention to the difference between a promising early project and a result that is already proven.",
            "Being precise about scope has become important to me, especially when a project is still smaller than the ambition behind it.",
        ]
        return variants[suffix % len(variants)]
    return "I am ambitious about what these ideas could become, but I know that ambition needs much tighter proof than it has right now."


def interview_for_spec(spec: ContrastSpec) -> str:
    if not spec.has_interview:
        return ""
    second = {
        "evidence": "What I want from inVision U is stronger iteration, clearer critique, and a community that expects useful evidence rather than general intention.",
        "polish": "I want to learn in a place where communication matters, but where communication still has to answer to execution and not replace it.",
        "initiative": "I do not need an easier environment. I need one that makes action more normal than hesitation and gives feedback before doubt hardens into habit.",
        "ambiguity": "The kind of growth I want now is more rigor around scale, truthfulness, and what impact actually looks like before it becomes a larger story.",
    }[contrast_family(spec)]
    third = f"I think I would contribute {spec.peer_contribution}, especially in work that asks people to stay honest about what a project is actually doing."
    if spec.bucket == "long":
        return sanitize_visible_text("\n\n".join([interview_opening(spec), second, third]))
    return sanitize_visible_text("\n\n".join([interview_opening(spec), second]))


def behavioral_signals_for_spec(spec: ContrastSpec) -> dict[str, object]:
    completion = {"short": 0.86, "medium": 0.92, "long": 0.96}[spec.bucket]
    if spec.polish_level == "high":
        completion += 0.02
    if contrast_family(spec) == "initiative" and spec.initiative_level == "reflective":
        completion -= 0.02
    if spec.ambiguity_level == "borderline":
        completion += 0.01
    completion = round(min(0.98, max(0.84, completion)), 2)
    skipped = {"short": 2, "medium": 1, "long": 0}[spec.bucket]
    if contrast_family(spec) == "initiative" and spec.initiative_level == "reflective":
        skipped += 1
    returned = spec.has_interview or spec.polish_level == "high" or spec.ambiguity_level == "borderline"
    return {
        "completion_rate": completion,
        "returned_to_edit": returned,
        "skipped_optional_questions": skipped,
    }


def generate_candidate(spec: ContrastSpec, submitted_at: str) -> tuple[dict[str, object], dict[str, object]]:
    motivation_questions = [
        {"question": QUESTION_BANK[qid], "answer": answer_for_question(spec, qid)}
        for qid in question_ids_for_spec(spec)
    ]
    raw = {
        "candidate_id": spec.candidate_id,
        "structured_data": {
            "education": {
                "english_proficiency": {"type": spec.english_type, "score": spec.english_score},
                "school_certificate": {"type": spec.school_type, "score": spec.school_score},
            }
        },
        "text_inputs": {
            "motivation_letter_text": letter_for_spec(spec),
            "motivation_questions": motivation_questions,
            "interview_text": interview_for_spec(spec),
        },
        "behavioral_signals": behavioral_signals_for_spec(spec),
        "metadata": {
            "source": "contrastive_batch_v2",
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
    return raw, sanitized


def reviewer_table_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        education = record["structured_data"]["education"]
        text_inputs = record["text_inputs"]
        answers = text_inputs["motivation_questions"]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "english_type": education["english_proficiency"]["type"],
                "english_score": education["english_proficiency"]["score"],
                "school_certificate_type": education["school_certificate"]["type"],
                "school_certificate_score": education["school_certificate"]["score"],
                "completion_rate": record["behavioral_signals"]["completion_rate"],
                "returned_to_edit": record["behavioral_signals"]["returned_to_edit"],
                "skipped_optional_questions": record["behavioral_signals"]["skipped_optional_questions"],
                "meaningful_answers_count": len([a for a in answers if a["answer"].strip()]),
                "scenario_depth": "",
                "has_interview_text": bool(text_inputs.get("interview_text")),
                "motivation_question_count": len(answers),
                "total_text_word_count": total_word_count(record),
                "motivation_letter_text": text_inputs["motivation_letter_text"],
                "motivation_questions_text": "\n\n".join(
                    f"Q: {item['question']}\nA: {item['answer']}" for item in answers
                ),
                "interview_text": text_inputs.get("interview_text") or "",
            }
        )
    return rows


def validate_near_duplicates(records: list[dict[str, object]]) -> list[tuple[str, str, float]]:
    issues: list[tuple[str, str, float]] = []
    for i, left in enumerate(records):
        left_text = left["text_inputs"]["motivation_letter_text"]
        for right in records[i + 1 :]:
            ratio = SequenceMatcher(
                None,
                re.sub(r"\s+", " ", left_text.lower()),
                re.sub(r"\s+", " ", right["text_inputs"]["motivation_letter_text"].lower()),
            ).ratio()
            if ratio >= 0.82:
                issues.append((left["candidate_id"], right["candidate_id"], round(ratio, 3)))
    return issues


VISIBLE_LEAKAGE_PATTERNS: dict[str, str] = {
    "contrast_word": r"\bcontrast(?:ive)?\b",
    "anchor_word": r"\banchor\b",
    "generator_word": r"\bgenerator\b",
    "rubric_label_word": r"\bhidden potential\b",
    "manual_review_word": r"\bmanual review\b",
}


def visible_text_fields(record: dict[str, object]) -> list[str]:
    text_inputs = record["text_inputs"]
    fields = [text_inputs.get("motivation_letter_text") or "", text_inputs.get("interview_text") or ""]
    fields.extend((item.get("answer") or "") for item in text_inputs.get("motivation_questions") or [])
    return [field for field in fields if field]


def visible_leakage_hits(records: list[dict[str, object]]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for label, pattern in VISIBLE_LEAKAGE_PATTERNS.items():
        regex = re.compile(pattern, re.IGNORECASE)
        matched = [record["candidate_id"] for record in records if any(regex.search(field) for field in visible_text_fields(record))]
        if matched:
            hits[label] = matched
    return hits


def person_drift_hits(records: list[dict[str, object]]) -> list[str]:
    pattern = re.compile(r"\b(?:he|she|his|her|him|herself|himself)\b", re.IGNORECASE)
    hits: list[str] = []
    for record in records:
        if any(pattern.search(field) for field in visible_text_fields(record)):
            hits.append(record["candidate_id"])
    return hits


def repeated_opening_hits(records: list[dict[str, object]], threshold: int = 2) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for record in records:
        letter = (record["text_inputs"].get("motivation_letter_text") or "").strip()
        opening = re.split(r"(?<=[.!?])\s+", letter)[0].strip()
        if opening:
            buckets.setdefault(opening, []).append(record["candidate_id"])
    return {opening: ids for opening, ids in buckets.items() if len(ids) > threshold}


def repeated_interview_opening_hits(records: list[dict[str, object]], threshold: int = 3) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for record in records:
        interview = (record["text_inputs"].get("interview_text") or "").strip()
        if not interview:
            continue
        opening = re.split(r"(?<=[.!?])\s+", interview)[0].strip()
        if opening:
            buckets.setdefault(opening, []).append(record["candidate_id"])
    return {opening: ids for opening, ids in buckets.items() if len(ids) > threshold}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2026, 4, 2, 1, 30, tzinfo=timezone.utc)
    raw_records: list[dict[str, object]] = []
    sanitized_records: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []

    for idx, spec in enumerate(SPECS):
        submitted_at = (base_time + timedelta(hours=7 * idx)).isoformat().replace("+00:00", "Z")
        raw, sanitized = generate_candidate(spec, submitted_at)
        raw = CandidateInput.model_validate(raw).model_dump(mode="json", exclude_none=False)
        CandidateInput.model_validate(sanitized)
        raw_records.append(raw)
        sanitized_records.append(sanitized)
        manifest_records.append(
            {
                "candidate_id": spec.candidate_id,
                "anchor_candidate_id": spec.anchor_candidate_id,
                "anchor_source": spec.anchor_source,
                "contrast_type": spec.contrast_type,
                "dimensions_changed": list(spec.dimensions_changed),
                "intended_effect": spec.intended_effect,
                "generator_notes": spec.generator_notes,
            }
        )

    ids = [record["candidate_id"] for record in raw_records]
    raw_to_sanitized_ids = [record["candidate_id"] for record in sanitized_records]
    near_duplicate_pairs = validate_near_duplicates(sanitized_records)
    visible_leakage = visible_leakage_hits(sanitized_records)
    person_drift = person_drift_hits(sanitized_records)
    repeated_letter_openings = repeated_opening_hits(sanitized_records)
    repeated_interview_openings = repeated_interview_opening_hits(sanitized_records)
    sanitized_payload_text = json.dumps(sanitized_records, ensure_ascii=False)

    contrast_type_counts = Counter(spec.contrast_type for spec in SPECS)
    dimensions_changed_counts = Counter(dim for spec in SPECS for dim in spec.dimensions_changed)
    text_length_counts = Counter(length_bucket_from_words(total_word_count(record)) for record in sanitized_records)
    with_interview_count = sum(1 for spec in SPECS if spec.has_interview)
    without_interview_count = len(SPECS) - with_interview_count
    anchor_sources_used = sorted({spec.anchor_source for spec in SPECS})

    leakage_terms = [
        spec.anchor_candidate_id for spec in SPECS
    ] + [
        "contrastive_batch_v2_generation_manifest",
        "evidence_contrast",
        "polish_contrast",
        "initiative_contrast",
        "ambiguity_manual_review_contrast",
        "evidence_strength",
        "presentation_polish",
        "initiative_level",
        "grounding_consistency",
        "claim_calibration",
        "anchor_source",
    ]

    validation_status = {
        "candidate_input_schema_raw": True,
        "candidate_input_schema_sanitized": True,
        "unique_candidate_ids": len(ids) == len(set(ids)),
        "raw_sanitized_one_to_one": ids == raw_to_sanitized_ids,
        "sanitized_has_no_metadata": all("metadata" not in item for item in sanitized_records),
        "hidden_manifest_leakage_absent": not any(term in sanitized_payload_text for term in leakage_terms),
        "english_only_heuristic": not re.search(r"[А-Яа-яЁё]", sanitized_payload_text),
        "person_drift_hits": person_drift,
        "visible_leakage_hits": visible_leakage,
        "repeated_letter_openings_over_threshold": repeated_letter_openings,
        "repeated_interview_openings_over_threshold": repeated_interview_openings,
        "near_duplicate_pairs": near_duplicate_pairs,
        "text_length_target_met": dict(text_length_counts) == {"short": 6, "medium": 10, "long": 8},
        "interview_target_met": 16 <= with_interview_count <= 18,
        "anchor_sources_used": anchor_sources_used,
    }

    summary = {
        "candidate_count": len(SPECS),
        "contrast_type_counts": dict(contrast_type_counts),
        "dimensions_changed_counts": dict(dimensions_changed_counts),
        "text_length_counts": dict(text_length_counts),
        "with_interview_count": with_interview_count,
        "without_interview_count": without_interview_count,
        "validation_status": validation_status,
        "notes": [
            "Visible payloads remain inside the frozen public CandidateInput contract.",
            "Reviewer pack removes metadata and keeps only candidate_id, structured_data, text_inputs, and behavioral_signals.",
            "Text length buckets are based on total visible word count with thresholds short < 400, medium <= 550, long > 550.",
            f"Anchor sources used: {', '.join(anchor_sources_used)}.",
        ],
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
                "english_type",
                "english_score",
                "school_certificate_type",
                "school_certificate_score",
                "completion_rate",
                "returned_to_edit",
                "skipped_optional_questions",
                "meaningful_answers_count",
                "scenario_depth",
                "has_interview_text",
                "motivation_question_count",
                "total_text_word_count",
                "motivation_letter_text",
                "motivation_questions_text",
                "interview_text",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    pack_manifest = {
        "source_file": str(RAW_JSONL),
        "output_file": str(PACK_JSONL),
        "pretty_output_file": str(PACK_JSON),
        "table_output_file": str(PACK_TABLE_CSV),
        "candidate_count": len(sanitized_records),
        "sanitization_rule": "remove metadata to reduce synthetic source leakage; keep only candidate_id, structured_data, text_inputs, behavioral_signals",
        "intended_use": "contrastive admissions batch v2 for human annotation and scorer stress-testing",
    }
    with PACK_MANIFEST_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(pack_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
