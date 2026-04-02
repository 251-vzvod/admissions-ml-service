from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.input import CandidateInput
RAW_DIR = ROOT / "data" / "ml_workbench" / "raw" / "generated" / "batch_v1"
PACK_DIR = ROOT / "data" / "ml_workbench" / "processed" / "annotation_packs" / "synthetic_batch_v1"

RAW_JSONL = RAW_DIR / "synthetic_batch_v1_api_input.jsonl"
GEN_MANIFEST_JSONL = RAW_DIR / "synthetic_batch_v1_generation_manifest.jsonl"
SUMMARY_JSON = RAW_DIR / "synthetic_batch_v1_summary.json"

PACK_JSONL = PACK_DIR / "synthetic_batch_v1_annotation_pack.jsonl"
PACK_JSON = PACK_DIR / "synthetic_batch_v1_annotation_pack.json"
PACK_TABLE_CSV = PACK_DIR / "synthetic_batch_v1_annotation_pack_table.csv"
PACK_MANIFEST_JSON = PACK_DIR / "synthetic_batch_v1_annotation_pack_manifest.json"


QUESTION_BANK = {
    1: "Tell us about the most difficult period in your life - what happened, how did you cope, and what changed in you afterwards?",
    2: "What habits or daily routines have you deliberately built for yourself, and how do they impact your results? Why did you choose specifically those?",
    3: "Was there a time when you worked toward a goal for a long time but never reached it? What did you take away from that experience?",
    4: "Describe a project, idea, or solution you came up with entirely on your own initiative - not as an assignment. Where did the idea come from, and what was the outcome?",
    5: "How do you usually make important decisions - do you rely on intuition, data, or advice from others? Walk us through a specific real example from your life.",
    6: "Have you ever faced a situation where you had to choose between what was efficient and what was right? What did you do, and why?",
    7: "Is there a problem in your city, country, or the world that genuinely troubles you? Why that one specifically - and what do you think is its root cause?",
    8: "How has your worldview shifted over the last 2-3 years? What drove that change - a book, a person, an event, an experience?",
    9: "What do you think Kazakhstan will look like in 20 years - and what role do you believe your generation will play in shaping it?",
    10: "Recall a moment when you took a stance or did something that was unpopular among the people around you. What drove you to do it?",
    11: "Tell us about a person or community you helped - not because you had to, but because you genuinely wanted to. What did that experience give you?",
    12: "What are your three core values - and can you give one real example from your life for each, showing how it actually showed up in what you did?",
    13: "Have you ever organized, launched, or promoted something - an event, an initiative, a small business, a club? What worked, what did not, and what would you do differently?",
    14: "How have you earned money or created something of value before applying to university? Walk us through the idea, the process, and the result.",
    15: "If you were given $1,000 and one month to do something meaningful - not necessarily to make a profit, but to create real value - what would you do and why?",
    16: "Which inVision U value feels most natural to you already, and which one would challenge you most right now? Explain with a real example.",
    17: "Describe a time you had to work with people who thought very differently from you. What made that hard, and what did you learn?",
    18: "When pressure rises in a team, what kind of teammate do you become? Give one concrete situation.",
    19: "If you joined inVision U tomorrow, what local problem would you want to turn into a semester project first, and why that one?",
    20: "What would your future classmates at inVision U learn from your background that they might not expect at first?",
    21: "Tell us about a time when someone challenged your assumptions and you actually changed your mind. What changed in your behavior after that?",
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    archetype: str
    ambiguity: str
    bucket: str
    has_interview: bool
    city: str
    english_type: str
    english_score: float
    school_type: str
    school_score: float
    focus: str
    issue: str
    initiative: str
    initiative_outcome: str
    hardship: str
    help_example: str
    future_goal: str
    failed_goal: str
    worldview_trigger: str
    value_creation: str
    ethics_case: str
    support_need: str
    risk_note: str


SPECS: list[CandidateSpec] = [
    CandidateSpec(
        "syn_eng_v1_001",
        "hidden_potential_low_polish",
        "clear",
        "short",
        False,
        "Zhezkazgan",
        "school classes",
        61,
        "Kazakhstan high school diploma",
        74,
        "practical learning and helping younger students stay interested in school",
        "many children in my building go home to empty apartments and have nowhere quiet to study",
        "set up a shared homework table in the corridor of her apartment building with old notebooks and borrowed lamps",
        "within two months six children were using it in the evenings, and two parents later brought extra pens and school notebooks",
        "her mother works late shifts at a pharmacy, so she often cooks and checks homework for her younger brother before doing her own assignments",
        "often carries groceries for an elderly neighbor and stays to read short stories with her granddaughter",
        "learn how to turn small neighborhood study corners into something more organized and reliable",
        "tried for a district scholarship but missed the cutoff because her school preparation was weaker than students from larger cities",
        "watching her younger brother fall behind during one winter term made her stop waiting for adults to notice simple problems",
        "sometimes helps younger children with basic math for free and once got paid a little for sorting notebooks and supplies in a kiosk",
        "refused to give a classmate copied homework answers and instead explained the task after school",
        "simple English, low exposure to formal projects, and little experience speaking in groups",
        "real initiative is present, but the application is modest and under-polished",
    ),
    CandidateSpec(
        "syn_eng_v1_002",
        "hidden_potential_low_polish",
        "clear",
        "medium",
        True,
        "Turkistan",
        "school + online videos",
        67,
        "Kazakhstan school certificate",
        79,
        "low-cost problem solving and making practical things useful for families",
        "younger teenagers in his area have little to do after school and often borrow unsafe old bicycles",
        "started a small bicycle repair corner with his grandfather and then passed repaired bikes to children in the neighborhood",
        "four bicycles became usable again, and one younger student was able to stop walking forty minutes each way to extra classes",
        "his father works seasonally away from home, so he helps his grandmother with errands and often manages younger cousins in the evenings",
        "helps an older man on the same street fix loose door handles and carry heavy bags from the market",
        "come back with stronger engineering and project skills so he can run a small repair workshop that also teaches younger students",
        "spent a year trying to win a city technical contest but only reached the first review round because his prototype was rough and badly presented",
        "seeing how much a repaired bike changed one boy's daily routine made him think small tools can change opportunities",
        "earns small payments for repairing brakes or chains and sometimes trades repair work for spare parts",
        "stayed after a school cleanup to repair a broken rack instead of leaving early with his friends",
        "quiet presentation style, uneven English, and very limited formal mentoring",
        "substantive practical signal is there even though the candidate describes it in plain language",
    ),
    CandidateSpec(
        "syn_eng_v1_003",
        "hidden_potential_low_polish",
        "borderline",
        "medium",
        True,
        "Kokshetau",
        "school classes",
        63,
        "Kazakhstan certificate",
        77,
        "education access and simple systems that help classmates share resources",
        "students in her class often miss materials because home internet is unstable and group chats become chaotic",
        "built a rotating paper folder and simple borrowing log so classmates could share printed notes, worksheets, and summaries",
        "the folder lasted for a term and became especially useful before exams, though it depended on her reminding everyone to return materials",
        "she helps in her family's kiosk before and after school and often studies late after closing time",
        "spends part of each weekend helping her cousin practice English speaking because the cousin is too shy to speak in class",
        "learn how to make small peer-support systems last even when one person cannot keep carrying everything alone",
        "wanted to raise her English score enough for a summer camp but missed it by a small margin after two attempts",
        "moving from feeling embarrassed about weak English to openly asking teachers for help changed how she sees learning",
        "earned a little money by arranging shelves and keeping inventory in the kiosk, and by printing and stapling study packets for classmates",
        "refused to let a group presentation use copied slides from the internet even though it made the team late",
        "would need confidence-building and more structure in order to scale anything beyond a class-level system",
        "the evidence is credible but still somewhat fragile because the initiative depended heavily on her personal reminders",
    ),
    CandidateSpec(
        "syn_eng_v1_004",
        "hidden_potential_low_polish",
        "borderline",
        "long",
        True,
        "a village near Pavlodar",
        "school classes",
        58,
        "Rural school certificate",
        4.0,
        "community problem solving and basic public usefulness in small places",
        "when the village water pipe fails, older people struggle first because the well is far and transport is limited",
        "organized a bucket rota with friends during a two-day water outage and later pushed the same group to clean and repair a neglected playground area",
        "three households received regular help with water, and the later cleanup convinced two parents to donate paint and wood for the playground fence",
        "his mother was sick for most of one winter, so he had to manage firewood, cooking, and looking after his younger sister before school",
        "regularly checks in on his grandmother and brings her water, groceries, and medicine from the shop",
        "study somewhere that teaches project planning, because in his village he usually starts by doing and only later realizes what structure was missing",
        "studied for a city scholarship exam and missed it, partly because he had no tutor and lost study time to family duties",
        "after watching neighbors quietly accept broken services for years, he started thinking that waiting politely is not the same as solving a problem",
        "sometimes gets paid for carrying heavy items, clearing snow, and helping older neighbors with yard work",
        "went to school instead of skipping class with friends to go fishing, even after they called him boring",
        "would need strong onboarding in language, academic confidence, and formal teamwork",
        "there is real evidence of reliability and initiative, but the candidate still lacks academic proof and polished explanation",
    ),
    CandidateSpec(
        "syn_eng_v1_005",
        "hidden_potential_low_polish",
        "borderline",
        "medium",
        True,
        "Semey",
        "school + online reading",
        69,
        "Attestat",
        3.9,
        "reading access and small community learning spaces",
        "children in her apartment block spend long evenings outside because there is no library branch or club nearby",
        "turned an unused shelf near the building entrance into a borrowed-book corner and started a Saturday reading hour for younger children",
        "the shelf kept changing as neighbors donated books, and the reading hour never became large but stayed steady for several months",
        "she often stays home with her younger sister while her father drives a taxi at night and her mother cleans offices",
        "visits an elderly neighbor who lives alone and reads newspaper headlines to her because the neighbor's eyesight is weak",
        "build after-school spaces that feel normal and welcoming instead of formal or expensive",
        "tried to organize a larger holiday reading event but only a few parents responded and she had to scale it down",
        "seeing parents thank her for doing something simple without asking permission first made her think small acts can make public space feel different",
        "has done a little paid tutoring in Russian reading and sometimes helps her aunt sort stock at a clothing stall",
        "insisted that a quieter child be included in a group activity even though others said it would slow everything down",
        "speaks honestly but still undersells her own contribution and would likely need confidence support",
        "good community signal is present, though the scale of the initiative stayed local and informal",
    ),
    CandidateSpec(
        "syn_eng_v1_006",
        "hidden_potential_low_polish",
        "borderline",
        "long",
        True,
        "Aksu",
        "school classes",
        60,
        "Kazakhstan high school diploma",
        76,
        "waste reduction and practical neighborhood organizing",
        "students at his school throw away usable clothes, notebooks, and bottles because there is no habit of sharing or sorting things",
        "started a hallway recycling corner and a simple clothes exchange day with one teacher and two classmates",
        "the system never became official, but for one term it collected bags of notebooks and several families took winter clothes they really needed",
        "after his uncle injured his leg, he helped cover shifts in the family stall and learned to track money and stock more carefully",
        "helps his younger cousin with school tasks and often translates simple forms for neighbors who do not like using digital services",
        "learn how to turn rough volunteer energy into something with clearer roles and longer follow-through",
        "tried to keep the exchange running monthly, but turnout dropped once exam season started and he could not keep reminding everyone",
        "realized that people sometimes join a useful idea only when they can see immediate benefit, not when it is explained abstractly",
        "earns a bit of money in the family stall and occasionally by repairing school bags or zippers for neighbors",
        "refused to throw mixed trash behind the school fence during a cleanup just to finish faster",
        "would need project structure and communication practice more than raw motivation",
        "the signal is promising and grounded, but outcomes remain small and inconsistent",
    ),
    CandidateSpec(
        "syn_eng_v1_007",
        "hidden_potential_low_polish",
        "hard",
        "long",
        True,
        "Taldykorgan",
        "school classes",
        64,
        "Attestat",
        3.8,
        "quiet social responsibility and practical mutual aid",
        "children of seasonal workers in his area often spend afternoons alone and drift away from school routines",
        "coordinated a shared lunch and school-supply pool with three neighbors so younger students would not come to class empty-handed",
        "it never became a formal program, but several children used the supplies regularly and one teacher quietly started sending extra worksheets home through him",
        "his mother works cleaning shifts and he often manages two younger siblings after school while also helping a relative with deliveries",
        "helps a neighbor's son review basic math because the boy had nearly stopped attending homework sessions",
        "study somewhere that can show him how to build trust and structure without needing a title first",
        "wanted to join a youth volunteer council in the city but gave up halfway because travel costs and timing were too hard to manage",
        "seeing younger children act older than they should because adults were exhausted changed the way he thinks about responsibility",
        "sometimes gets small payments for loading goods, running errands, and doing evening delivery help for a relative",
        "spoke against classmates mocking a quieter boy even though nobody backed him up",
        "academic record is mixed, and his strongest signal may be easy to overlook because he describes it so modestly",
        "substance is real but understated, which makes the case disagreement-prone rather than easy",
    ),
    CandidateSpec(
        "syn_eng_v1_008",
        "hidden_potential_low_polish",
        "hard",
        "long",
        True,
        "Kyzylorda",
        "school classes",
        62,
        "Kazakhstan certificate",
        78,
        "local problem solving, especially around dust, heat, and basic study conditions",
        "families in her block complain about dust and noise, but students still try to study in shared spaces with poor lighting",
        "created a wall timetable and shared study corner in a common room and later pushed for a courtyard cleanup after repeated dust storms",
        "the study corner was used during exam weeks and the cleanup improved one section of the yard, though the effort depended on her direct involvement",
        "her father has recurring health problems, so she often buys medicine, handles chores, and keeps an eye on younger relatives",
        "helped a retired neighbor set up a fan and organize papers so the apartment felt less chaotic during the summer",
        "learn how to connect everyday practical needs with better planning, data gathering, and teamwork",
        "tried to keep a larger neighborhood group active after the cleanup but most people lost interest once the weather changed",
        "after repeated summers of hearing adults say 'this is just how it is,' she started to distrust passivity more than difficulty",
        "once earned a little money helping an aunt clean and sort school uniforms for resale before the new term",
        "pushed a group project to use its own photos instead of copied images even though the result looked less polished",
        "would need more language confidence and external encouragement in order not to drop back into a purely helping-at-home role",
        "the case has genuine signal but sits in a gray zone because the candidate is credible, modest, and still underdeveloped",
    ),
    CandidateSpec(
        "syn_eng_v1_009",
        "quiet_technical_builder",
        "clear",
        "long",
        True,
        "Pavlodar",
        "TOEFL",
        92,
        "Attestat",
        4.5,
        "electronics, repair culture, and practical technology for outages and everyday failures",
        "power cuts and broken appliances affect small-city families more than people admit, especially in older apartment blocks",
        "built a rough hand-crank phone charger and later started repairing kettles and small heaters for neighbors from spare parts",
        "the charger worked during outages and the heater repairs made him the person neighbors started asking before they called a service shop",
        "his grandmother was ill last winter, and he spent long hours at home fixing small things and helping his mother keep the household running",
        "replaced a fuse in an elderly neighbor's heater for free and stayed to explain how to use it safely",
        "learn in teams without losing the hands-on, fix-it-first mindset that has shaped how he solves problems",
        "spent a summer trying to build a micro-hydropower device for a creek near the city, but the output was too weak to be useful",
        "watching people online build useful things from scrap parts made him stop thinking invention belongs only to experts or large labs",
        "repairs irons, kettles, and chargers for small payments or sometimes just for thanks and tea",
        "stayed behind to clean the worst part of the school grounds instead of pretending the job was finished",
        "quiet in groups and slower to ask for help than he should be",
        "the practical builder signal is strong and grounded in repeated real-world usefulness",
    ),
    CandidateSpec(
        "syn_eng_v1_010",
        "quiet_technical_builder",
        "clear",
        "medium",
        True,
        "Aktobe",
        "IELTS",
        6.5,
        "Kazakhstan Certificate",
        4.4,
        "simple software tools, workflow fixes, and behind-the-scenes technical support",
        "students in her grade kept missing deadlines because assignments, schedule changes, and club notices were scattered in too many chats",
        "built a small Telegram reminder bot and shared template messages so classmates could track deadlines more clearly",
        "usage started with friends and then spread to most of the grade, and even one teacher began sending reminders through it",
        "moving to a larger school made her retreat into quiet technical work because it felt safer than speaking in front of groups",
        "regularly helps a retired engineer next door use email and basic software without making him feel embarrassed",
        "learn product thinking and design so that useful tools do not stay limited to one classroom or one school",
        "tried to turn the bot into a more complete school platform but lost momentum when classmates preferred quicker chat-based workarounds",
        "once she saw how one small technical fix could reduce stress for dozens of people, she began valuing usability more than complexity",
        "earns a little money designing simple digital posters and fixing formatting problems for classmates' small tutoring pages",
        "rewrote a presentation with her team instead of letting them submit copied content from the internet",
        "needs more confidence in presenting ideas and asking users what they actually need before building extra features",
        "real problem-solving signal is clear even though the candidate is understated rather than charismatic",
    ),
    CandidateSpec(
        "syn_eng_v1_011",
        "quiet_technical_builder",
        "borderline",
        "medium",
        True,
        "Karaganda",
        "school + online courses",
        78,
        "Kazakh National Certificate",
        84,
        "practical science, water quality, and low-cost home experiments",
        "families near the industrial part of the city often distrust tap water but have few affordable ways to test or improve it",
        "experimented with bottle filters using sand, charcoal, and cloth after repeated complaints about water taste in his neighborhood",
        "the filter improved taste and clarity for some households, but he knew it was not a full solution and lacked proper testing equipment",
        "when his family apartment developed a leak, he became more aware of how quickly daily infrastructure problems become personal",
        "helps a younger neighbor with math and science homework because he remembers how confusing those subjects felt before anyone explained them patiently",
        "study where he can combine chemistry, engineering, and communication rather than relying only on improvised home experiments",
        "spent months preparing for a regional physics olympiad and missed the prize places despite working through past problems every night",
        "reading about pollution and then seeing similar problems in his own city made him stop treating science as only school theory",
        "sometimes tutors younger students for a small fee and has earned money translating simple science explanations into easier English",
        "refused to share answer sheets before a quiz and instead helped a classmate understand the topic afterward",
        "good technical curiosity, but the evidence is still partly prototype-level and not yet organized into a stronger project",
        "the candidate has real practical merit but not all experiments are well validated or scalable yet",
    ),
    CandidateSpec(
        "syn_eng_v1_012",
        "quiet_technical_builder",
        "borderline",
        "long",
        True,
        "Kostanay",
        "IELTS",
        7.0,
        "Attestat",
        4.6,
        "sensors, making, and simple technology for resource efficiency",
        "people talk a lot about saving water in shared gardens and school spaces, but nobody measures where water is actually being wasted",
        "assembled a rough soil-moisture monitor from spare sensors and used it on his grandmother's plot and later near a small school garden",
        "the prototype helped show when watering was obviously unnecessary, though it was fragile and needed repeated recalibration",
        "his family does not have technical professionals around them, so most of his learning came from translated forums and trial and error",
        "fixed a neighbor's desk lamp and then showed two younger students how to replace a simple wire without being afraid of tools",
        "develop enough engineering depth and teamwork skill to turn small field experiments into tools other people can trust",
        "tried to enter a maker fair with the monitor but was rejected because the prototype documentation was weak and incomplete",
        "moving from pure curiosity to measuring real conditions changed how he thinks about what counts as a useful idea",
        "repairs cables, lamps, and small electronics for modest payments and sometimes swaps the work for spare parts",
        "insisted on checking measurements again in a school lab instead of submitting guessed numbers to save time",
        "shy in public and still learning how to explain technical ideas in plain language",
        "strong builder promise is present, but the communication gap keeps the case from being fully straightforward",
    ),
    CandidateSpec(
        "syn_eng_v1_013",
        "quiet_technical_builder",
        "borderline",
        "long",
        True,
        "Shymkent",
        "TOEFL",
        88,
        "Attestat",
        4.3,
        "repair work, digital organization, and affordable access to school materials",
        "students lose time and money when school devices break and textbooks are treated as disposable instead of reusable resources",
        "repaired printers and phones for neighbors and then built a simple used-book inventory sheet for students in his school",
        "the inventory reduced confusion for one term and made it easier for classmates to find second-hand books, though he struggled to keep the system updated alone",
        "he learned to work quietly because louder classmates often took over presentations while he handled the technical parts",
        "helped a younger cousin recover files from an old phone so she would not lose all her study notes before exams",
        "learn how to move from useful isolated fixes toward stronger systems that survive without one person carrying everything",
        "spent a semester trying to build a small robot car club, but the turnout stayed low and he chose a project that was too difficult for beginners",
        "once he noticed that keeping things working is also a form of service, he became less interested in flashy competitions and more interested in reliability",
        "earns small payments for printer fixes, battery replacements, and setting up phones for older relatives",
        "declined to let his team buy a ready-made essay online even when everyone else was stressed",
        "very capable behind the scenes, but still limited in outreach, group-building, and public explanation",
        "the profile is credible and useful, but the strongest actions remain local and individually carried",
    ),
    CandidateSpec(
        "syn_eng_v1_014",
        "quiet_technical_builder",
        "hard",
        "short",
        False,
        "Oral",
        "IELTS",
        6.0,
        "Kazakhstan high school diploma",
        80,
        "basic web tools, library access, and quiet independent work",
        "school libraries can feel closed and difficult to use, especially for new students who do not know what is available",
        "built a very simple library catalog page with QR labels for shelves after noticing how often students gave up looking for books",
        "a few classmates used it, but the site never spread much because most students kept using chat apps or asking friends directly",
        "after switching schools, she spent a long time preferring screens and solo work to noisy group settings",
        "helps her aunt with online forms and explains simple browser steps to neighbors who get stuck",
        "learn how to ask for feedback earlier instead of building useful things in isolation and hoping people adopt them",
        "wanted to launch a larger student-notes site but stopped when classmates showed little interest in using it",
        "realizing that a technically neat idea can still fail if people do not understand or trust it made her more reflective",
        "has earned a bit of money formatting CVs and translating simple tech instructions into clearer English for acquaintances",
        "did extra work rewriting copied slides for a group project instead of letting the shortcut pass",
        "needs communication confidence and product feedback loops more than raw technical curiosity",
        "useful technical behavior is visible, but it is hard to tell how well the candidate would adapt to collaborative work",
    ),
    CandidateSpec(
        "syn_eng_v1_015",
        "community_oriented_helper",
        "clear",
        "medium",
        True,
        "Taraz",
        "IELTS",
        6.5,
        "Attestat",
        4.0,
        "community learning, inclusion, and creating safe after-school routines for children",
        "many children in her neighborhood spend afternoons alone because parents work late and there is no nearby after-school club",
        "started a small reading and homework circle in the courtyard with donated books and borrowed school desks",
        "attendance grew slowly, parents began sending snacks, and by the end of summer older children were helping younger ones read aloud",
        "after her father lost his job for part of the year, she became more aware of how quickly family stress reaches children",
        "also helps elderly neighbors by shopping for them and sitting to listen when they are lonely",
        "create simple neighborhood learning spaces that do not depend only on teachers or formal institutions",
        "tried to get a local business to sponsor the circle with more books but received no reply",
        "watching shy children start asking questions in public changed her view of who is allowed to start change",
        "earned modest tutoring income in Russian reading and spent part of it buying books for the younger group",
        "kept less-confident readers in the club even when some parents wanted her to focus only on stronger children",
        "could grow with better project planning, but the core community signal is already well grounded",
        "the case is strong because there is direct people-facing evidence rather than only good intentions",
    ),
    CandidateSpec(
        "syn_eng_v1_016",
        "community_oriented_helper",
        "clear",
        "long",
        True,
        "Shymkent",
        "IELTS",
        6.8,
        "Attestat",
        4.1,
        "belonging, inclusion, and making group spaces easier for quieter people to enter",
        "many students in her school stay in separate circles and some quieter classmates disappear socially even when they are physically present",
        "organized culture-sharing lunches and small welcome rituals for isolated classmates after seeing how many people stayed silent in group activities",
        "the events remained informal but made several students more willing to join class discussions and later volunteer for school activities",
        "moving to a new class in middle school taught her what it feels like to be invisible and to depend on one person noticing",
        "has a habit of checking on younger students who get left out during projects and making sure they have a role",
        "learn how to scale inclusive community work beyond one class so it does not disappear when one organizer graduates",
        "wanted to become eco-club president but lost the final vote after months of work and still kept helping with events",
        "reading about human rights movements and then seeing small versions of exclusion in school made her value daily solidarity more than popularity",
        "earns a little money from babysitting and helping neighbors with summer errands, but sees creating a welcoming atmosphere as value too",
        "refused to let a quiet student be excluded from a group project even when others said it would be more efficient",
        "her main growth area is structure, not motivation or sincerity",
        "the social leadership signal is concrete, warm, and sustained enough to read as an easier case",
    ),
    CandidateSpec(
        "syn_eng_v1_017",
        "community_oriented_helper",
        "borderline",
        "medium",
        True,
        "Kyzylorda",
        "school + online practice",
        71,
        "Kazakhstan certificate",
        82,
        "peer support, tutoring, and modest resource-sharing in low-pressure settings",
        "students without tutors or stable internet often assume they are weaker when they mostly lack help and structure",
        "started weekend exam-prep circles in her courtyard and added a small stationery-sharing basket when she noticed children arriving without pens",
        "some students improved and parents appreciated the effort, but the circle depended heavily on her being present every week",
        "her mother works nursing shifts and the candidate often handles evening chores before the study sessions even begin",
        "has also helped a younger cousin practice English speaking because the cousin was too embarrassed to answer in class",
        "build better community study systems that are welcoming, not intimidating or expensive",
        "planned a larger summer revision camp with friends, but it never reached enough families to become real",
        "seeing children stop apologizing for asking 'simple' questions changed how she sees confidence and fairness",
        "earns a small amount through tutoring and sometimes by helping neighbors fill out school application forms",
        "argued against letting stronger students dominate an exam-prep group just to move faster through material",
        "would benefit from tools for delegation and time protection so everything does not depend on her energy",
        "good local impact is present, but sustainability is still uncertain",
    ),
    CandidateSpec(
        "syn_eng_v1_018",
        "community_oriented_helper",
        "borderline",
        "long",
        True,
        "Ust-Kamenogorsk",
        "IELTS",
        7.0,
        "NIS Certificate",
        4.2,
        "resource sharing, environmental care, and practical mutual aid",
        "families throw away a surprising amount of usable clothing and school materials while other households in the same area struggle quietly",
        "started a winter clothes swap and a parent volunteer rota after seeing children come to school underdressed and embarrassed",
        "the swap grew after a slow start and eventually included books and bags, though sorting and logistics repeatedly became chaotic",
        "moving cities in high school made her more sensitive to how shame and loneliness keep people from asking for help",
        "often shovels snow for an elderly neighbor and stays to hear stories about how the neighborhood used to feel more connected",
        "study somewhere that can help her connect environmental thinking, logistics, and community building rather than treating them separately",
        "spent months designing a wider recycling campaign at school, but only a few posters appeared because she struggled to get teachers on board",
        "watching older residents talk about a more communal city made her think public trust is built from repeated small actions, not slogans",
        "has earned money tutoring literature and Russian and has used part of it to buy supplies for the swap and for younger students",
        "called out classmates for littering during an excursion even when they told her to stop making a scene",
        "needs stronger systems thinking and volunteer coordination to move beyond one-off wins",
        "the application is credible and useful, but still somewhat split between several small initiatives",
    ),
    CandidateSpec(
        "syn_eng_v1_019",
        "community_oriented_helper",
        "borderline",
        "medium",
        True,
        "Pavlodar",
        "IELTS",
        6.2,
        "Attestat",
        3.9,
        "peer listening, anti-bullying support, and youth belonging",
        "many teenagers in her city seem disconnected and unwilling to talk honestly because they feel their voices do not matter",
        "started a small lunch-circle and peer support routine for students who were being left out or mocked",
        "the effort never became an official club, but a few students became more willing to attend events and ask for help in public",
        "she herself felt invisible during the transition to high school and learned the value of one person making space",
        "helped a classmate practice presentations after school because public speaking was making the classmate avoid entire assignments",
        "learn to turn empathy and informal support into better structured youth spaces that do not disappear after one semester",
        "worked for months on a youth podcast about local problems but could not keep the team together long enough to publish an episode",
        "reading about communication and participating in a youth forum made her start valuing listening as much as talking",
        "earns some money tutoring English and literature and builds custom exercises for younger students who need more confidence",
        "stood up for a shy classmate when her own friend group was mocking her",
        "good relational leadership is there, but outcomes remain soft and difficult to measure",
        "the case is promising but not fully easy because evidence is people-centered rather than project-system centered",
    ),
    CandidateSpec(
        "syn_eng_v1_020",
        "community_oriented_helper",
        "hard",
        "short",
        False,
        "Semey",
        "school classes",
        66,
        "Attestat",
        3.8,
        "elderly digital help, low-key volunteering, and quiet trust building",
        "older residents in her neighborhood are increasingly cut off from services because everything assumes comfort with phones and apps",
        "started simple phone-help afternoons for elderly neighbors and later added tea-and-story evenings because people kept staying to talk",
        "the gatherings were small but repeated, and the candidate noticed that loneliness was often a bigger problem than the phone itself",
        "she became this kind of helper after spending more time with grandparents while her parents were working long hours",
        "regularly helps one widow pay bills online and read confusing messages from service providers",
        "create intergenerational community spaces where practical help and human connection can happen at the same time",
        "wanted to partner with a library branch for a larger digital-help day, but the staff did not respond to her messages",
        "hearing older people describe how much easier it used to be to ask neighbors for help made her think social trust is also a resource",
        "does not earn much money herself, but sometimes gets small thank-you payments for digital help and grocery runs",
        "kept inviting slower learners into a group even when others wanted the sessions to move faster",
        "limited scale and soft evidence make the case harder even though the sincerity is clear",
        "the signal is good, but some reviewers may find the impact too diffuse or too informal",
    ),
    CandidateSpec(
        "syn_eng_v1_021",
        "hardship_responsibility_growth",
        "clear",
        "short",
        False,
        "Karaganda",
        "school classes",
        64,
        "Kazakhstan Certificate",
        79,
        "practical science and learning that could help ordinary families, not only students with resources",
        "small-city families living near heavy industry often feel pollution and stress every day but do not have tools to respond",
        "started doing tiny home science demonstrations for children in his building after his younger brother said school science only felt like memorizing",
        "the sessions were simple, but the younger children kept coming back and asking when the next one would happen",
        "his father lost his job in the mine, and he began managing meals, chores, and his brother's homework while both parents were overwhelmed",
        "helped a neighbor's child stop falling behind in math by turning homework into short games instead of punishments",
        "study practical science and low-cost problem solving so families like his do not feel that useful knowledge belongs somewhere else",
        "worked hard for a regional physics prize for two years and never got beyond participation level",
        "after his family's financial pressure grew, he stopped thinking success only means prizes and started valuing usefulness more",
        "tutors two children in his building for small fees and has used that money for supplies and groceries",
        "reported quiz cheating even though classmates stopped talking to him for several days",
        "good resilience is obvious, though the profile is still developing beyond family duty and small tutoring",
        "the growth narrative is well grounded and easier to trust than many more polished applications",
    ),
    CandidateSpec(
        "syn_eng_v1_022",
        "hardship_responsibility_growth",
        "borderline",
        "medium",
        True,
        "Atyrau",
        "school classes",
        70,
        "Attestat",
        4.1,
        "education access, family steadiness, and learning how to carry responsibility without becoming passive",
        "students from ordinary schools often feel invisible next to better-resourced applicants, even when they are doing a lot outside class",
        "kept a small study group alive while also covering shifts in the family market stall after his mother needed surgery",
        "the group stayed small but consistent and gave a few classmates enough confidence to ask teachers for extra help before exams",
        "for several months he balanced schoolwork with taking inventory, opening the stall, and looking after a younger cousin",
        "helped one classmate prepare for a retake exam because he knew that person was also working after school",
        "learn how to turn endurance into initiative and not only survive difficult periods efficiently",
        "hoped to raise enough marks for a stronger scholarship path but could not improve as much as he wanted while home responsibilities were high",
        "seeing how adults around him kept going without drama made him value reliability more than image",
        "earns small payments through market work and occasional tutoring in history and social studies",
        "refused to buy a ready-made essay with classmates even when everyone was exhausted and deadlines were close",
        "would benefit from academic structure and more confidence that his quieter form of leadership counts as leadership",
        "the hardship and growth are real, but the broader initiative record is still moderate",
    ),
    CandidateSpec(
        "syn_eng_v1_023",
        "hardship_responsibility_growth",
        "borderline",
        "long",
        True,
        "a village near Kokshetau",
        "school classes",
        59,
        "Village school certificate",
        4.1,
        "resilience, resource sharing, and trying to stabilize everyday life after disruption",
        "rural students lose momentum quickly when family housing, transport, or school routines are interrupted",
        "created a shared supply cupboard after a flood damaged several homes and students kept coming to school without basics",
        "teachers noticed the cupboard and quietly started leaving spare notebooks there, though the candidate never made it formal",
        "after flood damage forced her family to move temporarily, she changed schools and had to help manage younger siblings while adults dealt with repairs",
        "walked with one new classmate after school for several weeks because both of them were struggling to adjust after family disruption",
        "study somewhere that can show her how recovery, community support, and practical organization can work together",
        "prepared for an urban scholarship program but could not keep pace after the school change and unstable travel situation",
        "having to rebuild daily routines from almost nothing made her stop romanticizing stability and start valuing systems",
        "has earned a small amount helping relatives sort vegetables for sale and tutoring younger children in basic reading",
        "insisted on proper cleanup during a school volunteer day instead of letting people dump waste behind a fence",
        "would need support in confidence, language, and formal study habits after a disrupted school experience",
        "the trajectory is credible and strong, but some reviewers may want more evidence of proactive initiative beyond survival",
    ),
    CandidateSpec(
        "syn_eng_v1_024",
        "hardship_responsibility_growth",
        "borderline",
        "medium",
        False,
        "Ekibastuz",
        "school + online videos",
        72,
        "Kazakhstan certificate",
        81,
        "practical organization and caring work that is easy to overlook",
        "families caring for disabled children often have no simple systems to reduce daily stress",
        "built a shared family calendar and rotating chore board so appointments, medicine, and schoolwork for her younger sister would stop clashing",
        "the system made home life less chaotic and later she adapted the same idea for a small peer study plan with one friend",
        "her younger sister needs regular care, so the candidate often schedules around therapy, school transport, and household tasks",
        "helps a neighbor's child with math because she knows what it feels like to study in a noisy home",
        "learn service design and communication so she can solve practical problems for families who do not have time to explain themselves twice",
        "tried to qualify for a selective regional camp but missed the threshold while family responsibilities were highest",
        "once she realized that calm routines can be life-changing, she stopped dismissing 'small' solutions as unimportant",
        "does not earn much cash, but has been paid a little for babysitting and organizing school materials for relatives",
        "rewrote a group report instead of letting classmates submit copied material that would have been faster",
        "clear promise is there, but she would need support, exposure, and room to focus on herself too",
        "credible growth and care signal are present, though the application stays modest and not very expansive",
    ),
    CandidateSpec(
        "syn_eng_v1_025",
        "hardship_responsibility_growth",
        "hard",
        "long",
        True,
        "Zhanaozen",
        "school classes",
        63,
        "Attestat",
        3.9,
        "responsibility, youth direction, and trying to create steadier routines where boredom easily turns destructive",
        "teenagers in his area often drift into idleness because adults are busy and there are few low-cost structured activities",
        "started a weekend football-and-homework hour for younger boys after noticing they were hanging around shops until late",
        "the group was inconsistent but real, and a few boys began showing up early with schoolwork before the football part started",
        "one parent works rotational shifts away from home, so he often covers chores, errands, and emotional support at home without much planning time",
        "checks on a younger neighbor whose older brothers already stopped attending school regularly",
        "learn how to design youth programs that are attractive enough to compete with boredom and strong enough to survive without constant improvisation",
        "wanted to earn a city scholarship so his family would worry less about university costs, but his scores stayed below the line",
        "watching younger boys copy the habits of discouraged adults made him start thinking more about environment than about blame",
        "earns small amounts through market loading, delivery help, and assisting at a repair stall",
        "went to class instead of skipping with friends and later stood by that decision even after they mocked him",
        "academic readiness and follow-through are both uncertain, so the promise is mixed with real support needs",
        "the case is emotionally credible but harder because initiative, hardship, and readiness pull in different directions",
    ),
    CandidateSpec(
        "syn_eng_v1_026",
        "hardship_responsibility_growth",
        "hard",
        "long",
        True,
        "Ridder",
        "school + independent study",
        74,
        "Attestat",
        4.0,
        "mono-town resilience, repair mentality, and trying to create order under economic stress",
        "families in shrinking factory towns often feel that every plan is temporary because money and jobs can change suddenly",
        "started a very small household-budget workshop with cousins and neighbors after his family debt grew and arguments about money kept repeating",
        "the workshop remained rough, but it helped two relatives track spending for several months and made financial stress easier to discuss at home",
        "after layoffs affected extended family income, he worked at a relative's repair stall and studied at night when the house was finally quiet",
        "has also shown older neighbors how to compare bills and keep records instead of relying only on memory",
        "learn how technology, data, and local organizing can make struggling towns less reactive and more prepared",
        "aimed to enter a stronger urban college pipeline but could not sustain the grades while working evenings",
        "once he saw how much conflict came from confusion rather than bad intent, he started valuing simple systems more than arguments",
        "earns modest money at the repair stall and through sorting paperwork or deliveries for relatives",
        "refused to let a school team use copied financial data even though everyone wanted to finish quickly",
        "would need support to close gaps in academic focus and to rebuild confidence after prolonged family stress",
        "the candidate feels real and thoughtful, but some reviewers may see more survival than forward momentum",
    ),
    CandidateSpec(
        "syn_eng_v1_027",
        "academically_strong_but_narrow",
        "clear",
        "medium",
        True,
        "Astana",
        "IELTS",
        7.5,
        "specialized mathematics school certificate",
        96,
        "mathematics, analytical thinking, and disciplined academic work",
        "education inequality still wastes talent when strong math instruction is concentrated in a few schools",
        "co-organized a small mathematics circle for younger students and built an error-tracking spreadsheet for olympiad preparation",
        "the circle stayed small but steady, and the spreadsheet was later shared with classmates who found it genuinely useful",
        "pressure from family expectations during exam and olympiad season forced her to manage stress more deliberately",
        "spent weekends tutoring a cousin in algebra and breaking hard ideas into simpler steps",
        "learn how strong analytical training can connect to real interdisciplinary work instead of staying inside contest preparation",
        "worked for years toward a national olympiad medal and never made it beyond the city stage",
        "reading more about bias and decision-making made her question the idea that intelligence is only about fast, correct answers",
        "earns small tutoring income from algebra and exam preparation sessions",
        "refused to share solutions during practice sessions and instead proposed a study group where everyone solved problems honestly",
        "collaboration and project breadth are still weaker than academic discipline",
        "this is a clean academically strong case, but the limitation is clear: depth is narrower than range",
    ),
    CandidateSpec(
        "syn_eng_v1_028",
        "academically_strong_but_narrow",
        "borderline",
        "short",
        False,
        "Almaty",
        "TOEFL",
        101,
        "NIS diploma",
        4.7,
        "physics, mathematics, and careful independent study",
        "many talented students leave Kazakhstan because research pathways feel thin and uncertain at home",
        "made a detailed mistake log and small set of self-made practice drills to improve weak points before competitions",
        "the system helped her work more systematically, but it remained mostly a personal tool rather than a wider project",
        "joining a more competitive school made her realize how much she relied on solitary work and how little she enjoyed presentations",
        "occasionally helps a classmate who falls behind in calculus or mechanics before exams",
        "discover how theory can connect to practical problems and how to work with people who think differently",
        "trained for a national debate selection as a stretch goal and was not selected, which reminded her that communication is a separate skill",
        "meeting classmates from different regions made her more aware that rigor can look different outside elite classrooms",
        "has made value mainly through tutoring and by preparing careful study materials for juniors",
        "refused to let her team skip lab steps even though staying accurate meant leaving school late",
        "strong student, but still more scholar than builder",
        "the academic quality is clear, while real-world initiative remains limited",
    ),
    CandidateSpec(
        "syn_eng_v1_029",
        "academically_strong_but_narrow",
        "borderline",
        "medium",
        True,
        "Aktau",
        "IELTS",
        7.0,
        "specialized biology-chemistry school diploma",
        4.8,
        "biology, disciplined study, and careful explanation of difficult topics",
        "many schools outside larger cities do not have enough lab access to keep strong students curious",
        "designed neat revision sheets and small lab-summary templates for classmates preparing for biology and chemistry tests",
        "people appreciated the sheets, but the work remained mostly academic support rather than a broader public project",
        "preparing for exams while trying to keep perfect results made her realize how much of her identity was attached to grades",
        "voluntarily tutored a cousin who was afraid of chemistry and improved her confidence over time",
        "move from being good at mastering knowledge to being useful with knowledge in a more applied environment",
        "prepared for a national biology olympiad stage and missed it after strong early rounds",
        "reading about how scientists work through repeated failed experiments made her slightly less afraid of imperfect outcomes",
        "has earned money tutoring science and writing simple review sheets for neighborhood students",
        "declined to let classmates copy answers during a practice session and offered explanations instead",
        "would need encouragement to take initiative before she feels fully prepared",
        "real strength is present, but the profile still reads as academically excellent and operationally narrow",
    ),
    CandidateSpec(
        "syn_eng_v1_030",
        "academically_strong_but_narrow",
        "borderline",
        "medium",
        True,
        "Kokshetau",
        "IELTS",
        7.0,
        "Kazakhstan certificate",
        93,
        "history, writing, and carefully structured argument rather than spontaneous teamwork",
        "students from smaller cities rarely get serious humanities opportunities outside exams and essays",
        "ran a tiny essay-feedback circle for juniors and shared annotated examples of how to strengthen arguments",
        "the circle helped a few students write with more confidence, but it stayed heavily dependent on her preparing all the materials",
        "she spent years tying her self-worth to exam rankings and teacher praise, which made her cautious about unfamiliar work",
        "helped one classmate prepare for a literature competition by reading drafts and giving line-by-line comments",
        "test whether strong academic thinking in the humanities can become useful in public projects and not only in competitions",
        "spent months preparing for a national essay contest and stopped at the regional level",
        "realizing that clear thinking matters outside classrooms too came partly from mentoring younger students who only needed someone patient",
        "earns small fees for editing essays and university statements for neighbors and cousins",
        "argued against using copied quotations in a presentation even though her group wanted to finish faster",
        "would need practice acting under uncertainty rather than only when the rubric is known",
        "the candidate is strong, but the comfort zone is still visibly academic and tightly structured",
    ),
    CandidateSpec(
        "syn_eng_v1_031",
        "academically_strong_but_narrow",
        "hard",
        "long",
        True,
        "Astana",
        "IELTS",
        7.5,
        "specialized mathematics high school diploma",
        97,
        "statistics, programming, and long independent practice on abstract problems",
        "fast technological change can widen gaps if only already-prepared students know how to use new tools well",
        "built a personal system for tracking problem-solving errors, coding practice, and reading across statistics topics",
        "the system clearly improved her own discipline and helped a few classmates, but she still struggles to move from self-optimization to outward initiative",
        "high family expectations made even small academic setbacks feel larger than they should have",
        "sometimes explains probability and coding basics to younger students who want to join competitions but do not know how to begin",
        "find a direction where rigorous quantitative work actually helps solve messy real problems with other people involved",
        "planned for years to qualify for a national-level medal and never broke through beyond strong school-level performance",
        "after seeing how often initial assumptions fail in both math and life, she became more cautious and less attached to appearing right immediately",
        "has earned money tutoring math and basic coding online",
        "stood against a group norm of sharing answers in a study circle even though it made her temporarily unpopular",
        "project breadth and social courage are still underdeveloped relative to raw academic ability",
        "this case is harder because the ability is obvious while the practical translation is still mostly theoretical",
    ),
    CandidateSpec(
        "syn_eng_v1_032",
        "academically_strong_but_narrow",
        "hard",
        "short",
        False,
        "Karaganda",
        "TOEFL",
        96,
        "Attestat",
        4.7,
        "computer science competitions and self-directed technical study",
        "students in ordinary schools often never see what modern computing work can look like beyond exam tasks",
        "coded a small checker for practice tasks and kept a formula-and-bug notebook to avoid repeating the same mistakes",
        "the tool made his own preparation better and helped one friend, but he has little experience using it in a wider setting",
        "being bullied in middle school pushed him toward solo work and made teamwork feel risky even when he values it intellectually",
        "helped a neighbor's child with math because one-on-one support felt easier than public mentoring",
        "learn how to keep technical depth while becoming less isolated and more useful to real teams",
        "spent nearly a year preparing for a regional physics olympiad and still missed the final round",
        "tutoring others made him realize knowledge feels different when it has to make sense to someone else",
        "earns a little money helping schoolchildren with homework and making practice sheets",
        "redid a chemistry experiment honestly instead of letting classmates copy results from the internet",
        "very solid academically, but little evidence yet that he can translate ability into broader action",
        "the case is disagreement-prone because it is strong on substance but thin on outward initiative",
    ),
    CandidateSpec(
        "syn_eng_v1_033",
        "polished_but_thin",
        "clear",
        "short",
        False,
        "Astana",
        "IELTS",
        7.5,
        "NIS diploma",
        4.4,
        "leadership, communication, and social entrepreneurship in broad terms",
        "young people in Kazakhstan need more confidence to speak up and imagine bigger possibilities for themselves",
        "helped moderate school debates and a charity bake sale and often writes reflective posts about youth leadership",
        "the activities were positive but remained small and did not develop into a longer-running project",
        "during the pandemic she struggled with motivation and learned that she works best when surrounded by active people",
        "volunteered once at a children's home and found the experience meaningful",
        "create youth leadership workshops that help students communicate and believe in their own ideas",
        "prepared intensely for a national debate team but was not selected despite months of practice",
        "meeting exchange students made her more interested in global perspectives and the idea of social entrepreneurship",
        "ran a tiny handmade notebook sale online and learned basic promotion from it",
        "argued against copying content in a school presentation because originality mattered more than appearance",
        "good polish, but the concrete evidence of execution is still light",
        "the case is relatively easy because the gap between rhetoric and track record is visible without being deceptive",
    ),
    CandidateSpec(
        "syn_eng_v1_034",
        "polished_but_thin",
        "borderline",
        "medium",
        False,
        "Almaty",
        "TOEFL",
        104,
        "NIS graduation",
        4.5,
        "global-minded communication, youth empowerment, and creative campaigns",
        "students outside the biggest cities often miss opportunities that could help them imagine wider futures",
        "managed school social media and joined a virtual exchange project about climate and youth opportunity",
        "the communication work was competent and energetic, but the application offers little evidence of a sustained initiative she personally carried through",
        "online schooling made her more disciplined and more conscious of how much she depends on social energy",
        "spent one winter volunteering at a children's home during holiday events",
        "use branding and communication skills for educational equity campaigns that connect urban and rural students",
        "hoped to launch a student podcast but never got past topic lists and guest messages",
        "hearing students from other countries describe their local youth initiatives made her want similar energy at home",
        "earned small payments making Canva posts and English captions for classmates' school projects",
        "tried to persuade her group not to use borrowed slides from the internet, even though they said no one would care",
        "the profile is articulate and plausible, but it leans much more on direction than on evidence",
        "this one can split opinions because the candidate sounds capable and sincere while still remaining quite thin on specifics",
    ),
    CandidateSpec(
        "syn_eng_v1_035",
        "polished_but_thin",
        "borderline",
        "medium",
        True,
        "Shymkent",
        "IELTS",
        7.0,
        "Attestat",
        4.2,
        "entrepreneurship, youth confidence, and energizing peers around new ideas",
        "young people in his city often have energy but no frameworks for trying projects or learning from failure",
        "organized one local entrepreneur Q and A, helped with a business game day, and often motivates classmates to think more boldly",
        "the events were well received at the time, but most of the projects faded quickly and there is little evidence of deeper follow-up",
        "being class president for a year taught him that enthusiasm alone does not keep a group active",
        "mentored his cousin through the first steps of an online baking page and helped with captions and product photos",
        "build programs that teach young people how to move from ideas to actual execution",
        "spent weeks trying to build a used-book marketplace page that lost activity almost as soon as it launched",
        "following Kazakh entrepreneurs online changed his view of failure from embarrassment to part of experimentation",
        "did small digital marketing tasks and basic DM management for a friend's store",
        "refused to let a group buy an essay online even when deadlines were closing in",
        "good energy, but the record still shows more starting than sustaining",
        "the candidate may be stronger than thin polished cases usually are, but the evidence is still mostly early-stage",
    ),
    CandidateSpec(
        "syn_eng_v1_036",
        "polished_but_thin",
        "borderline",
        "long",
        True,
        "Atyrau",
        "IELTS",
        7.5,
        "specialized school diploma",
        4.6,
        "climate communication, youth discussion spaces, and digital outreach",
        "environmental issues in western Kazakhstan need better public conversation, but school efforts often stay performative",
        "wrote blog posts about water and waste, led a few discussion circles, and helped share environmental content across student channels",
        "the communication was polished and thoughtful, but most examples stop at discussion or posting rather than clear execution or measurable change",
        "during exam periods she noticed how easily her own motivation shifted from substance to appearance and tried to correct for that",
        "once volunteered in a beach cleanup and later kept checking on one younger student who felt intimidated by public discussions",
        "create campaigns that make environmental issues feel immediate and personal to ordinary students",
        "planned a larger school campaign on plastic use but never moved far beyond draft materials and a few conversations",
        "watching a documentary about pollution in the Caspian region made her think awareness without persistence is not enough",
        "has earned some money editing English grammar, translating short texts, and preparing polished captions for student groups",
        "pushed her group to make original content rather than using copied images in a project deck",
        "the profile is appealing and articulate, but many claims still sit at the level of intention or framing",
        "the case is borderline because it sounds strong enough to tempt a reviewer into over-crediting polish",
    ),
    CandidateSpec(
        "syn_eng_v1_037",
        "polished_but_thin",
        "borderline",
        "medium",
        True,
        "Turkistan",
        "IELTS",
        6.8,
        "Kazakhstan certificate",
        4.3,
        "law, public dialogue, and fairness in educational access",
        "students from rural or ordinary schools are often excluded from serious conversations about policy and opportunity",
        "helped run debate events, drafted a small petition for better student communication, and writes reflective essays about fairness",
        "the writing is clear and thoughtful, but the public-facing work remains small and the petition never moved beyond a few signatures",
        "moving from passively agreeing with teachers to asking harder questions made her feel both more capable and more uncertain",
        "sometimes helps younger students with essay structure and public speaking nerves",
        "learn how to move from discussion and argument into designing something that actually changes how institutions work",
        "wanted to represent her region at a national debate event but was not chosen after the final school round",
        "reading about legal reform and civic participation made her start seeing education problems as systemic rather than personal failures",
        "earns a little money editing essays and doing translation help for school-related applications",
        "spoke against classmates mocking a quiet student during a group activity",
        "polish is higher than execution, though there is still some sincerity and effort here",
        "the case may look stronger on first read than it really is once evidence is separated from aspirations",
    ),
    CandidateSpec(
        "syn_eng_v1_038",
        "polished_but_thin",
        "hard",
        "short",
        False,
        "Oral",
        "TOEFL",
        98,
        "Attestat",
        4.4,
        "creative industries, storytelling, and youth connection",
        "many teenagers feel disconnected and underestimate the value of their own stories and perspectives",
        "ran a positive online storytelling page and occasionally shared supportive prompts for peers",
        "the page looked thoughtful, but the application gives very little grounded evidence of what changed because of it",
        "after feeling isolated during online schooling, she became more interested in belonging and expression",
        "helped one younger student feel less nervous about sharing writing in class",
        "create storytelling spaces where students feel seen and learn to express themselves confidently",
        "wanted to launch a larger cross-school creative workshop series but never found a reliable team",
        "following international youth forums made her think broad creativity matters, but also made her style more abstract than practical",
        "has earned modest money designing highlight covers and simple social media graphics",
        "refused to repost a friend's event because the tone felt exclusionary even though it was socially awkward",
        "good communication instincts, but visible evidence is especially thin here",
        "this case is hard because the candidate may be sincere, yet the record stays unusually outcome-light",
    ),
    CandidateSpec(
        "syn_eng_v1_039",
        "support_needed_promising",
        "clear",
        "medium",
        True,
        "a village near Pavlodar",
        "school classes",
        57,
        "Village school certificate",
        4.2,
        "technology access and practical learning for students who are usually last in line",
        "rural students are expected to use digital tools without having reliable access to computers, guidance, or confidence",
        "helped a teacher reopen an old computer room and sorted the working machines so younger students could actually use them",
        "the room was still basic, but several students began staying after class to practice typing and search for school information",
        "he grew up doing farm work and chores before school, so he is used to responsibility but not used to presenting himself strongly",
        "helps neighbors fill online forms because they do not trust themselves with official websites",
        "learn enough technology and teamwork to bring simple digital opportunities back to schools like his",
        "wanted to win a small district scholarship but fell short because his exam preparation was inconsistent",
        "watching classmates get excited just by turning on old computers made him realize how low the access bar still is in many places",
        "has earned a little through field help and by fixing basic settings on old phones or laptops",
        "did his own work instead of giving a friend copied homework because he wanted to stay fair",
        "needs language development, confidence support, and exposure to people who expect him to contribute intellectually",
        "promise is clear, but the candidate would need substantial support to convert that promise into performance",
    ),
    CandidateSpec(
        "syn_eng_v1_040",
        "support_needed_promising",
        "clear",
        "medium",
        True,
        "Taraz",
        "school + online practice",
        66,
        "Attestat",
        4.0,
        "safe study spaces for girls and confidence-building through small structured groups",
        "many girls in her area do well quietly but avoid leadership because there are few spaces where they can practice speaking without embarrassment",
        "started an after-school study table with two friends for younger girls who were too shy to ask questions in larger groups",
        "the table stayed small but consistent, and several younger students became more comfortable asking for help instead of copying work",
        "after her father was ill for several months, she began taking more responsibility at home and became less carefree about time",
        "helps a neighbor's daughter with English speaking and school applications because the family does not know how to start",
        "gain the structure and language confidence to lead programs that feel welcoming to students who underestimate themselves",
        "set a goal of scoring above 7.0 on IELTS and stopped at 6.5 after repeated effort",
        "once she saw how fast younger girls opened up in a low-pressure setting, she stopped believing confidence is something people either have or do not have",
        "has earned a little money from tutoring and from making simple study packs for younger students",
        "kept weaker students in the study group even when friends said it would be easier to work only with fast learners",
        "would benefit from language, presentation, and project-management support, not from pressure alone",
        "the profile is promising in a fairly direct way because the support need and the useful action are both visible",
    ),
    CandidateSpec(
        "syn_eng_v1_041",
        "support_needed_promising",
        "borderline",
        "medium",
        True,
        "Kostanay",
        "school classes",
        71,
        "Kazakhstan certificate",
        83,
        "adaptation, welcoming systems, and quiet organizational care",
        "students who transfer schools often lose confidence before anyone notices that the main problem is not ability but disorientation",
        "made a short welcome guide for new students with classroom maps, teacher habits, and practical tips after struggling with transfer himself",
        "a few new students said it helped, but the guide was informal and never became a school-wide practice",
        "switching schools left him cautious about speaking, and he still finds big group settings draining",
        "has helped newer students understand routines and avoid small mistakes that make school feel hostile",
        "learn how to combine quiet organization with stronger communication so that useful support does not stay invisible",
        "wanted to join a regional leadership camp but chose not to go after worrying his English and social confidence were not enough",
        "being the lost student first changed how he thinks about orientation, belonging, and the cost of confusion",
        "has done some light translation and typing work for neighbors and relatives",
        "objected when a team wanted to leave a quieter student out of a faster-working group",
        "support need is central because the candidate is capable but still hesitant and self-limiting",
        "there is real promise, though some reviewers may see the initiative as too modest",
    ),
    CandidateSpec(
        "syn_eng_v1_042",
        "support_needed_promising",
        "borderline",
        "medium",
        True,
        "Zhezkazgan",
        "school classes",
        60,
        "Kazakhstan high school diploma",
        78,
        "practical problem solving, family reliability, and trying to become more than the circumstances around him",
        "water waste, broken public space, and low expectations combine to make people in his area assume that nothing small is worth fixing",
        "started a simple water-saving and repair-awareness poster effort at school after repeated complaints about leaking taps and broken sinks",
        "the posters sparked conversation and one teacher supported him, but the effort did not become an official student initiative",
        "family money has been unstable for years, and he often helps in a small shop before coming home to more chores",
        "helps younger kids in the building with basic homework when their parents are still working",
        "study in a place that can turn his scattered practical ideas into better built projects with real support",
        "tried to improve his English enough for a city exchange opportunity but missed the cut and then lost confidence for a while",
        "watching adults accept repeated small failures in public space made him start thinking that maintenance is also a form of dignity",
        "earns small amounts through shop help, deliveries, and occasional tutoring in math",
        "told classmates he would not join them in skipping a cleanup just to get home early",
        "needs clearer academic structure, better English, and stronger feedback than he has locally",
        "the promise is there, but the evidence still sits between useful instinct and early execution",
    ),
    CandidateSpec(
        "syn_eng_v1_043",
        "support_needed_promising",
        "hard",
        "long",
        True,
        "a village near Turkistan",
        "school classes",
        55,
        "Village school certificate",
        4.0,
        "design-minded practical help and small visual improvements that make spaces easier to use",
        "students in low-resource schools often live inside environments that quietly tell them not to expect care, clarity, or modern tools",
        "designed clearer classroom labels, simple direction signs, and basic notice-board posters because new students were always lost",
        "teachers noticed the change and one hallway became easier to navigate, but the candidate still worries that the work was too small to matter",
        "as a first-generation university hopeful, she carries strong family hopes while still feeling unsure whether she belongs in competitive spaces",
        "helps her mother with forms, bills, and translating simple online instructions when the family is unsure what to do",
        "learn how design, communication, and service can solve everyday problems without needing expensive technology",
        "wanted to lead a larger school redesign day but stopped after assuming teachers would say no",
        "seeing how much calmer younger students became once signs were clearer made her rethink what counts as meaningful improvement",
        "has earned a little designing flyers and name cards for neighbors' small services",
        "rewrote copied poster text with her own words even though it meant starting the design work again",
        "very promising, but very underexposed; she would need strong support to step into larger opportunities",
        "this case is hard because the upside is visible, but the evidence is subtle and the confidence level is low",
    ),
    CandidateSpec(
        "syn_eng_v1_044",
        "support_needed_promising",
        "hard",
        "short",
        False,
        "Semey outskirts",
        "school classes",
        58,
        "Attestat",
        3.9,
        "caregiving, simple teaching, and trying not to stay mentally small",
        "students in ordinary neighborhoods often give up on language learning because nobody around them can show what improvement looks like",
        "started helping a disabled cousin and two younger children with basic English words and simple speaking practice at home",
        "the work was tiny and home-based, but one child stopped being afraid to answer in English class",
        "her home life revolves around care tasks, and she rarely has uninterrupted time for her own study",
        "watches over children for neighbors and helps an older aunt with errands",
        "find a learning environment that gives her structure, encouragement, and a clearer idea of what she could become",
        "tried to prepare for an English camp application but never finished because home duties kept interrupting the process",
        "seeing one younger child become less ashamed of speaking badly made her stop treating low confidence as a fixed trait",
        "sometimes gets paid a little for babysitting or for helping younger children with homework",
        "refused to give copied answers to a friend even when the friend said she was being selfish",
        "would need significant support in time structure, language, and self-belief",
        "the promise is real but thinly evidenced and easy to undervalue",
    ),
    CandidateSpec(
        "syn_eng_v1_045",
        "borderline_manual_review",
        "clear",
        "medium",
        True,
        "Astana",
        "IELTS",
        7.5,
        "specialized high school diploma",
        4.6,
        "social entrepreneurship, youth consulting, and educational access",
        "education inequality and weak youth participation are major barriers to Kazakhstan's long-term development",
        "describes having built a regional youth consulting initiative that advised schools and community groups, but the concrete example that follows is a one-day classroom fundraiser and a few meetings",
        "the application sounds impressive and highly structured, yet the visible outcomes remain unusually small compared with the scale of the claim",
        "online learning during the pandemic made her more disciplined and more aware of how much she likes active communities",
        "mentions helping younger students with public speaking and volunteering once at a children's center",
        "launch broader youth leadership programs that connect urban and rural schools",
        "spent months preparing for a prestigious youth competition and was not selected, which she says taught her to grow through disappointment",
        "meeting exchange students made her believe more in global collaboration and ambitious youth-led change",
        "has sold handmade stationery online and made polished digital posts for school events",
        "insisted on original presentation work instead of copied content even when teammates complained",
        "the case may be promising, but the scale of the rhetoric sits well above the scale of the evidence",
        "strong language and vague large claims create a clear manual-review need even without an obvious broken fact",
    ),
    CandidateSpec(
        "syn_eng_v1_046",
        "borderline_manual_review",
        "borderline",
        "medium",
        True,
        "Atyrau",
        "IELTS",
        7.0,
        "Attestat",
        4.4,
        "youth outreach, environmental awareness, and community education",
        "coastal pollution and low civic participation make environmental issues feel both urgent and strangely ignored",
        "says he built an outreach program linking his former village school and his current city network, but later describes having studied in city schools the whole time and gives no clean explanation of when the move happened",
        "the visible project elements are plausible in isolation, yet the timeline across sections does not line up neatly",
        "he writes that the pandemic pushed him toward digital organizing and made him more interested in public communication",
        "also says he helped a younger cousin with schoolwork and joined several local cleanups",
        "build education and environmental campaigns that connect smaller communities to larger-city resources",
        "worked on an air-monitoring idea for months but says it stayed at planning stage because team members drifted away",
        "reading about regional environmental damage made him start paying more attention to local public information and community response",
        "has earned a little through English editing and through managing captions for a local student page",
        "pushed his team to avoid copied images and use original work instead",
        "the application may be genuine, but the location and timeline details need a human check",
        "the candidate could be real and promising, yet the internal chronology is loose enough to justify manual review",
    ),
    CandidateSpec(
        "syn_eng_v1_047",
        "borderline_manual_review",
        "borderline",
        "long",
        True,
        "Almaty",
        "TOEFL",
        105,
        "specialized diploma",
        4.7,
        "technology, social impact, and product-building for student coordination",
        "students need digital tools that feel local and practical rather than imported ideas that do not fit school realities",
        "claims to have built a student app with hundreds of users and school-wide traction, but later explains that only a few friends tested the prototype and that most classmates stayed with chat apps",
        "some parts of the story sound like a real early prototype, while other phrasing overstates reach and maturity",
        "joining a highly competitive school made him more ambitious but also more aware of how easy it is to compare image instead of evidence",
        "mentions helping younger students with coding and fixing classmates' devices when they break",
        "develop real products that solve ordinary problems and are honest about what stage they are actually in",
        "worked through several competition cycles without winning the recognition he expected from his prototype work",
        "following startup content online made him more interested in building quickly, but he also admits it may have shaped how he talks about unfinished work",
        "has earned money doing light digital product mockups and repairing small electronics",
        "opposed buying a ready-made project report online despite group pressure",
        "strong technical interest is plausible, but the application inflates outcome language in a way that needs checking",
        "this is an ambiguous manual-review case because the gap may be exaggeration, not fabrication",
    ),
    CandidateSpec(
        "syn_eng_v1_048",
        "borderline_manual_review",
        "hard",
        "short",
        False,
        "Shymkent",
        "IELTS",
        7.0,
        "Attestat",
        4.3,
        "business, storytelling, youth labs, and social impact language",
        "youth in regional cities need stronger entrepreneurial imagination and more places to test ideas safely",
        "describes leading an interschool youth lab and advising small businesses, yet the visible examples narrow down to one school speaker event, a few polished posts, and broad claims about helping others think bigger",
        "the writing is confident and polished, but specific evidence stays unusually evasive when it should be most concrete",
        "he says a difficult period in high school taught him to rely on discipline and public confidence, but details stay abstract",
        "mentions mentoring a cousin's small baking page and helping classmates prepare posts for events",
        "build entrepreneurship programs for students in smaller cities and create a culture that is less afraid of experimentation",
        "spent nearly a year planning a youth podcast and never published an episode because the team drifted away",
        "watching polished entrepreneur content online made him believe strongly in startup culture, though he admits he is still learning what real execution requires",
        "has done small digital marketing and content-support jobs online",
        "refused to repost a campaign that felt ethically questionable even though friends were involved",
        "the candidate may have real energy, but the large-scale framing and thin grounding make the case especially hard to interpret",
        "this case is hard because the story could reflect either genuine potential with inflated language or a meaningfully unreliable self-presentation",
    ),
]


ARCHETYPE_MIX = {
    "hidden_potential_low_polish": 8,
    "quiet_technical_builder": 6,
    "community_oriented_helper": 6,
    "hardship_responsibility_growth": 6,
    "academically_strong_but_narrow": 6,
    "polished_but_thin": 6,
    "support_needed_promising": 6,
    "borderline_manual_review": 4,
}

AMBIGUITY_MIX = {"clear": 12, "borderline": 24, "hard": 12}
TEXT_LENGTH_MIX = {"short": 10, "medium": 22, "long": 16}


def normalize_space(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def paragraph(*sentences: str) -> str:
    return normalize_space(" ".join(s for s in sentences if s))


def spec_index(spec: CandidateSpec) -> int:
    return int(spec.candidate_id.rsplit("_", 1)[-1])


def choose_variant(spec: CandidateSpec, *options: str) -> str:
    if not options:
        return ""
    return options[spec_index(spec) % len(options)]


def first_personize(text: str) -> str:
    result = f" {normalize_space(text)} "
    phrase_replacements = [
        (r"\bhis\b", "my"),
        (r"\bher\b", "my"),
        (r"\bhe\b", "I"),
        (r"\bshe\b", "I"),
        (r"\bhim\b", "me"),
        (r"\bhimself\b", "myself"),
        (r"\bherself\b", "myself"),
    ]
    for pattern, replacement in phrase_replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    verb_replacements = {
        "helps": "help",
        "schedules": "schedule",
        "manages": "manage",
        "keeps": "keep",
        "buys": "buy",
        "handles": "handle",
        "stays": "stay",
        "visits": "visit",
        "earns": "earn",
        "gets": "get",
        "pushes": "push",
        "builds": "build",
        "belongs": "belong",
        "claims": "claim",
        "coordinates": "coordinate",
        "creates": "create",
        "describes": "describe",
        "starts": "start",
        "learns": "learn",
        "knows": "know",
        "wants": "want",
        "speaks": "speak",
        "undersells": "undersell",
        "stops": "stop",
        "refuses": "refuse",
        "insists": "insist",
        "thinks": "think",
        "feels": "feel",
        "likes": "like",
        "uses": "use",
        "struggles": "struggle",
        "covers": "cover",
        "checks": "check",
        "tracks": "track",
        "repairs": "repair",
        "replaces": "replace",
        "spends": "spend",
        "notices": "notice",
        "tries": "try",
        "organizes": "organize",
        "runs": "run",
        "reads": "read",
        "drives": "drive",
        "makes": "make",
        "begins": "begin",
        "grows": "grow",
        "carries": "carry",
        "turns": "turn",
        "comes": "come",
        "worries": "worry",
        "writes": "write",
        "values": "value",
        "translates": "translate",
        "depends": "depend",
        "says": "say",
        "solves": "solve",
        "works": "work",
        "goes": "go",
        "has": "have",
        "was": "was",
        "is": "am",
    }
    for source, target in verb_replacements.items():
        result = re.sub(rf"\bI {source}\b", f"I {target}", result)
        result = re.sub(rf"\bI (often|sometimes|usually|still|later|already|regularly|quietly|also) {source}\b", rf"I \1 {target}", result)
        result = re.sub(rf"\band {source}\b", f"and {target}", result)
        result = re.sub(rf"\band (often|sometimes|usually|still|later|already|regularly|quietly|also) {source}\b", rf"and \1 {target}", result)

    result = re.sub(r"\bI am applying because I want to\b", "I want to", result)
    result = re.sub(r"\bI am applying from\b", "I am from", result)
    result = re.sub(r"\bmy father lost my job\b", "my father lost his job", result, flags=re.IGNORECASE)
    result = re.sub(r"\bmy mother lost my job\b", "my mother lost her job", result, flags=re.IGNORECASE)
    result = re.sub(r"\bmy to\b", "me to", result, flags=re.IGNORECASE)
    result = re.sub(r"\blike my\b", "like mine", result, flags=re.IGNORECASE)
    result = re.sub(r"\btaught my to\b", "taught me to", result, flags=re.IGNORECASE)
    result = re.sub(r"\bforced my to\b", "forced me to", result, flags=re.IGNORECASE)
    return normalize_space(result)


def sanitize_visible_text(text: str) -> str:
    cleaned = first_personize(text)
    forbidden_replacements = [
        (r"\bsupport need is central because the candidate is capable but still hesitant and self-limiting\b", "I still have a lot to grow into, especially when a setting asks for confidence and initiative."),
        (r"\bwould benefit from language, presentation, and project-management support, not from pressure alone\b", "I know I still need stronger language practice, better planning habits, and more confidence speaking up."),
        (r"\bneeds clearer academic structure, better English, and stronger feedback than I have locally\b", "I know I would grow faster in a place with clearer structure, better English practice, and more honest feedback."),
        (r"\bvery promising, but very underexposed; I would need strong support to step into larger opportunities\b", "I have done more in small settings than in visible ones, and I want to learn how to step into larger opportunities with more confidence."),
        (r"\bwould need significant support in time structure, language, and self-belief\b", "I know I still need stronger time structure, better English, and more confidence in what I can contribute."),
        (r"\beasy to overlook\b", "easy to miss at first"),
        (r"\bacademic record is mixed, and I strongest signal may be easy to overlook because I describe it so modestly\b", "My academic record is mixed, and I know I do not always explain my work in the strongest way."),
        (r"\bI know my English and self-presentation are not the strongest parts of my application\b", "I know my English and self-presentation still need work."),
        (r"\bI do not think I have the strongest or loudest application\b", "I know I am still growing into my voice and confidence."),
        (r"\bI do not have a big profile\b", "I have grown mostly through small, practical experiences."),
    ]
    for pattern, replacement in forbidden_replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthe candidate\b", "I", cleaned, flags=re.IGNORECASE)
    return normalize_space(cleaned)


def visible_fragment(text: str) -> str:
    return sanitize_visible_text(text)


def support_growth_text(spec: CandidateSpec) -> str:
    return choose_variant(
        spec,
        "I am ready for a more demanding environment, but I know I grow best when expectations come with clear feedback and structure.",
        "What I need now is not rescue, but a place where guidance, challenge, and accountability exist at the same time.",
        "I have learned a lot in small settings, and the next step for me is an environment that asks more of me while still teaching me how to meet that standard.",
    )


def routine_text(spec: CandidateSpec) -> str:
    if spec.archetype == "quiet_technical_builder":
        return "I keep a notebook of unfinished technical ideas and try to return to it every evening, even if only for twenty minutes. I also watch one tutorial or forum explanation every two or three days because I learn best by seeing how someone actually solves a problem."
    if spec.archetype == "community_oriented_helper":
        return "I keep a small notebook where I write what worked and what did not in the groups I help run. I also try to speak to at least one younger student or neighbor each day, because community work only makes sense if people feel noticed."
    if spec.archetype == "hardship_responsibility_growth":
        return "My day starts early because family chores happen before school, not after everything is already calm. To stay steady, I try to finish schoolwork the same day I get it and I write a short list at night so home pressure does not make everything feel chaotic."
    if spec.archetype == "academically_strong_but_narrow":
        return "I keep a strict study schedule and usually review mistakes before I sleep, because otherwise I repeat the same weak spots. This routine helps me stay calm during difficult academic periods, although I know it also keeps me in a very controlled comfort zone."
    if spec.archetype == "polished_but_thin":
        return "I start most mornings by reviewing my planner and choosing three priorities, because structure helps me turn ideas into action. I also try to reflect at the end of the day on whether I was actually useful or only busy."
    if spec.archetype == "support_needed_promising":
        return "I try to study at the same time each evening, even if the amount is not large, because routine helps when confidence is unstable. I also keep a list of words or concepts I do not understand so I can come back to them instead of pretending I already know."
    return "I wake up early and try to finish schoolwork before home tasks become heavy, because otherwise the day disappears. I also keep small notes about what helped me and what caused problems, so I do not repeat the same mistakes."


def decision_text(spec: CandidateSpec) -> str:
    if spec.archetype in {"hidden_potential_low_polish", "hardship_responsibility_growth", "support_needed_promising"}:
        return "For important decisions, I usually talk first with my mother or one trusted adult and then check what information I can find online. That is how I make sense of big choices, because I do not like pretending I know everything alone."
    if spec.archetype in {"quiet_technical_builder", "academically_strong_but_narrow"}:
        return "I usually start with facts, lists, or comparisons and then ask one person I trust if I feel uncertain. I like decisions that can be explained, even though I know not everything important fits into a table."
    return "I usually mix advice, reflection, and a quick check of whatever information is available. If a choice affects other people, I also ask myself who gains and who gets left out by the easier option."


def core_values_text(spec: CandidateSpec) -> str:
    if spec.archetype == "quiet_technical_builder":
        return "My three core values are curiosity, reliability, and honesty. Curiosity shows up when I keep testing and fixing things until I understand them better. Reliability shows up when people trust me with repairs or practical problems. Honesty matters because I would rather admit uncertainty than pretend something works when it does not."
    if spec.archetype == "community_oriented_helper":
        return "My three core values are empathy, fairness, and persistence. Empathy matters because I notice who is left out. Fairness matters because I do not like support going only to the already confident. Persistence matters because community efforts look small before they look useful."
    if spec.archetype == "hardship_responsibility_growth":
        return "My three core values are family responsibility, honesty, and endurance. Family responsibility became real through daily care work. Honesty matters because difficult periods only become manageable when people stop hiding the truth. Endurance matters because not every important effort gives quick results."
    if spec.archetype == "academically_strong_but_narrow":
        return "My three core values are discipline, integrity, and usefulness. Discipline helps me keep working when a problem is difficult. Integrity matters because I do not want academic success to be empty. Usefulness matters because knowledge should eventually help someone beyond my own score."
    if spec.archetype == "polished_but_thin":
        return "My three core values are initiative, integrity, and kindness. Initiative matters because I prefer trying to start something over waiting for perfect timing. Integrity matters because polished work means little if it is not honest. Kindness matters because confidence should help others, not only make a person visible."
    if spec.archetype == "support_needed_promising":
        return "My three core values are patience, honesty, and care. Patience matters because growth is slow when resources are limited. Honesty matters because I do not want to act more confident than I really am. Care matters because I think people learn faster when they feel safe."
    return "My three core values are honesty, hard work, and respect. Honesty matters even in small school situations. Hard work matters because resources are uneven and effort is often the only thing I can control. Respect matters because community depends on how we treat people in ordinary moments."


def unpopular_text(spec: CandidateSpec) -> str:
    if "quieter" in spec.ethics_case or "excluded" in spec.ethics_case:
        return "There was a moment when others wanted to move faster by leaving a quieter person behind, and I argued against that even though it made me unpopular in the moment. I did it because I know how quickly people start believing they do not belong when nobody makes space for them."
    if "copied" in spec.ethics_case or "essay" in spec.ethics_case or "slides" in spec.ethics_case:
        return "I once pushed back when classmates wanted to use copied work because it felt dishonest, even though I knew the easier choice would also be the more popular one. I stayed with my position because I would rather submit imperfect work than pretend shortcuts are normal."
    return "There was a time when my friends wanted to skip responsibilities and I chose not to join them. It was uncomfortable, but I did not want short-term approval to matter more than what I thought was right."


def question_indices(spec: CandidateSpec) -> list[int]:
    if spec.bucket == "short":
        base = {
            "hidden_potential_low_polish": [1, 4, 11],
            "quiet_technical_builder": [3, 4, 19],
            "community_oriented_helper": [1, 4, 16],
            "hardship_responsibility_growth": [1, 4, 12],
            "academically_strong_but_narrow": [2, 3, 20],
            "polished_but_thin": [3, 4, 21],
            "support_needed_promising": [1, 4, 19],
            "borderline_manual_review": [3, 4, 21],
        }
        return base[spec.archetype]
    if spec.bucket == "medium":
        base = {
            "hidden_potential_low_polish": [1, 2, 4, 11, 19],
            "quiet_technical_builder": [2, 3, 4, 14, 20],
            "community_oriented_helper": [1, 4, 11, 16, 17],
            "hardship_responsibility_growth": [1, 2, 3, 11, 18],
            "academically_strong_but_narrow": [1, 2, 3, 4, 20],
            "polished_but_thin": [2, 3, 4, 13, 21],
            "support_needed_promising": [1, 2, 4, 15, 19],
            "borderline_manual_review": [1, 3, 4, 7, 21],
        }
        return base[spec.archetype]
    base = {
        "hidden_potential_low_polish": [1, 2, 3, 4, 6, 11, 19, 20],
        "quiet_technical_builder": [1, 2, 3, 4, 5, 7, 14, 20],
        "community_oriented_helper": [1, 2, 4, 6, 8, 11, 16, 17],
        "hardship_responsibility_growth": [1, 2, 3, 4, 8, 11, 15, 18],
        "academically_strong_but_narrow": [1, 2, 3, 4, 5, 6, 15, 20],
        "polished_but_thin": [1, 2, 3, 4, 6, 8, 13, 21],
        "support_needed_promising": [1, 2, 4, 5, 11, 15, 18, 19],
        "borderline_manual_review": [1, 2, 3, 4, 6, 7, 17, 21],
    }
    return base[spec.archetype]


def answer_for_question(spec: CandidateSpec, question_index: int) -> str:
    if question_index == 1:
        return sanitize_visible_text(
            f"The most difficult period for me was when {spec.hardship} It changed how I think about responsibility because daily life stopped being theoretical and became something I had to help hold together. I coped by focusing on what I could do in front of me, even when the larger situation felt uncertain."
        )
    if question_index == 2:
        return sanitize_visible_text(routine_text(spec))
    if question_index == 3:
        return sanitize_visible_text(
            f"I worked toward a goal for a long time when I {spec.failed_goal} At first I felt frustrated because I wanted the result to prove something about me. Later I understood that the effort still changed how I work and what I notice about my own weak points."
        )
    if question_index == 4:
        return sanitize_visible_text(
            f"One initiative I took on my own was that I {spec.initiative} The idea came from repeatedly noticing that {spec.issue} {spec.initiative_outcome} It was not perfect, but it showed me that useful action can begin before everything is fully official."
        )
    if question_index == 5:
        return sanitize_visible_text(
            f"{decision_text(spec)} A recent example was deciding whether to apply to a more unfamiliar learning environment instead of staying with the most predictable option. I gathered what information I could, listened to advice, and then chose the path that seemed more demanding but also more meaningful."
        )
    if question_index == 6:
        return sanitize_visible_text(
            f"A situation where efficiency and what felt right were in tension happened when {spec.ethics_case} It would have been easier to go along with the shortcut, but I chose the slower option because I did not want convenience to become the reason I ignored my own standards."
        )
    if question_index == 7:
        return sanitize_visible_text(
            f"A problem that genuinely troubles me is that {spec.issue} I think the root cause is a mix of unequal access, low expectations, and institutions that are often slow to notice everyday problems until they become serious."
        )
    if question_index == 8:
        return sanitize_visible_text(
            f"My worldview shifted because {spec.worldview_trigger} That experience made me less passive and less likely to believe that only experts or officials are allowed to start useful change."
        )
    if question_index == 9:
        return sanitize_visible_text(
            f"In twenty years, I hope Kazakhstan looks more equal in opportunity and more practical in how it solves local problems. My generation should not only move toward opportunity but also create it. Personally, I would like to {spec.future_goal}."
        )
    if question_index == 10:
        return sanitize_visible_text(unpopular_text(spec))
    if question_index == 11:
        return sanitize_visible_text(
            f"One person or community I helped was when I {spec.help_example} What that gave me was a stronger sense that impact is often ordinary before it is visible, and that usefulness matters even when nobody turns it into a big story."
        )
    if question_index == 12:
        return sanitize_visible_text(core_values_text(spec))
    if question_index == 13:
        return sanitize_visible_text(
            f"The closest thing I have organized is that I {spec.initiative} What worked was that the need was real and people recognized it once they saw it. What did not work was sustainability, because the effort still depended too much on my reminders or energy. If I did it again, I would create clearer roles earlier."
        )
    if question_index == 14:
        return sanitize_visible_text(
            f"Before applying, I created value because I {spec.value_creation} That experience taught me that people do not need a perfect expert; they often need someone patient enough to make something understandable or functional."
        )
    if question_index == 15:
        return sanitize_visible_text(
            f"If I had $1,000 and one month, I would use it to {spec.future_goal} I would keep the plan practical and local so that people could see a direct result rather than only hear another speech about change."
        )
    if question_index == 16:
        return sanitize_visible_text(
            f"One inVision U value that already feels natural to me is collaboration through real usefulness, because a lot of what I have done comes from moments when I {spec.initiative} The part that would challenge me most is public confidence, because I still want to get better at explaining my ideas before they are fully polished."
        )
    if question_index == 17:
        return sanitize_visible_text(
            f"A time I had to work with people who thought differently from me was when I {spec.ethics_case} What made it hard was that the easier option looked normal to other people even though it did not feel right to me. I learned that disagreement is easier to handle when I can explain the concrete reason behind my choice."
        )
    if question_index == 18:
        return sanitize_visible_text(
            f"When pressure rises in a team, I usually become the person who looks for structure and the next practical step. That became clear to me when {spec.hardship} I do not become loud, but I do become more serious about what needs to be done and who might be getting left behind."
        )
    if question_index == 19:
        return sanitize_visible_text(
            f"If I joined inVision U tomorrow, the first local problem I would want to turn into a semester project would be that {spec.issue} I would choose it because I already understand the problem from close range and I can imagine a practical first version instead of only a slogan."
        )
    if question_index == 20:
        return sanitize_visible_text(
            f"My future classmates might not expect how much my background has taught me about patience, repetition, and making things useful with limited resources. A lot of that comes from how I {spec.value_creation} or from how I learned through {spec.hardship}"
        )
    if question_index == 21:
        return sanitize_visible_text(
            f"A time I changed my mind was after I {spec.failed_goal} At first I thought effort alone would be enough, but later I realized I also needed feedback, better structure, and more honest evaluation of what was actually working. After that, I became more willing to revise my approach instead of defending it."
        )
    raise ValueError(f"Unsupported question index: {question_index}")


def letter_for_spec(spec: CandidateSpec) -> str:
    if spec.archetype == "hidden_potential_low_polish":
        intro = choose_variant(
            spec,
            f"I grew up in {spec.city}, and one reason I am applying is that I keep noticing how {spec.issue}",
            f"I am from {spec.city}, where I learned early that {spec.issue}",
            f"Most of my reasons for applying come from life in {spec.city}, especially from seeing how {spec.issue}",
            f"What pushes me to apply from {spec.city} is something practical: {spec.issue}",
        )
        context = choose_variant(
            spec,
            f"At home, {spec.hardship}",
            f"My family life also shaped me early, because {spec.hardship}",
            f"A lot of my daily routine changed once {spec.hardship}",
        )
        action = choose_variant(
            spec,
            f"Instead of waiting for someone else, I {spec.initiative}",
            f"One concrete thing I started on my own was that I {spec.initiative}",
            f"The most useful action I took myself was that I {spec.initiative}",
        )
        paragraphs = [
            paragraph(
                intro,
                "I want an education that keeps ideas tied to real life instead of separating learning from everyday problems.",
            ),
            paragraph(
                context,
                "Because of that, I learned to do practical things without waiting for perfect conditions.",
            ),
            paragraph(
                action,
                spec.initiative_outcome,
            ),
            paragraph(
                f"I am interested in {spec.focus}.",
                choose_variant(
                    spec,
                    "What I want from inVision U is not prestige by itself.",
                    "What attracts me is not status, but the chance to learn in a place where useful work is taken seriously.",
                    "I am less interested in labels than in learning how to turn rough effort into something stronger and more lasting.",
                ),
                f"I want to {spec.future_goal}.",
            ),
        ]
        if spec.bucket == "long":
            paragraphs.insert(
                3,
                paragraph(
                    choose_variant(
                        spec,
                        "I know I still need to grow in English, confidence, and how I explain what I have done.",
                        "I do not always present myself in the strongest way, but I know growth becomes real only when someone enters a harder environment.",
                        "My application is more practical than polished, and that is exactly why I want to keep growing in a place that expects more from me.",
                    ),
                    "That is one reason I am applying to a place where I can grow with other people and not only by myself.",
                ),
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    if spec.archetype == "quiet_technical_builder":
        intro = choose_variant(
            spec,
            f"I am from {spec.city}, and most of my strongest learning has happened when I was trying to make or repair something with my own hands.",
            f"My background in {spec.city} pushed me toward hands-on learning long before I could explain it well in words.",
            f"I learned best in {spec.city} when I was taking apart, testing, or repairing something practical on my own.",
        )
        paragraphs = [
            paragraph(
                intro,
                f"My main interest is {spec.focus}.",
            ),
            paragraph(
                f"The local problem that keeps my attention is that {spec.issue}",
                "I usually start by trying a small prototype before I know exactly how to explain it.",
            ),
            paragraph(
                f"One example is that I {spec.initiative}",
                spec.initiative_outcome,
            ),
            paragraph(
                "I am quieter in groups than many applicants probably are, but I am persistent once I decide a problem is worth solving.",
                f"At inVision U, I want to {spec.future_goal}.",
            ),
        ]
        if spec.bucket == "long":
            paragraphs.insert(
                3,
                paragraph(
                    f"A difficult part of my life has been that {spec.hardship}",
                    "That period made me more serious about solving concrete problems instead of only collecting ideas.",
                ),
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    if spec.archetype == "community_oriented_helper":
        intro = choose_variant(
            spec,
            f"I am applying from {spec.city} because I want to study in a place where community work and practical projects are taken seriously.",
            f"From {spec.city}, I am applying because I have seen how much ordinary community life changes when even one person starts something useful.",
            f"I want to study at inVision U because life in {spec.city} taught me that small acts of inclusion can change how people see themselves.",
        )
        paragraphs = [
            paragraph(
                intro,
                f"I keep noticing that {spec.issue}",
            ),
            paragraph(
                f"Because of that, I {spec.initiative}",
                spec.initiative_outcome,
            ),
            paragraph(
                "What matters to me is not only being useful once, but helping people feel less left out or less alone in ordinary situations.",
                f"I {spec.help_example} That experience made me want to build more spaces where people can participate without fear.",
            ),
            paragraph(
                f"My long-term goal is to {spec.future_goal}.",
                "I think inVision U could help me turn local, people-centered efforts into stronger and more sustainable work.",
            ),
        ]
        if spec.bucket == "long":
            paragraphs.insert(
                2,
                paragraph(
                    f"A lot of my perspective also comes from the fact that {spec.hardship}",
                    "It made me less interested in status and more interested in whether people around me actually feel supported.",
                ),
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    if spec.archetype == "hardship_responsibility_growth":
        intro = choose_variant(
            spec,
            f"I come from {spec.city}, and one reason I want to study at inVision U is that my life has taught me early that growth is usually messy and practical, not ideal.",
            f"My reasons for applying from {spec.city} are tied to responsibility as much as ambition.",
            f"In {spec.city}, I learned early that growth rarely arrives in clean, simple conditions.",
        )
        paragraphs = [
            paragraph(
                intro,
                f"I care about {spec.focus}.",
            ),
            paragraph(
                f"A central part of my recent years was that {spec.hardship}",
                "That period changed how I organize my time, how I think about responsibility, and what I value in education.",
            ),
            paragraph(
                f"Even during that period, I still tried to act on what I could control. For example, I {spec.initiative}",
                spec.initiative_outcome,
            ),
            paragraph(
                f"I want to {spec.future_goal}.",
                "I do not think hardship alone should count as merit, but I do think it changed how seriously I now treat effort, systems, and care.",
            ),
        ]
        if spec.bucket == "long":
            paragraphs.insert(
                3,
                paragraph(
                    f"I {spec.help_example} I think experiences like that taught me that reliability is not a glamorous quality, but it is still a real one.",
                ),
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    if spec.archetype == "academically_strong_but_narrow":
        paragraphs = [
            paragraph(
                choose_variant(
                    spec,
                    f"I am applying from {spec.city} with a background that has been strongly shaped by {spec.focus}.",
                    f"My path in {spec.city} has been shaped mostly by disciplined academic work, especially around {spec.focus}.",
                    f"I am applying from {spec.city} after several years of living mostly inside structured academic goals tied to {spec.focus}.",
                ),
                "Most of my school life has been organized around disciplined study, competitions, and understanding difficult material well.",
            ),
            paragraph(
                f"One initiative I am quietly proud of is that I {spec.initiative}",
                spec.initiative_outcome,
            ),
            paragraph(
                f"The limitation in my profile is that I have spent much more time mastering structured academic tasks than testing myself in open-ended projects.",
                f"I want to {spec.future_goal}.",
            ),
        ]
        if spec.bucket != "short":
            paragraphs.append(
                paragraph(
                    f"I know this because {spec.failed_goal}",
                    "That experience showed me that improvement is real, but it is not the same as breadth.",
                )
            )
        if spec.bucket == "long":
            paragraphs.append(
                paragraph(
                    "I do not want to abandon rigor.",
                    "I want to keep rigor and add more range, more communication skill, and more willingness to work with people whose strengths are not the same as mine.",
                )
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    if spec.archetype == "polished_but_thin":
        paragraphs = [
            paragraph(
                choose_variant(
                    spec,
                    f"I am applying from {spec.city} because I believe young people in Kazakhstan need more imagination, confidence, and interdisciplinary collaboration.",
                    f"From {spec.city}, I am applying with a lot of energy around communication, initiative, and interdisciplinary collaboration.",
                    f"My motivation from {spec.city} comes from wanting to connect ideas, people, and opportunities more boldly than I have done so far.",
                ),
                f"My interests center on {spec.focus}.",
            ),
            paragraph(
                f"In school and outside it, I {spec.initiative}",
                spec.initiative_outcome,
            ),
            paragraph(
                f"What attracts me to inVision U is the chance to {spec.future_goal}.",
                "I care about energy, collaboration, and building the kind of communities where people take initiative instead of waiting for permission.",
            ),
        ]
        if spec.bucket != "short":
            paragraphs.append(
                paragraph(
                    f"I know that not every idea I have had has turned into something durable. For example, I {spec.failed_goal}",
                    "Still, I think the attempt matters because it showed me what kind of learning environment I am looking for.",
                )
            )
        if spec.bucket == "long":
            paragraphs.append(
                paragraph(
                    "I am especially interested in how communication changes what people believe is possible.",
                    "The more I read and talk across different settings, the more I feel that creative and social leadership matter as much as technical competence.",
                )
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    if spec.archetype == "support_needed_promising":
        intro = choose_variant(
            spec,
            f"I am applying from {spec.city}, and I know a lot of my growth has happened in small settings rather than visible ones.",
            f"I am from {spec.city}, and I am applying because I feel ready for a harder environment than the one that has shaped me so far.",
            f"My application comes from {spec.city}, where most of my progress has happened quietly and close to home.",
        )
        paragraphs = [
            paragraph(
                intro,
                f"What draws me most is {spec.focus}.",
            ),
            paragraph(
                f"A practical thing I tried was that I {spec.initiative}",
                spec.initiative_outcome,
            ),
            paragraph(
                f"A lot of my context is that {spec.hardship}",
                support_growth_text(spec),
            ),
            paragraph(
                f"My hope is to {spec.future_goal}.",
                choose_variant(
                    spec,
                    "I do not need a perfect path; I need one where effort, guidance, and patience actually matter.",
                    "What I need most now is a place where effort is taken seriously and guidance helps that effort grow into stronger work.",
                    "I am looking for a place where expectations are real, but so is the support needed to grow into them.",
                ),
            ),
        ]
        if spec.bucket == "long":
            paragraphs.insert(
                3,
                paragraph(
                    f"I {spec.help_example} That kind of work taught me that even when I do not feel impressive, I can still be useful in real situations.",
                ),
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    if spec.archetype == "borderline_manual_review":
        paragraphs = [
            paragraph(
                choose_variant(
                    spec,
                    f"I am applying from {spec.city} because I see my future in {spec.focus}.",
                    f"My application from {spec.city} is shaped by a strong pull toward {spec.focus}.",
                    f"I want to study at inVision U because I can imagine a future in {spec.focus}, even if I am still learning what that really demands.",
                ),
                f"The issue that motivates me most is that {spec.issue}",
            ),
            paragraph(
                f"Over the last years, I {spec.initiative}",
                spec.initiative_outcome,
            ),
            paragraph(
                "I am especially drawn to inVision U because it values interdisciplinary problem solving, communication, and the courage to test ideas publicly.",
                f"My long-term goal is to {spec.future_goal}.",
            ),
        ]
        if spec.bucket != "short":
            paragraphs.append(
                paragraph(
                    f"I {spec.failed_goal}",
                    "That experience pushed me to refine how I think about initiative, discipline, and influence.",
                )
            )
        return sanitize_visible_text("\n\n".join(paragraphs))

    raise ValueError(f"Unsupported archetype: {spec.archetype}")


def interview_for_spec(spec: CandidateSpec) -> str:
    if not spec.has_interview:
        return ""
    if spec.archetype == "quiet_technical_builder":
        return sanitize_visible_text(
            choose_variant(
                spec,
                f"""
                I am more comfortable building than speaking, so interviews are not the easiest format for me. Still, I think my strongest signal is practical: when I notice a problem, I usually start testing a fix instead of only discussing it. In my case that has included how I {spec.initiative}

                What I want from inVision U is a chance to work with people who think differently from me and can challenge my blind spots. Right now, I am good at persistence and at learning by doing, but I am slower at presentation, feedback, and asking for help early enough. I want to {spec.future_goal}
                """,
                f"""
                I usually speak more clearly through work than through slogans. When I notice a practical problem, I tend to test a fix first and explain it second. A good example is that I {spec.initiative}

                At inVision U, I want to keep that hands-on side of myself while becoming better at teamwork, presentation, and early feedback. I do not want to stay the person who can build quietly but cannot yet explain the larger idea well.
                """,
                f"""
                In group settings, I am rarely the loudest person, but I am often the one still adjusting the details after others move on. That fits how I learned through practical work like when I {spec.initiative}

                I am applying because I want a more demanding environment that keeps technical curiosity connected to people, teamwork, and real constraints. I think that is the step I need now.
                """,
            )
        )
    if spec.archetype == "community_oriented_helper":
        return sanitize_visible_text(
            choose_variant(
                spec,
                f"""
                My main strength in groups is usually not being the loudest person, but making sure people feel able to contribute. A lot of what I have done already comes from seeing that {spec.issue} That is why I {spec.initiative}

                At inVision U, I want to learn how to keep the human side of teamwork while also becoming more structured. I think I already know how to start trust. What I still need is stronger planning, clearer delegation, and a better sense of how to make local efforts last.
                """,
                f"""
                In teams, I usually notice who has gone quiet, who feels left behind, and which practical tasks nobody has claimed yet. That way of working comes directly from experiences like when I {spec.initiative}

                What I want to improve is scale. I can help something feel human and welcoming, but I still want to learn how to build systems that keep working when one person gets tired.
                """,
                f"""
                I am usually the kind of teammate who keeps paying attention to atmosphere as well as tasks. I have seen that when people feel safe, they contribute more honestly. That belief became stronger through work like when I {spec.initiative}

                At inVision U, I want to combine that instinct with stronger project design, better follow-through, and more confidence in leading openly.
                """,
            )
        )
    if spec.archetype == "hardship_responsibility_growth":
        return sanitize_visible_text(
            choose_variant(
                spec,
                f"""
                I do not think hardship itself should automatically make a person stand out. What matters more is what you become responsible for and what habits you build because of it. In my case, {spec.hardship}

                I am applying because I want to be in a place where the next step after endurance is actual growth. I know I can carry weight. I now want to learn how to turn that into better thinking, stronger collaboration, and more intentional action.
                """,
                f"""
                A lot of my maturity did not come from planned leadership roles. It came from responsibilities that arrived early and could not be postponed. In my case, {spec.hardship}

                I am now looking for a place where responsibility is not only about surviving the week, but also about becoming more thoughtful, more capable, and more useful in a team.
                """,
            )
        )
    if spec.archetype == "academically_strong_but_narrow":
        return sanitize_visible_text(
            choose_variant(
                spec,
                f"""
                My background is strongest in disciplined academic work. I like difficult material, and I can work independently for a long time. The part I want to develop is range. I know that I have spent more time in controlled academic settings than in real team projects.

                What attracts me to inVision U is that it might help me keep my rigor while also forcing me to communicate more openly, take more practical risks, and test whether my strengths are useful outside narrow academic contexts.
                """,
                """
                I am comfortable with difficult material and long periods of focused work. Where I still need to grow is outside the controlled structure of academic tasks. I want more experience where ideas have to survive teamwork, feedback, and real-world ambiguity.

                That is why inVision U attracts me. I do not want to lose rigor; I want to see whether rigor can become more useful when it meets a wider range of people and problems.
                """,
            )
        )
    if spec.archetype == "polished_but_thin":
        return sanitize_visible_text(
            choose_variant(
                spec,
                f"""
                I see myself as someone who can energize people and connect ideas across different settings. The reason I am applying is that I want to stop having good intentions and early-stage experiments only, and start building things that are more durable.

                I think inVision U could help me with that because it combines collaboration, critique, and real-world execution. I know that some of my work so far has been stronger in communication than in follow-through, and that is exactly what I want to improve.
                """,
                """
                I usually find it easy to speak about ideas, motivate people, and imagine what a bigger initiative could become. The harder part for me has been building something that keeps working after the first burst of energy.

                I think inVision U could challenge me in the right way because it values execution, critique, and teamwork, not only strong framing.
                """,
            )
        )
    if spec.archetype == "support_needed_promising":
        return sanitize_visible_text(
            choose_variant(
                spec,
                f"""
                Speaking in interviews is not the easiest setting for me, but I still wanted to try because I know growth will require me to step forward more often. What I think I bring already is seriousness, patience, and a real wish to become useful.

                If I joined inVision U, I would want to grow in confidence, language, and teamwork without losing the careful way I already work. I know structure helps me use effort better, and I think I would respond well to that kind of environment.
                """,
                f"""
                I usually speak more carefully than quickly, especially when I am in a new setting. But once I trust a group, I become steady and reliable. That is also how I approached work like when I {spec.initiative}

                What I want from inVision U is not an easier path. I want clearer challenge, stronger feedback, and a place where I have to become more visible than I have been so far.
                """,
                """
                I am still learning how to speak with confidence before I feel fully ready, and that is one reason this interview matters to me. I do not want hesitation to become my permanent style.

                At the same time, I know I can work seriously, respond to feedback, and grow when expectations are clear. That combination is why I think a more demanding environment would help me.
                """,
            )
        )
    if spec.archetype == "borderline_manual_review":
        return sanitize_visible_text(
            choose_variant(
                spec,
                """
                I am especially motivated by the idea of building something larger than myself through communication, leadership, and interdisciplinary collaboration. What excites me is the possibility of taking local ideas and turning them into broader platforms or programs.

                At the same time, I know I am still learning how to move from strong framing to strong execution. That is part of why I want an environment that pushes me to be more rigorous about what impact actually looks like in practice.
                """,
                """
                I am drawn to environments where ideas are not only discussed but publicly tested. I enjoy the part where a project gets framed, explained, and shared. The harder part for me has been proving that the work underneath is as strong as the framing around it.

                That is one reason I want a more demanding environment. I want to see which of my instincts are real strengths and which still need much more discipline behind them.
                """,
            )
        )
    return sanitize_visible_text(
        choose_variant(
            spec,
            f"""
            I am applying because I want to learn in a place where ideas are expected to become something real. Right now my experience is still local and limited, but it is sincere. I think inVision U could help me {spec.future_goal}
            """,
            f"""
            What attracts me most is the chance to study somewhere that expects people to build, reflect, and improve in public. I think inVision U could help me {spec.future_goal}
            """,
        )
    )


def manifest_signals(spec: CandidateSpec) -> list[str]:
    base = {
        "hidden_potential_low_polish": ["credible small-scale initiative", "low-resource responsibility", "community usefulness"],
        "quiet_technical_builder": ["hands-on making", "self-directed problem solving", "practical technical value"],
        "community_oriented_helper": ["people-centered initiative", "inclusion work", "community trust building"],
        "hardship_responsibility_growth": ["family responsibility", "resilience under pressure", "mature perspective"],
        "academically_strong_but_narrow": ["high academic discipline", "teaching/tutoring support", "strong analytical ability"],
        "polished_but_thin": ["strong communication polish", "broad ambition", "some school-level initiative"],
        "support_needed_promising": ["modest but real initiative", "growth potential", "visible support need"],
        "borderline_manual_review": ["high polish or large claims", "plausible interest area", "ambiguous grounding"],
    }
    return base[spec.archetype]


def manifest_risks(spec: CandidateSpec) -> list[str]:
    base = {
        "hidden_potential_low_polish": ["understated self-presentation", "limited formal achievements"],
        "quiet_technical_builder": ["weak teamwork evidence", "communication gap"],
        "community_oriented_helper": ["small-scale outcomes", "systems execution still emerging"],
        "hardship_responsibility_growth": ["initiative may be overshadowed by survival duties", "academic readiness uncertainty"],
        "academically_strong_but_narrow": ["limited breadth", "few open-ended projects"],
        "polished_but_thin": ["generic impact framing", "thin concrete evidence"],
        "support_needed_promising": ["confidence and language gaps", "underprepared transition risk"],
        "borderline_manual_review": ["possible inconsistency or exaggeration", "thin grounding for stronger claims"],
    }
    return base[spec.archetype]


def behavioral_signals_for_spec(spec: CandidateSpec) -> dict[str, object]:
    completion = {
        "short": 0.88,
        "medium": 0.94,
        "long": 0.98,
    }[spec.bucket]
    if spec.ambiguity == "borderline":
        completion -= 0.02
    if spec.ambiguity == "hard":
        completion -= 0.04
    if spec.archetype == "borderline_manual_review":
        completion -= 0.04
    if spec.archetype == "polished_but_thin" and spec.bucket != "short":
        completion -= 0.01
    skipped = {"short": 2, "medium": 1, "long": 0}[spec.bucket]
    if spec.archetype == "borderline_manual_review":
        skipped += 1
    returned = spec.bucket != "short" or spec.archetype in {"polished_but_thin", "borderline_manual_review"}
    return {
        "completion_rate": round(max(0.84, completion), 2),
        "returned_to_edit": returned,
        "skipped_optional_questions": skipped,
    }


def total_word_count(record: dict[str, object]) -> int:
    text_inputs = record["text_inputs"]
    parts: list[str] = [text_inputs["motivation_letter_text"]]
    for item in text_inputs["motivation_questions"]:
        parts.append(item["answer"])
    parts.append(text_inputs.get("interview_text") or "")
    return len(re.findall(r"\b\w+\b", " ".join(parts)))


def length_bucket_from_words(words: int) -> str:
    if words < 450:
        return "short"
    if words <= 720:
        return "medium"
    return "long"


def generate_candidate(spec: CandidateSpec, submitted_at: str) -> tuple[dict[str, object], dict[str, object]]:
    question_ids = question_indices(spec)
    motivation_questions = [
        {"question": QUESTION_BANK[idx], "answer": answer_for_question(spec, idx)}
        for idx in question_ids
    ]
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
            "motivation_letter_text": letter_for_spec(spec),
            "motivation_questions": motivation_questions,
            "interview_text": interview_for_spec(spec),
        },
        "behavioral_signals": behavioral_signals_for_spec(spec),
        "metadata": {
            "source": "synthetic_batch_v1",
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
        combined_answers = "\n\n".join(
            f"Q: {item['question']}\nA: {item['answer']}" for item in answers
        )
        row = {
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
            "motivation_questions_text": combined_answers,
            "interview_text": text_inputs.get("interview_text") or "",
        }
        rows.append(row)
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
    "support_need_phrase": r"\bsupport need is central\b",
    "candidate_meta_phrase": r"\bthe candidate is\b",
    "underexposed_meta_phrase": r"\bvery underexposed\b",
    "would_benefit_meta_phrase": r"\bwould benefit from\b",
    "needs_clearer_meta_phrase": r"\bneeds clearer\b",
    "easy_to_overlook_meta_phrase": r"\beasy to overlook\b",
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
    pattern = re.compile(r"\b(?:his|her|he|she|him|herself|himself)\b", re.IGNORECASE)
    hits: list[str] = []
    for record in records:
        normalized_fields: list[str] = []
        for field in visible_text_fields(record):
            cleaned = re.sub(r"\bmy father lost his job\b", "my father lost a job", field, flags=re.IGNORECASE)
            cleaned = re.sub(r"\bmy mother lost her job\b", "my mother lost a job", cleaned, flags=re.IGNORECASE)
            normalized_fields.append(cleaned)
        if any(pattern.search(field) for field in normalized_fields):
            hits.append(record["candidate_id"])
    return hits


def repeated_opening_hits(records: list[dict[str, object]], threshold: int = 4) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for record in records:
        letter = (record["text_inputs"].get("motivation_letter_text") or "").strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", letter)[0].strip()
        if not first_sentence:
            continue
        buckets.setdefault(first_sentence, []).append(record["candidate_id"])
    return {opening: ids for opening, ids in buckets.items() if len(ids) > threshold}


def repeated_interview_opening_hits(records: list[dict[str, object]], threshold: int = 4) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for record in records:
        interview = (record["text_inputs"].get("interview_text") or "").strip()
        if not interview:
            continue
        first_sentence = re.split(r"(?<=[.!?])\s+", interview)[0].strip()
        if not first_sentence:
            continue
        buckets.setdefault(first_sentence, []).append(record["candidate_id"])
    return {opening: ids for opening, ids in buckets.items() if len(ids) > threshold}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2026, 2, 3, 9, 15, tzinfo=timezone.utc)
    raw_records: list[dict[str, object]] = []
    sanitized_records: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []
    validation_fixes: list[str] = []

    for idx, spec in enumerate(SPECS):
        submitted_at = (base_time + timedelta(hours=9 * idx)).isoformat().replace("+00:00", "Z")
        raw, sanitized = generate_candidate(spec, submitted_at)
        raw = CandidateInput.model_validate(raw).model_dump(mode="json", exclude_none=False)
        CandidateInput.model_validate(sanitized)

        raw_records.append(raw)
        sanitized_records.append(sanitized)
        manifest_records.append(
            {
                "candidate_id": spec.candidate_id,
                "intended_archetype": spec.archetype,
                "intended_ambiguity": spec.ambiguity,
                "intended_primary_signals": manifest_signals(spec),
                "intended_primary_risks": manifest_risks(spec),
                "generator_notes": spec.risk_note,
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
    leakage_terms = [
        "hidden_potential_low_polish",
        "quiet_technical_builder",
        "community_oriented_helper",
        "hardship_responsibility_growth",
        "academically_strong_but_narrow",
        "polished_but_thin",
        "support_needed_promising",
        "borderline_manual_review",
    ]

    archetype_counts: dict[str, int] = {}
    ambiguity_counts: dict[str, int] = {}
    planned_text_length_counts: dict[str, int] = {}
    for spec in SPECS:
        archetype_counts[spec.archetype] = archetype_counts.get(spec.archetype, 0) + 1
        ambiguity_counts[spec.ambiguity] = ambiguity_counts.get(spec.ambiguity, 0) + 1
        planned_text_length_counts[spec.bucket] = planned_text_length_counts.get(spec.bucket, 0) + 1

    actual_text_length_counts = {"short": 0, "medium": 0, "long": 0}
    for record in sanitized_records:
        actual_text_length_counts[length_bucket_from_words(total_word_count(record))] += 1

    with_interview_count = sum(1 for spec in SPECS if spec.has_interview)
    without_interview_count = len(SPECS) - with_interview_count

    validation_status = {
        "candidate_input_schema_raw": True,
        "candidate_input_schema_sanitized": True,
        "unique_candidate_ids": len(ids) == len(set(ids)),
        "raw_sanitized_one_to_one": ids == raw_to_sanitized_ids,
        "sanitized_has_no_metadata": all("metadata" not in item for item in sanitized_records),
        "reviewer_pack_has_no_archetype_leakage": not any(term in sanitized_payload_text for term in leakage_terms),
        "english_only_heuristic": not re.search(r"[А-Яа-яЁё]", sanitized_payload_text),
        "near_duplicate_pairs": near_duplicate_pairs,
        "visible_leakage_hits": visible_leakage,
        "person_drift_hits": person_drift,
        "repeated_letter_openings_over_threshold": repeated_letter_openings,
        "repeated_interview_openings_over_threshold": repeated_interview_openings,
        "actual_text_length_counts": actual_text_length_counts,
        "planned_text_length_counts": planned_text_length_counts,
        "planned_bucket_validation_fixes": validation_fixes,
    }

    summary = {
        "candidate_count": len(SPECS),
        "archetype_counts": archetype_counts,
        "ambiguity_counts": ambiguity_counts,
        "text_length_counts": actual_text_length_counts,
        "with_interview_count": with_interview_count,
        "without_interview_count": without_interview_count,
        "validation_status": validation_status,
        "notes": [
            "Visible payloads stay inside the frozen public CandidateInput contract.",
            "Reviewer pack removes metadata and keeps only candidate_id, structured_data, text_inputs, and behavioral_signals.",
            "Text length counts reflect actual generated payload lengths; planned bucket targets are also included in validation_status.",
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
        "intended_use": "synthetic admissions batch v1 for human annotation and scorer stress-testing",
    }
    with PACK_MANIFEST_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(pack_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
