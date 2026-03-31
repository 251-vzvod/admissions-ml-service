# Backend To ML Contract

Этот файл можно отправить бэкендеру как целевой контракт интеграции с ML-сервисом.

## Что важно

- ML-сервис сейчас принимает:
  - `POST /score`
  - `POST /score/batch`
- Для интеграции с вашим `backend` достаточно `POST /score` на один application form.
- Вызывать ML лучше после того, как уже готовы:
  - `motivation_letter_text`
  - `video_presentation_transcript_text` или `video_interview_transcript_text`, если они реально извлечены
- Если текстов ещё нет, ML всё равно ответит, но confidence будет слабее, а recommendation может уйти в `incomplete_application` или `insufficient_evidence`.

## Целевой request в ML-сервис

Бэкенд должен собирать и отправлять **canonical payload**:

```json
{
  "candidate_id": "form_123",
  "consent": true,
  "profile": {
    "academics": {
      "english_proficiency": {
        "type": "ielts",
        "score": 6.5
      },
      "school_certificate": {
        "type": "unt",
        "score": 110
      }
    },
    "materials": {
      "documents": [
        "https://storage.example.com/motivation-letter.pdf"
      ],
      "video_presentation_link": "https://youtube.com/watch?v=abc123"
    },
    "narratives": {
      "motivation_letter_text": "Extracted text from motivation letter",
      "motivation_questions": [],
      "interview_text": null,
      "video_interview_transcript_text": null,
      "video_presentation_transcript_text": "Transcript extracted from the presentation video"
    },
    "process_signals": {
      "completion_rate": 1.0,
      "returned_to_edit": false,
      "skipped_optional_questions": 0,
      "meaningful_answers_count": 0,
      "scenario_depth": null
    },
    "metadata": {
      "source": "backend",
      "submitted_at": "2026-04-01T12:00:00Z",
      "scoring_version": null
    }
  }
}
```

## Как маппить из вашего backend

Источник: [application_form.py](/C:/Users/Admin/Desktop/decentrathon/ml-service/backend/src/api/http/v1/endpoints/application_form.py) и [req.py](/C:/Users/Admin/Desktop/decentrathon/ml-service/backend/src/schemas/application_form/req.py)

Маппинг:

- `candidate_id`
  - использовать `application_form.id`
  - рекомендованный формат: `"form_<id>"`

- `consent`
  - использовать `personal_data_consent`

- `profile.academics.english_proficiency.type`
  - из `education.english_proficiency.type`

- `profile.academics.english_proficiency.score`
  - из `education.english_proficiency.score`

- `profile.academics.school_certificate.type`
  - из `education.school_certificate.type`

- `profile.academics.school_certificate.score`
  - из `education.school_certificate.score`

- `profile.materials.documents`
  - положить сюда `motivation_letter.link`, если ссылка есть

- `profile.materials.video_presentation_link`
  - из `motivation.presentation.link`

- `profile.narratives.motivation_letter_text`
  - из `motivation.motivation_letter.text`

- `profile.narratives.video_presentation_transcript_text`
  - из `motivation.presentation.text`

- `profile.narratives.video_interview_transcript_text`
  - пока `null`, если у вас нет отдельного интервью

- `profile.narratives.interview_text`
  - пока `null`, если у вас нет отдельного interview stage

- `profile.narratives.motivation_questions`
  - пока `[]`, если у вас нет этого блока в backend form

- `profile.process_signals.completion_rate`
  - если форма полностью закрыта, ставить `1.0`
  - если есть свой status/progress, маппить в `0..1`

- `profile.metadata.source`
  - `"backend"`

- `profile.metadata.submitted_at`
  - `application_form.created_at` или фактическое submit time в ISO формате

## Что бэку не надо отправлять в ML

- `first_name`
- `last_name`
- `patronymic`
- `birth_date`
- `gender`
- `citizenship`
- `phone`
- `whatsapp`
- `instagram`
- `telegram`

Это не нужно для merit scoring. Лучше вообще не тащить эти поля в ML request.

## Response от ML-сервиса, который бэк должен сохранить/проксировать

Бэкенду стоит сохранять или проксировать только публичные поля:

```json
{
  "candidate_id": "form_123",
  "scoring_run_id": "run_abc",
  "scoring_version": "v1.4.0",
  "eligibility_status": "conditionally_eligible",
  "eligibility_reasons": [],
  "merit_score": 72,
  "confidence_score": 61,
  "authenticity_risk": 37,
  "recommendation": "review_priority",
  "review_flags": [
    "low_confidence"
  ],
  "hidden_potential_score": 68,
  "support_needed_score": 44,
  "shortlist_priority_score": 71,
  "evidence_coverage_score": 58,
  "trajectory_score": 74,
  "committee_cohorts": [
    "Hidden potential"
  ],
  "why_candidate_surfaced": [
    "Growth and agency signals are stronger than presentation polish."
  ],
  "what_to_verify_manually": [
    "Ask for one concrete example with measurable outcome."
  ],
  "suggested_follow_up_question": "What changed after your first attempt did not work?",
  "evidence_highlights": [
    {
      "claim": "Helped classmates prepare for exams",
      "support_level": "supported",
      "source": "motivation_letter_text",
      "snippet": "I shared notes with classmates before exams.",
      "support_score": 78,
      "rationale": "Concrete action tied to other people."
    }
  ],
  "top_strengths": [
    "Clear initiative and agency markers in self-driven actions."
  ],
  "main_gaps": [
    "Specificity is limited; more concrete examples and outcomes would improve reliability."
  ],
  "explanation": {
    "summary": "This profile was scored using deterministic feature extraction with deterministic internal scoring.",
    "scoring_notes": {
      "potential": "Potential axis reflects growth, resilience, initiative, and fit signals.",
      "motivation": "Motivation axis combines clarity, fit, and grounded evidence.",
      "confidence": "Confidence score measures reliability of this assessment, not candidate quality.",
      "authenticity_risk": "Authenticity risk is a review-risk signal.",
      "recommendation": "Recommendation is a workflow routing label for committee review."
    }
  }
}
```

## Batch contract

Если бэк захочет гонять пачку:

`POST /score/batch`

```json
{
  "candidates": [
    {
      "candidate_id": "form_123",
      "consent": true,
      "profile": {
        "academics": {},
        "materials": {},
        "narratives": {},
        "process_signals": {},
        "metadata": {}
      }
    }
  ]
}
```

Batch response:

```json
{
  "scoring_run_id": "run_abc",
  "scoring_version": "v1.4.0",
  "count": 1,
  "ranked_candidate_ids": [
    "form_123"
  ],
  "shortlist_candidate_ids": [],
  "hidden_potential_candidate_ids": [],
  "support_needed_candidate_ids": [],
  "authenticity_review_candidate_ids": [],
  "results": []
}
```

## Главные ограничения

- `motivation_questions[].id` отправлять нельзя
- лишние top-level поля отправлять нельзя
- payload должен быть либо canonical, либо legacy-совместимый без лишних полей
- `authenticity_risk` это review signal, не verdict
- `recommendation` это routing label, не admission decision

## Что сейчас отсутствует в backend

По текущему коду backend:

- прямой интеграции с ML-сервисом пока нет
- interview text отсутствует
- motivation questions отсутствуют
- хранение ML response пока не видно

То есть бэкендеру сейчас нужно:

1. собрать canonical ML payload
2. вызвать `POST /score`
3. сохранить публичный ML response рядом с application form или в отдельной scoring таблице
