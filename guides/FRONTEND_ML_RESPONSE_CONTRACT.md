# Frontend ML Response Contract

Этот файл можно отправить фронтендеру как точный contract по тому, что реально приходит из ML-сервиса.

## Главное

Сейчас фронт в `invision-front` сидит на mock data, а не на живом ML API.

Это видно по [api.ts](/C:/Users/Admin/Desktop/decentrathon/ml-service/invision-front/src/features/applicants/api.ts): там используются `MOCK_APPLICANT_PROFILES`.

Поэтому ниже не "как уже работает", а "что нужно ожидать от реального ML response".

Отдельно: у фронта сейчас есть ещё и route mismatch с backend.

- [application/route.ts](/C:/Users/Admin/Desktop/decentrathon/ml-service/invision-front/src/app/api/application/route.ts) шлёт в `/applications`
- [file/route.ts](/C:/Users/Admin/Desktop/decentrathon/ml-service/invision-front/src/app/api/file/route.ts) шлёт в `/file`

А backend реально поднимает:

- `POST /api/v1/forms`
- `POST /api/v1/s3/upload`
- `DELETE /api/v1/s3/delete`

Это не ML contract, но это надо починить, иначе интеграция всё равно не взлетит.

## Живые public endpoints ML-сервиса

- `GET /health`
- `POST /score`
- `POST /score/batch`

## Реальный public response для одного кандидата

```ts
type Recommendation =
  | 'invalid'
  | 'incomplete_application'
  | 'insufficient_evidence'
  | 'review_priority'
  | 'manual_review_required'
  | 'standard_review'

type ReviewFlag =
  | 'eligibility_gate'
  | 'low_confidence'
  | 'insufficient_evidence'
  | 'low_evidence_density'
  | 'moderate_authenticity_risk'
  | 'high_authenticity_risk'
  | 'contradiction_risk'
  | 'possible_contradiction'
  | 'polished_but_empty_pattern'
  | 'high_polished_but_empty'
  | 'high_genericness'
  | 'cross_section_mismatch'
  | 'section_mismatch'
  | 'missing_required_materials'
  | 'auxiliary_ai_generation_signal'

type ClaimEvidenceItem = {
  claim: string
  support_level: string
  source: string
  snippet: string
  support_score: number
  rationale: string
}

type ScoreResponse = {
  candidate_id: string
  scoring_run_id: string
  scoring_version: string
  eligibility_status: string
  eligibility_reasons: string[]
  merit_score: number
  confidence_score: number
  authenticity_risk: number
  recommendation: Recommendation
  review_flags: ReviewFlag[]
  hidden_potential_score: number
  support_needed_score: number
  shortlist_priority_score: number
  evidence_coverage_score: number
  trajectory_score: number
  committee_cohorts: string[]
  why_candidate_surfaced: string[]
  what_to_verify_manually: string[]
  suggested_follow_up_question: string | null
  evidence_highlights: ClaimEvidenceItem[]
  top_strengths: string[]
  main_gaps: string[]
  explanation: {
    summary: string
    scoring_notes: {
      potential: string
      motivation: string
      confidence: string
      authenticity_risk: string
      recommendation: string
    }
  }
}
```

## Реальный public batch response

```ts
type BatchScoreResponse = {
  scoring_run_id: string
  scoring_version: string
  count: number
  ranked_candidate_ids: string[]
  shortlist_candidate_ids: string[]
  hidden_potential_candidate_ids: string[]
  support_needed_candidate_ids: string[]
  authenticity_review_candidate_ids: string[]
  results: ScoreResponse[]
}
```

## Что фронту можно уверенно показывать

- `candidate_id`
- `merit_score`
- `confidence_score`
- `authenticity_risk`
- `recommendation`
- `hidden_potential_score`
- `support_needed_score`
- `shortlist_priority_score`
- `trajectory_score`
- `evidence_coverage_score`
- `committee_cohorts`
- `why_candidate_surfaced`
- `what_to_verify_manually`
- `suggested_follow_up_question`
- `evidence_highlights`
- `top_strengths`
- `main_gaps`
- `explanation`

## Что фронту НЕ надо ожидать как public contract

Эти поля в живом ML response скрыты:

- `feature_snapshot`
- `llm_metadata`
- `llm_rubric_assessment`
- `merit_breakdown`
- `semantic_rubric_scores`
- `supported_claims`
- `weakly_supported_claims`
- `uncertainties`
- `authenticity_review_reasons`
- `ai_detector`
- `evidence_spans`
- `prompt_version`
- `extractor_version`

То есть если фронт будет завязываться на них, он сломается.

## Главные mismatch'и с текущим фронтом

Источник: [types.ts](/C:/Users/Admin/Desktop/decentrathon/ml-service/invision-front/src/features/applicants/types.ts)

### 1. `ApplicantProfile` сейчас ожидает много лишнего

Текущий фронтовый тип ждёт поля, которых public API не отдаёт:

- `prompt_version`
- `extractor_version`
- `feature_snapshot`
- `llm_metadata`
- `merit_breakdown`
- `semantic_rubric_scores`
- `uncertainties`
- `authenticity_review_reasons`
- `ai_detector`
- `evidence_spans`

Это нужно убрать из обязательного runtime contract.

### 2. `evidence_highlights` сейчас вообще не типизирован как основная public evidence surface

Во фронте есть `EvidenceSpan`, но public API теперь отдаёт не `evidence_spans`, а `evidence_highlights`.

Фронту нужно опираться именно на `evidence_highlights`.

### 3. Фронт всё ещё мыслит через старую expanded schema

Например:

- `MeritBreakdown` во фронте не содержит `community_values`, хотя у нас именно такая ось есть внутри
- `SemanticRubricScores` во фронте не соответствует текущей внутренней реальности

Но это даже не главный вопрос: эти поля вообще не должны быть обязательной частью live contract.

### 4. Dashboard уже хорошо совпадает по главным полям

Хорошая новость:

По UI-ключам фронт уже использует именно то, что реально есть в public API:

- `candidate_id`
- `merit_score`
- `confidence_score`
- `authenticity_risk`
- `recommendation`

То есть для таблицы и базового ranking view интеграция несложная.

## Минимальный runtime type для фронта

Если фронтендер хочет быстро перейти с mock data на live contract, минимальный безопасный тип должен быть таким:

```ts
export type ApplicantProfile = {
  candidate_id: string
  scoring_run_id: string
  scoring_version: string
  eligibility_status: string
  eligibility_reasons: string[]
  merit_score: number
  confidence_score: number
  authenticity_risk: number
  recommendation:
    | 'invalid'
    | 'incomplete_application'
    | 'insufficient_evidence'
    | 'review_priority'
    | 'manual_review_required'
    | 'standard_review'
  review_flags: string[]
  hidden_potential_score: number
  support_needed_score: number
  shortlist_priority_score: number
  evidence_coverage_score: number
  trajectory_score: number
  committee_cohorts: string[]
  why_candidate_surfaced: string[]
  what_to_verify_manually: string[]
  suggested_follow_up_question: string | null
  evidence_highlights: Array<{
    claim: string
    support_level: string
    source: string
    snippet: string
    support_score: number
    rationale: string
  }>
  top_strengths: string[]
  main_gaps: string[]
  explanation: {
    summary: string
    scoring_notes: {
      potential: string
      motivation: string
      confidence: string
      authenticity_risk: string
      recommendation: string
    }
  }
}
```

## Какой batch payload фронт может ожидать от бэка

Если бэк будет просто проксировать ML batch:

```ts
type ApplicantsRankingResponse = {
  scoring_run_id: string
  scoring_version: string
  count: number
  ranked_candidate_ids: string[]
  shortlist_candidate_ids: string[]
  hidden_potential_candidate_ids: string[]
  support_needed_candidate_ids: string[]
  authenticity_review_candidate_ids: string[]
  results: ApplicantProfile[]
}
```

## Практический вывод для фронтендера

Нужно сделать три вещи:

1. перестать считать `mock ApplicantProfile` равным live ML response
2. сократить runtime type до public contract
3. использовать `evidence_highlights`, а не `evidence_spans`
