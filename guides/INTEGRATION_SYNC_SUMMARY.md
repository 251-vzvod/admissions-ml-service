# Integration Sync Summary

Короткий файл, который можно отправить обоим сразу.

## Текущее состояние

- `backend` пока не интегрирован с ML-сервисом
- `frontend` пока не интегрирован с live ML response и использует mock data
- значит сейчас нужен не "bug fix", а синхронизация контрактов

## Что отправить бэкендеру

Файл:

- [BACKEND_TO_ML_CONTRACT.md](/C:/Users/Admin/Desktop/decentrathon/ml-service/guides/BACKEND_TO_ML_CONTRACT.md)

Короткое сообщение:

```text
Ниже точный contract для вызова ML-сервиса.
Нужно собирать canonical payload и вызывать POST /score после того, как готовы motivation_letter_text и transcript.
Личные поля в ML не отправляем.
```

## Что отправить фронтендеру

Файл:

- [FRONTEND_ML_RESPONSE_CONTRACT.md](/C:/Users/Admin/Desktop/decentrathon/ml-service/guides/FRONTEND_ML_RESPONSE_CONTRACT.md)

Короткое сообщение:

```text
Ниже live response contract ML-сервиса.
Текущий mock ApplicantProfile не равен реальному API.
Нужно перейти на public contract и опираться на evidence_highlights, а не на старые скрытые поля.
```

## Самые важные синхронизационные точки

### Для backend

- отправлять только `candidate_id`, `consent`, `profile`
- ждать extracted texts, а не только ссылки на файлы
- не тащить personal/demographic/contact fields в ML request

### Для frontend

- использовать только public ML fields
- не ждать `feature_snapshot`, `llm_metadata`, `semantic_rubric_scores` как обязательные runtime поля
- использовать `evidence_highlights` как главную evidence surface

## Критические текущие mismatch'и

### Front -> Backend route mismatch

Во фронте сейчас:

- [application/route.ts](/C:/Users/Admin/Desktop/decentrathon/ml-service/invision-front/src/app/api/application/route.ts) шлёт в `BACKEND_API + /applications`
- [file/route.ts](/C:/Users\Admin/Desktop/decentrathon/ml-service/invision-front/src/app/api/file/route.ts) шлёт в `BACKEND_API + /file`

Но в backend реально есть:

- `POST /api/v1/forms`
- `POST /api/v1/s3/upload`
- `DELETE /api/v1/s3/delete`

То есть это надо синхронизировать отдельно от ML.

### Front runtime type mismatch

Во фронте текущий `ApplicantProfile` не равен реальному public ML response.

### Backend ML integration missing

В backend пока не видно прямого вызова ML-сервиса и хранения scoring response.

## Главное архитектурное правило

Цепочка должна быть такой:

`frontend form -> backend storage/extraction -> backend calls ML -> backend/front consume ML public response`

Не наоборот.
