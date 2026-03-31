# Canonical Applicant Payload

The runtime contract is now centered on one top-level object: `profile`.

Minimum useful request:

```json
{
  "candidate_id": "cand_001",
  "profile": {
    "narratives": {
      "motivation_letter_text": "I started a peer tutoring group, changed the format after the first month failed, and kept improving it."
    }
  },
  "consent": true
}
```

Full example:

```json
{
  "candidate_id": "cand_001",
  "profile": {
    "academics": {
      "english_proficiency": {
        "type": "ielts",
        "score": 7.0
      },
      "school_certificate": {
        "type": "unt",
        "score": 108
      }
    },
    "materials": {
      "documents": ["cv.pdf"],
      "attachments": ["portfolio.pdf"],
      "portfolio_links": ["https://example.com/portfolio"],
      "video_presentation_link": "https://example.com/video"
    },
    "narratives": {
      "motivation_letter_text": "I noticed younger students in my city were falling behind, so I started sharing notes and running small peer sessions after class.",
      "motivation_questions": [
        {
          "question": "Why this program?",
          "answer": "I want to learn how to build useful projects and bring that experience back to my city."
        }
      ],
      "interview_text": "I can explain what failed in the first version, how I adapted, and what improved afterward.",
      "video_interview_transcript_text": null,
      "video_presentation_transcript_text": null
    },
    "process_signals": {
      "completion_rate": 1.0,
      "returned_to_edit": false,
      "skipped_optional_questions": 0,
      "meaningful_answers_count": 6,
      "scenario_depth": 0.72
    },
    "metadata": {
      "source": "application_portal",
      "submitted_at": "2026-03-31T18:45:00Z",
      "scoring_version": "v1.4.0"
    }
  },
  "consent": true
}
```

## Contract Notes

- `candidate_id`, `profile`, and `consent` are the only top-level runtime fields.
- `profile.academics` is optional.
- `profile.materials` is optional.
- `profile.narratives` is the main scoring input.
- `profile.process_signals` is optional reviewer/process context.
- `profile.metadata` is optional audit context.

Legacy payloads with top-level `structured_data`, `text_inputs`, `behavioral_signals`, and `metadata` are still accepted, but they are normalized into the canonical `profile` shape before scoring.
