# Candidate Expansion Report

## 1. How Many Candidates Were Added
- Added candidates: 28

## 2. How Many Counterfactual vs New Archetype
- counterfactual: 16
- new_archetype: 12

## 3. How Many By Modality Profile
- text_only: 8
- text_plus_interview: 7
- text_plus_video: 4
- full_multimodal: 9

## 4. How Many By Language Profile
- ru: 10
- en: 9
- mixed: 9

## 5. Which Source Candidates Were Reused Most Often
- `cand_003`: 2
- `cand_016`: 2
- `cand_023`: 2
- `cand_033`: 2
- `cand_034`: 2
- `cand_037`: 2
- `cand_040`: 2
- `cand_045`: 2

## 6. Which Archetype Gaps Were Filled
- sustained builder/founder profiles with repeat execution and process ownership
- strong evidence but weak motivation cases that separate competence from narrative fit
- non-STEM / arts-first / vocational execution profiles with grounded outputs
- polished but low-evidence negative controls
- support-needing but high-potential applicants whose promise is not driven by polish
- stronger multimodal profiles with transcripts, documents, and portfolio-like proof

## 7. Major Realism Risks Still Remaining
- Some counterfactual families still share recognizable motifs, so pairwise leakage risk remains if train and eval splits are not source-aware.
- Multimodal proof is still synthetic and cleaner than many real applicant attachments or transcripts would be.
- Community-action and problem-solving narratives remain more common than truly ordinary low-signal applicants.

## 8. Recommendation For Next Annotation Pass
- Annotate counterfactual families with explicit source-aware split rules so near-neighbors do not leak across train and evaluation sets.
- Prioritize adjudication on the new multimodal and strong-evidence/weak-motivation cases because they expand boundary conditions in the rubric.
- Add one more small batch of mundane, medium-signal profiles to reduce overexposure to unusually proactive narratives.
