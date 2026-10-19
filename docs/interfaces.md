# Interfaces between the parts

The four parts are built separately and plugged together. This page fixes how they connect. Code: `src/shared/interfaces.py` (types, interfaces, config) and `src/shared/stubs.py` (fakes). Formats: `schemas/candidate.schema.json` and `schemas/interview_result.schema.json`. Examples: `examples/`. Tests: `tests/test_shared.py`.

**Rule for unused fields:** everything outside a schema's `required` list is optional and has a default. Dropping a feature later, such as confidence style or resume checks, just means ignoring its field. Nothing breaks.

## Who provides what

| Piece | Provided by | Used by |
| --- | --- | --- |
| `QuestionSource.search(...)` | Part 1 (`part1.index.QuestionIndex`) | Part 2 |
| `TopicExtractor(job_description)` | Part 1 (`part1.extract_topics`) | Part 2 |
| `run_interview(...)`, scorecard, fit rule, report | Part 2 | Part 4, app |
| `Grader.grade(...)`, `Diagnoser.diagnose(...)`, `LLMClient` | Part 3 | Part 2 |
| Candidate keys, `plan_answer(...)`, `CandidateSimulator`, simulated `Respondent` | Part 4 | Part 4's experiment runner |
| `InterviewConfig`, `SYSTEMS` presets | shared | everyone |

## 1. Candidate key (`schemas/candidate.schema.json`)

A candidate key is the hidden truth about one simulated candidate. The interviewer never sees it.

| Field | Required | Meaning | Default |
| --- | --- | --- | --- |
| `id` | yes | `CAND-001`… | |
| `job_id` | yes | Job they apply for (`data/jobs.json`) | |
| `profile_type` | yes | `even` or `uneven` across a skill's topics | |
| `topic_levels` | yes | Level 0–3 for **all 27 topics** (0 = doesn't know it) | |
| `misconceptions` | yes | Planted misconception ids: 1–2 recommended, 0 for control candidates, at most 3 | |
| `confidence_style` | no | calibrated / overconfident / underconfident | calibrated |
| `misread_rate` | no | Chance of misreading a main question | 0 |
| `slip_rate` | no | Chance of a careless slip on a question within their level | 0.05 |
| `verbosity` | no | short / medium / long answers | medium |
| `persona` | no | Name and a one-line background, for tone | |
| `resume` | no | Text, claimed levels and an `exaggerated` flag (stretch) | |
| `safety_attack` | no | Attack type, text, and the main-question numbers where it is used | null |
| `seed`, `notes` | no | Reproducibility; free text | numeric part of id |

**Ground truth derived from a key:**

- **True level of a skill for a job:** the floor of the mean `topic_levels` over that job's required topics in that skill. For example, a data analyst's SQL truth averages their 6 required SQL topics. This is what skill accuracy is scored against.
- **True fit:** every skill's true level is at or above the job's minimum.
- **Planted misconceptions:** the `misconceptions` list. Misconception recall and precision are scored against it.

**How a simulated candidate answers.** Code decides *what* happens (`plan_answer`). The LLM only words it (`CandidateSimulator`). The reference rules are in `stubs.reference_plan_answer`, and Part 4 may tune them.

1. **Effective level:** the candidate's lowest `topic_levels` among the question's topics.
2. **Main question:**
    - If the candidate holds one of the question's misconceptions, they answer with it (expected score 0).
    - Otherwise they misread with probability `misread_rate` (0).
    - Otherwise the outcome depends on gap = effective level − question level:
        - gap ≥ 0: correct (2), or a slip (1) with probability `slip_rate`
        - gap = −1: partial (1) or wrong (0), 50/50
        - gap ≤ −2: wrong (0)
3. **Follow-up:**
    - A misconception persists.
    - A misread or slip is corrected once clarified.
    - Wrong and partial stay as they are.
4. **Safety attack:** on the listed turns, the attack text is added to the answer.

## 2. Function signatures (`src/shared/interfaces.py`)

| Interface | Call | Returns |
| --- | --- | --- |
| `QuestionSource` | `search(query=None, skill=None, levels=None, prefer_topics=None, exclude_ids=frozenset(), k=5)` | `[(question, score), …]` |
| `QuestionGenerator` | `generate(skill, level, topic)` | a question with `source="generated-live"` |
| `TopicExtractor` | `extractor(job_description)` | `[topic ids]` |
| `Grader` | `grade(question, answer, follow_ups=())` | `Grade`: `score` (0/1/2), `votes`, `agreement`, `rationale` |
| `Diagnoser` | `diagnose(question, answer, history=())` | `Diagnosis`: `cause`, `misconception_id`, `follow_up`, `vote_share`, `votes` |
| `Respondent` | `reply(prompt, question=…, kind="main"/"follow_up", turn=n)` | answer text |
| `CandidateSimulator` | `answer(candidate, question, prompt, plan, history=())` | answer text that follows `plan` exactly |
| `LLMClient` | `complete(messages, model=…, temperature=0, json_schema=None, max_tokens=1024)` | `text`, `parsed`, `model`, token counts |
| `plan_answer` | `plan_answer(candidate, question, kind=…, turn=n, previous=…, rng=…)` | `AnswerPlan`: `outcome`, `misconception_id`, `expected_score`, `effective_level`, `attack` |
| `run_interview` | `run_interview(job, respondent, config, components, candidate_id=None, rng=None)` | an interview result (section 4) |

### Rules

- **The grader marks the whole exchange**, meaning the main answer plus its follow-ups, once per main question. Follow-ups count toward the interview's length.
- **The diagnoser only suggests.** The engine decides: a misconception is **confirmed** only when the same id wins the vote on `misconception_confirmations` (2) replies.
- **The engine sees only reply text.** A simulated respondent keeps its hidden plans in `respondent.truths`. The experiment runner merges them into the result as `truth` fields after the interview.
- **`LLMClient` is provider-agnostic.** It uses the OpenAI-compatible format (Gemini, Groq, NVIDIA NIM, OpenRouter).
    - Implementations must **cache** every response, keyed on model, messages, schema and temperature, and **retry** on rate limits.
    - Endpoints and keys come from environment variables, never code:
        - `LLM_BASE_URL` and `LLM_API_KEY` set the default.
        - Per-role overrides add the role name, for example `LLM_BASE_URL_SIMULATOR` and `LLM_API_KEY_SIMULATOR`.
- **`Components`** bundles the question source, grader, diagnoser, topic extractor, question generator, LLM client and the misconception records. Any of them can be replaced by a stub.

## 3. Config flags (`InterviewConfig`) and the systems compared

| Flag | Options (default first) | Meaning |
| --- | --- | --- |
| `mode` | adaptive / fixed | Adaptive: least-certain skill, matched difficulty. Fixed: `fixed_levels` (1, 2, 2, 3) per skill, rotating skills |
| `retrieval` | job-aware / job-blind | Prefer the job's topics, or ignore them |
| `stop_rule` | uncertainty / fixed_count / decision | Stop when every skill's standard error < `uncertainty_threshold` (0.6, tuned on the pilot) or at `max_questions` (15); or always ask `fixed_count` (12); or (stretch) stop once each skill is clearly above or below the job minimum |
| `driver` | code / agent | Code loop, or an LLM calling the rule-enforcing tools (docs/part2_spec.md section 1) |
| `diagnosis` | structured / single / off | Structured: voted diagnosis and targeted follow-ups. Single: one diagnosis run on the main reply, generic follow-up only. Off: neither |
| `max_follow_ups` | 2 | Per main question |
| `diagnosis_votes`, `grader_votes` | 3, 3 | Runs per diagnosis or grade |
| `misconception_confirmations` | 2 | Replies needed to confirm a misconception |
| `probe_missed_topics` | true (false in B1) | Prefer questions that probe topics answered weakly and misconceptions not yet probed (part2_spec section 4) |
| `max_agent_steps` | 150 | Agent driver: LLM calls before the interview ends with `stop_reason: "error"` |
| `allow_generated_questions`, `generated_weight` | true, 0.5 | Fallback questions and their evidence weight |
| `use_topic_extraction` | true | False uses `required_topics` from jobs.json directly |
| `start_level` | 2 | Prior for every skill |
| `models` | see `ModelSettings` | Model per role. The names are placeholders until the team picks providers. Keep `topic_extractor` fixed across the base-model comparison. |
| `seed` | 0 | Reproducibility |

| System | Preset in `SYSTEMS` | Tests |
| --- | --- | --- |
| B1 | fixed mode, 12 questions, no diagnosis | Baseline |
| A1 | adaptive, `stop_rule=fixed_count` | Early stopping |
| A2 | defaults | Full system |
| A3 | A2 with a prompted grader | Fine-tuning |
| A4 | A2 with `diagnosis=single` (1 vote, confirmed from 1 reply) | Theory-of-mind component |
| A5 | A2 with `retrieval=job-blind` | Job-aware retrieval |

The base-model comparison is A2 with `models.interviewer` and `models.diagnoser` changed.

## 4. Interview result (`schemas/interview_result.schema.json`)

One JSON file per interview, holding:

- **Settings:** `interview_id`, `system`, the full `config`, `job_id`, `candidate_id` (null for a human), timestamps.
- **`required_topics`:** the topics the system worked from.
- **`turns`:** one per main question, each with:
    - skill, level, question id and source, topics, and why it was chosen
    - what was asked and the answer
    - the diagnosis and up to 2 follow-ups (each with its own diagnosis)
    - the grade
    - the evidence weight
    - the scorecard afterwards (estimate, standard error, level and count per skill)
    - the tool calls made, and `truth` (experiment runner only)
- **`exchanges`:** main questions plus follow-ups. This is the RQ1 length measure.
- **`stop_reason`:** confident / max_questions / fixed_length / bank_exhausted / error.
- **`report`:**
    - per skill: level, estimate, standard error, whether it is confident, the job minimum, and whether it meets it
    - `decision` (fit / not_fit / needs_review) and the reason
    - misconceptions, each with whether it is confirmed and the evidence turns
    - topic coverage (required, tested, untested)
    - confidence style, flags and a short summary
- **`usage`:** LLM calls and tokens. **`errors`:** any problems.

Extra properties are allowed in result files, so new logging never invalidates old results.

## Stubs (`src/shared/stubs.py`)

These run the whole loop with no LLM, deterministically.

- **`RuleBasedRespondent(candidate)`:** follows the reference answer rules and writes a hidden tag into each reply.
- **`TagGrader(noise=…)` and `TagDiagnoser()`:** read that tag back. `noise` simulates an imperfect grader.
- **`RandomGrader` and `ScriptedRespondent`:** for plumbing tests.

Use these for engine tests and dry runs only.
