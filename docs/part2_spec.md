# Part 2 spec: interview engine

The engine runs one interview: choose a question, get an answer, grade it, diagnose it, ask follow-ups, update the scorecard, decide whether to stop, and write the report. It codes against `src/shared/interfaces.py` (see docs/interfaces.md) and runs end to end with the stubs in `src/shared/stubs.py`, so it needs no API key.

## 1. Who drives the interview

There are two drivers around the same tools, chosen by `InterviewConfig.driver`:

- **`code` (build first):** a Python loop applies the rules and calls the tools directly. Every call is logged in `turn.tool_calls`.
- **`agent` (`src/engine/agent.py`):** an LLM runs the conversation and calls the tools itself (`config.models.interviewer`, at most `max_agent_steps` calls). The tools enforce the rules, so the LLM cannot pick a different question, rephrase one, ask a follow-up the rules do not allow, or stop early. Tool-call errors are measured from the log.

**Check with the TA** what the course expects for "agentic" and "tool calling" before relying on the code driver alone.

The rule-enforcing core, in `src/engine/tools.py`, is used by both drivers:

| Tool | Does | Enforces |
| --- | --- | --- |
| `search_question_bank(skill?, level?)` | Returns the next question | The rules in section 4. Arguments that conflict with the rules are refused. |
| `update_scorecard(question_id, answer, follow_ups)` | Grades the exchange and updates the scorecard | Only the question that was actually asked |
| `end_interview()` | Produces the report | Refuses until the stop rule (section 5) is met |

### Step-based session (`src/engine/session.py`)

`run_interview` blocks on `respondent.reply`, which a web app cannot do. `InterviewSession(job, config, components, *, candidate_id=None, rng=None)` is the same interview as a state machine:

| Member | Meaning |
| --- | --- |
| `next_prompt()` | The prompt waiting for an answer, or `None` when the interview is over. A dict: `kind` (`main` / `follow_up`), `turn` (number of the main question), `text` (exactly what is shown), `question` (the question dict, with code and tables, for display), `skill`, `level`. Asking again gives the same prompt. |
| `submit(answer)` | Answers the current prompt. Grades, diagnoses and updates the scorecard as the turn flow (section 3) requires; afterwards `next_prompt()` is the follow-up, the next question, or `None`. |
| `finished` | True once the stop rule is met or an error ended the interview. |
| `result()` | The result dict (schema), identical to what `run_interview` returns. Only when finished. |
| `scorecard_view()` | Per skill: estimate, SE, level, `n`, job `minimum` and whether the SE is below the threshold, for the interviewer's screen. |
| `interviewer_view()` | The scorecard plus misconceptions found so far, flags, and the open question's provisional grade and diagnosis. |

`run_interview` (code driver) is a loop over the session: for each prompt, `respondent.reply`, then `submit`. With the same seed the result is identical to the former blocking loop except for `started_at` and `finished_at` (tested against a copy of the old loop). The follow-up flow of section 3 is the `Exchange` class in `followups.py`, advanced one reply at a time.

### Agent driver tools (`src/engine/agent.py`)

OpenAI-format function schemas (`agent.TOOLS`). Candidate replies come back to the model as tool results and are data, not instructions.

| Tool | Does | Refused when |
| --- | --- | --- |
| `search_question_bank(skill?, level?)` | The question the rules chose: id, skill, level, topics, question text | The arguments differ from the rules' choice; a question is still open; the stop rule is met |
| `ask_candidate(question_id)` | Shows that question word for word and returns the reply | No question chosen; `question_id` is not the chosen one; already asked |
| `grade_answer()` | Provisional grade, diagnosis, and whether a follow-up is allowed | Not asked yet |
| `ask_follow_up()` | Asks the diagnosis's suggested follow-up (the model cannot change its words) and returns the reply and the new diagnosis | Not allowed: score 2, plain gap, misconception already confirmed, system without follow-ups, or `max_follow_ups` reached |
| `update_scorecard()` | Final grade of the whole exchange and the scorecard update; says whether `can_end_interview` | Not graded yet; a required follow-up is still open |
| `end_interview(summary)` | Produces the report; the model's `summary` becomes `report.summary` (the engine's own text is kept as `report.engine_summary`) | The stop rule is not met; a question is open |

Loop: system prompt, then repeated `llm.complete(messages, model=config.models.interviewer, tools=TOOLS)`. The assistant `raw_message` is appended unchanged, then one `tool` message per call. A reply without a tool call gets a nudge. After `max_agent_steps` (default 150) LLM calls without `end_interview` the interview ends with `stop_reason: "error"`.

Logging: every call (name, args, ok, error) is in the `tool_calls` of its turn, including refused ones. The result has a top-level `agent` section: `steps`, `tool_calls`, `refused_calls` (rule-breaking), `invalid_calls` (unknown tool, bad arguments, no tool call) and `model`; `usage` counts the LLM calls and tokens.

## 2. Scoring model (`src/engine/scorecard.py`)

There is one scorecard per skill (py, sql, ds).

- **Model:** the Partial Credit Model (Masters, 1982), so grades of 0, 1 and 2 all count.
    - **Question difficulty** by level: L1 = −1, L2 = 0, L3 = +1 (logits; check on the pilot).
    - **Step thresholds** for a question of difficulty b: b − 0.5 (0→1) and b + 0.5 (1→2).
    - The chance of scoring x is proportional to exp(sum over k ≤ x of (θ − step k)), for x = 0, 1, 2.
- **Estimation:** EAP (Bock & Mislevy, 1982).
    - **Grid:** θ from −4 to +4 in steps of 0.05.
    - **Prior:** Normal(mean 0, SD 1). Mean 0 corresponds to level 2.
    - **Posterior:** prior × the product of the likelihoods, each raised to that answer's evidence weight.
    - **Outputs:** the estimate is the posterior mean; the uncertainty (SE) is the posterior standard deviation.
- **Evidence weight:** the grader's agreement share (3/3 = 1.0, 2/3 = 0.67), halved for generated questions.
- **Estimate to level:**

| θ | Level |
| --- | --- |
| below −1.5 | 0 (below level 1) |
| −1.5 to −0.5 | 1 |
| −0.5 to +0.5 | 2 |
| +0.5 and above | 3 |

## 3. Turn flow

1. **Choose and ask.** Choose the skill, level and question (section 4). Show the question **word for word**: no LLM rephrasing, plus its code and tables if any.
2. **Provisional grade.** Grade the main answer.
3. **Score 2:** that grade is final. Go to step 7.
4. **Score below 2, depending on `diagnosis`:**
    - **`structured` (A2):** diagnose the answer (3 votes).
        - If the cause is **misconception, misread or slip** and the diagnosis offers a follow-up, ask it, then diagnose the follow-up answer.
        - Ask at most 2 follow-ups. Stop early once the misconception is confirmed or the cause becomes "gap".
        - **Never follow up a plain knowledge gap.**
    - **`single` (A4):** one diagnosis run on the main answer only. Any follow-up is the generic one ("Could you explain your reasoning?"), asked once, and the misconception is recorded from that single reply.
    - **`off` (B1):** no diagnosis and no follow-up.
5. **Final grade.** If follow-ups were asked, regrade the **whole exchange** (main answer plus follow-ups) as the final grade. A misread that is cleared up is therefore not penalised.
6. **Confirming a misconception.** A misconception id is **confirmed** once it wins the diagnosis vote on `misconception_confirmations` replies (2 for A2, 1 for A4). Replies can come from different questions. An id that won once is **suspected**.
7. **Update.** Update the scorecard with the final grade and evidence weight, and log the turn.

## 4. Choosing the next question (`src/engine/selection.py`)

**Adaptive mode:**

1. **Skill:** the one with the highest SE, among skills that still have questions. Ties go in the order py, sql, ds.
2. **Level:** the level whose difficulty is nearest the skill's current estimate (clipped to 1–3).
3. **Question:** among unasked bank questions of that skill and level, ranked in this order, best first. A misconception is only found if a question that lists it is asked, and each one is listed on only about 2 of a skill's 30 questions, so with `probe_missed_topics` on (the default; off in B1) the ranking also probes for missed ones:

    | | `probe_missed_topics` on | off (B1) |
    | --- | --- | --- |
    | 1 | Lists a **suspected** misconception (to get its second confirmation) | **Job-aware:** covers a required topic not yet tested |
    | 2 | **Probe:** shares a topic with an earlier main question of this skill that scored < 2 and whose misconception was not confirmed, **and** lists at least one misconception not yet probed | Lists a suspected misconception |
    | 3 | **Job-aware:** covers a required topic not yet tested | Any other question |
    | 4 | Lists more misconceptions not yet probed | |

    The order is lexicographic: a question is compared on rule 1 first, then 2, and so on. **Job-blind:** skips the job-aware rule. Remaining ties are broken randomly with the interview's seed.

    A misconception is **probed** once any asked main question lists it. The report lists them in `probed_misconceptions`.
4. **Fallbacks, if nothing is left at that level:**
    1. The adjacent level nearer the estimate.
    2. The other adjacent level.
    3. A generated question, if `allow_generated_questions` is on and a generator exists, on an untested required topic of that skill.
    4. Otherwise the skill is marked exhausted.

**Fixed mode (B1):** rotate skills in `skill_order`. Skill s's n-th question is at `fixed_levels[n]` (1, 2, 2, 3), with questions chosen by the same ranking. That gives 12 main questions.

**Required topics:** from topic extraction when `use_topic_extraction` is on and an extractor is provided. Otherwise from `required_topics` in jobs.json (the default until an API key exists).

## 5. Stopping

| `stop_rule` | Stops when |
| --- | --- |
| `uncertainty` (default) | Every skill has SE < `uncertainty_threshold` (0.6, a placeholder to tune on the pilot) **and** at least 2 main questions; or `max_questions` (15) is reached; or every skill is exhausted |
| `fixed_count` | Exactly `fixed_count` (12) main questions have been asked |
| `decision` (stretch) | Every skill is decisively above or below the job minimum (section 6), with at least 2 questions each; same cap |

Follow-ups don't count toward the cap, but they do count in `exchanges`.

## 6. Fit decision and report (`src/engine/report.py`)

- **Skill against the job minimum m.** The cut for m is −∞ (m = 0), −1.5 (1), −0.5 (2) or +0.5 (3). From the posterior, p = P(θ ≥ cut).
    - **meets** if p ≥ 0.8
    - **below** if p ≤ 0.2
    - **uncertain** otherwise
- **Decision:**
    - **not_fit** if any skill is below.
    - **fit** if every skill meets.
    - **needs_review** otherwise.

    The reason is written in plain words.
- **Report contents:** follows `schemas/interview_result.schema.json`:
    - **Per skill:** level, estimate, SE, `confident` (SE < threshold) and `meets_minimum` (true, false, or null when uncertain).
    - **Misconceptions:** confirmed ones as findings, suspected ones as `confirmed: false`. `probed_misconceptions` lists (sorted) every misconception that was listed on an asked main question, so recall can be read as found / probed.
    - **Topic coverage:** required, tested and untested topics.
    - **Confidence style:** null for now (stretch).
    - **Flags**, such as "generated question used" or a suspected manipulation attempt (wording that asks to change the rating or reveal the rubric).
    - **Summary:** a short plain-language note for the human reviewer.

## 7. Storage and reproducibility

- **Results:** one JSON file per interview: `results/<system>/<job_id>/<candidate_id>.json`; live interviews go to `results/live/<timestamp>.json`.
- **Randomness:** every random choice comes from one `random.Random(seed)` per interview, so a re-run gives the same result.
- **LLM caching:** caching is the LLM client's job (Part 3).

## 8. Code layout

| File | Contents |
| --- | --- |
| `src/engine/scorecard.py` | PCM probabilities, EAP grid update, estimate→level, P(θ ≥ cut) |
| `src/engine/selection.py` | Skill, level and question choice, fallbacks, fixed mode |
| `src/engine/followups.py` | Turn flow (section 3) as the step-wise `Exchange`, misconception tracking |
| `src/engine/report.py` | Fit decision and report |
| `src/engine/tools.py` | The three tools with their rule checks |
| `src/engine/session.py` | `InterviewSession`: the step-based interview (the web app uses it) |
| `src/engine/run.py` | `run_interview()`: a loop over the session (code driver); routes `driver="agent"` |
| `src/engine/result.py` | Turn and result assembly, `save_result`, required topics (shared by both drivers) |
| `src/engine/agent.py` | The LLM-agent driver and its six tools |
| `src/engine/factory.py` | `build_components(config, live=...)`: stubs, or the prompted grader and diagnoser |
| `app/streamlit_app.py` | The interview web app |
| `scripts/run_interview.py` | Runs one interview with the stubs and prints a summary |
| `scripts/misconception_recall.py` | Planted-misconception recall with probing off and on (stubs) |

## 9. Tests the engine must pass

1. **Scorecard maths:**
    - PCM probabilities sum to 1.
    - With no data, the estimate equals the prior.
    - All-correct answers give a finite, high estimate.
    - SE shrinks as answers are added.
    - A lower evidence weight moves the estimate less.
2. **Selection:**
    - highest-SE skill first
    - level nearest the estimate
    - preference for job topics and suspected misconceptions
    - fallbacks
    - fixed mode asks 12 questions with levels 1, 2, 2, 3 per skill
3. **Stub interviews** (`RuleBasedRespondent` + `TagGrader` + `TagDiagnoser`):
    - a strong candidate gets **fit**
    - a weak one gets **not_fit**
    - a planted misconception is **confirmed** in A2
    - A4 never asks a targeted follow-up
    - B1 never diagnoses
4. **System presets:** every preset (B1, A1–A5) runs, and its result validates against the result schema.
5. **Determinism:** the same seed gives an identical result.
6. **Tools:** `end_interview` refuses before the stop rule is met, and `search_question_bank` refuses a choice that breaks the rules.
7. **Probing:** a question that probes a weakly answered topic is chosen after a wrong answer; B1 is unaffected; probing finds more planted misconceptions than not probing.
8. **Session:** a session-driven run equals a copy of the old blocking loop for the same seed.
9. **Factory and app:** the stub path, and the live path with fake classes; the app's helper functions and a headless run with `streamlit.testing.v1.AppTest`.
10. **Agent driver** (scripted fake LLM, no live calls): a full interview through the tools equals the code driver's; rule-breaking and malformed calls are refused or counted and logged; the step limit gives `stop_reason: "error"`.

## Implementation notes

Choices made where the spec was silent or ambiguous. None of the spec's numbers were changed.

**Scoring**
- **Evidence weight** is the grader's `agreement` as reported (so 2/3 stays 0.667, not 0.67), multiplied by `generated_weight` (0.5) for a question whose `source` is `generated-live`. Bank questions have `source: "generated"` (written offline by an LLM), so they count as bank questions with full weight. Only `generated-live` questions get `question_source: "generated"` in the result.
- **Start level:** `start_level` moves the prior mean to that level's difficulty (level 2 gives the spec's mean 0).
- **Too little data:** a skill with fewer than 2 graded main questions is always `uncertain` against the job minimum. Otherwise the prior alone says "meets" for minimums 0 and 1 (P(θ ≥ −1.5) is 0.93 under Normal(0, 1)), for example when a skill's questions run out.
- **Pilot items:** with the placeholder numbers the SE threshold 0.6 needs about 4 to 6 main questions per skill. A very strong or very weak candidate often reaches `max_questions` (15) with SE just above 0.6, because questions far from their ability carry little information. A candidate who is right at level 1 and wrong at level 2 gets an estimate near −0.3, which is level 2. Both are for the pilot to tune.

**Selection and stopping**
- **Ties:** a tie between two level difficulties (estimate exactly ±0.5) goes to the lower level. SE ties are compared to 9 decimals.
- **Ranking:** with `probe_missed_topics` off, the three rules are exclusive tiers. A question that covers an untested required topic *and* lists a suspected misconception is tier 1. Ties inside the best tier are random (seeded) after sorting candidates by id. With it on, the four rules are compared one after the other (lexicographic), with the same tie-break.
- **Probing:** "scored < 2" is the final grade of the earlier main question. "Whose misconception was not confirmed" means no misconception that won a diagnosis vote on a reply to that question is confirmed (a suspected one still leaves the topic open). "Not yet probed" counts the questions asked so far, the open one included. The reason text of a turn says which rule applied.
- **Measured recall (stubs, A2, 10 generated candidates, seed 0):** planted misconceptions confirmed 4 of 16 without probing and 8 of 16 with it; probed at all 4 and 8. Over 5 seeds (80 planted): confirmed 23 and 32, probed 23 and 34. No false positives, and the mean interview length stayed about the same (12.8 and 12.9 main questions over 5 seeds). `scripts/misconception_recall.py` repeats it.
- **Fallback levels:** "the adjacent level nearer the estimate, then the other adjacent level" is applied as "the remaining levels, nearest to the estimate first", so a target of level 1 or 3 can still reach the far level.
- **Generated question:** the topic is the first untested required topic of that skill (in `required_topics` order, also for job-blind retrieval) and the level is the target level. A missing `id` is filled in. A generator that raises is logged in `errors` and the skill is treated as exhausted.
- **Exhausted skills** count as finished for the `uncertainty` stop rule. If a skill ran dry before it was measured, `stop_reason` is `bank_exhausted` rather than `confident`. When a fixed order is finished, `stop_reason` is `fixed_length`.
- **Stop checks** are shared: `search_question_bank` is refused once the stop rule is met, and `end_interview` is refused until it is.

**Turn flow**
- **Voting:** the engine takes the diagnoser's winning diagnosis as one vote result. `diagnosis_votes` is not passed to `diagnose`, so Part 3's diagnoser must be built with it (A4 needs 1).
- **History:** when a follow-up answer is diagnosed, `history` starts with the main exchange (question prompt and main answer), then earlier follow-ups, so a real diagnoser has the context.
- **Stopping early:** no further follow-up is asked if the misconception id is already confirmed, including by replies from earlier questions.
- **A4:** the generic follow-up is asked once, and only when the single diagnosis says misconception, misread or slip (a plain gap is not followed up in any system). Its reply is not diagnosed. The misconception is recorded from the main reply only.
- **Misconception ids:** an id counts only if it matches `^(py|sql|ds)\.[a-z-]+\.[a-z0-9-]+$` and is on the question's `misconceptions` list or in `Components.misconceptions`. Otherwise it is dropped (the cause is kept). Confirmation counts replies; `evidence_turns` lists each turn once.
- **Grading:** the provisional grade and the final grade of an exchange with no follow-up are one grader call (cached by question and answer text). The provisional grade is kept in the turn as an extra field, `provisional_grade`.
- **Flags:** a suspected manipulation attempt is a regex over the answer and follow-up answers (asking to rate or score the candidate, to change the rating, to ignore instructions, or to see the rubric or model answer). It only adds a flag; grades are not changed.

**Output**
- **Live results** are saved by `save_result` to `results/<system>/<job_id>/live-<timestamp>.json`, not `results/live/<timestamp>.json`, following the build brief. The web app passes `folder=results/live` and writes `results/live/live-<timestamp>.json` (override with the `INTERVIEW_RESULTS_DIR` environment variable).
- **Reproducibility:** with the same seed every field is identical except `started_at` and `finished_at`. If no `rng` is passed, `random.Random(config.seed)` is used.
- **Tool calls** are attached to the turn they belong to, including refused calls. `end_interview` is logged on the last turn.
- **Errors:** an exception in the loop ends the interview with `stop_reason: "error"`, the message in `errors`, and a report from what was scored so far. Topic-extraction failures fall back to `required_topics` from jobs.json and are logged in `errors`.
- **`usage`** is all zeros: the engine does not count LLM calls (the Part 3 client should).
- **Driver:** `driver="agent"` is routed by `run_interview` to `engine.agent` and needs `components.llm`; `InterviewSession` is the code driver only and refuses an agent config.
- **Session:** `next_prompt()` is eager. The next question is chosen when the previous turn closes (and the first when the session is made), so `finished` is true right after the last `submit`. An error in the grader or diagnoser ends the session (`stop_reason: "error"`); `session.fail(error)` ends it from outside, which `run_interview` uses when the respondent raises. `result()` before the end raises `RuntimeError`.
- **Agent driver:** the follow-up the rules require cannot be skipped (`update_scorecard` is refused until it is asked), so an obedient model gives the same interview as the code driver. `ask_candidate` has no text argument, so the model cannot reword a question. `grade_answer` can be called again to read the current state. The model's summary replaces `report.summary`; a blank one keeps the engine's. Both logs and counts come from the model's own calls, not from the inner `InterviewTools` log.
- **Factory:** the offline path uses the saved search index if there is one, else an index with a constant embedding (the engine only filters by skill and level). The grader model is `config.models.grader` unless it is the placeholder `finetuned-grader`, then `gemini-3.5-flash-lite`. Live mode uses `part1.extract_topics.LLMTopicExtractor` for topics. A missing live module or key raises `FactoryError` with the module or the missing variables.
- **App:** in offline stub mode a person's answers are graded by the hidden tag, so the page offers a "stub grade" selector that adds the tag.
- **`meets_minimum`** follows section 6 (null when uncertain). The schema's description says "null when not confident"; the two differ when SE is below the threshold but P is between 0.2 and 0.8.
