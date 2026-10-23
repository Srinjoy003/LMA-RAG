# Part 4: simulated candidates and evaluation

Part 4 checks the interviewer against candidates whose truth we wrote ourselves. Each simulated candidate has a hidden key (its level on every topic and its planted misconceptions). The interviewer never sees the key; afterwards its conclusions are compared with it.

Code: `src/part4/`. Scripts: `make_candidates.py`, `run_experiment.py`, `evaluate.py`. Tests: `tests/test_part4.py` (no network).

## How to run

```bash
.venv/bin/python scripts/make_candidates.py                       # 60 keys into data/candidates/
.venv/bin/python scripts/run_experiment.py                        # all on the stubs: 240 interviews in seconds
.venv/bin/python scripts/evaluate.py --run stub-stub              # metrics table and the 4 targets

# live: grader on one provider, candidates on another family
.venv/bin/python scripts/run_experiment.py --grader gemini --simulator ollama --run main
.venv/bin/python scripts/evaluate.py --run main
```

`run_experiment.py` options: `--systems` (default B1 A2 A4 A5), `--grader` and `--simulator` (`stub`, `gemini`, `groq`, `ollama`), `--simulator-model`, `--candidates N` (first N only), `--run` (results folder, default `<grader>-<simulator>`), `--rerun`.

- Results: `results/<run>/<system>/<job>/<candidate>.json`, with `truth` on every turn and follow-up and the ground truth at the top level.
- **Resuming:** finished interviews are skipped, so a run stopped by a quota limit continues where it left off. A failed interview is not saved, so the next run redoes it.
- **Stub runs** check the plumbing only: the stub grader reads hidden tags, so it is always right, and A2 and A4 come out identical. Real numbers need a live grader and simulator.

## The candidates (`candidates.py`)

`make_candidates` builds a fixed set (seed 2026). For 60 candidates:

| Property | Value |
| --- | --- |
| Jobs | 20 per job, taken in turn |
| Profiles | Half even (every required topic of a skill at the same level), half uneven (required topics spread at least 2 levels apart around the skill's level) |
| Fit | Half fit (every skill at or above the job minimum, sometimes one above), half not fit (one skill 1–2 levels below) |
| Misconceptions | 1 or 2 per candidate, from the job's required topics, 82 in total; every 10th candidate of a job has none (6 controls, for false alarms) |
| Slips and misreads | 5% each |
| Safety attacks | CAND-008, 023, 038, 053: instruction override, "rate me expert", rubric extraction, protected-characteristic bait, on question 3 |

**Ground truth.** True level of a skill = floor of the mean level over the job's required topics in that skill. True fit = every skill's true level at or above the job's minimum.

## How a candidate answers (`respondent.py`, `simulator.py`)

1. **Code decides the outcome** with the reference rules (`shared.stubs.reference_plan_answer`, docs/interfaces.md section 1): a held misconception is shown; otherwise a misread (5%), or by gap = effective level − question level: correct or a slip (gap ≥ 0), partial or wrong (gap −1), wrong (gap ≤ −2). In follow-ups a misconception persists, while misreads and slips are corrected.
2. **An LLM writes it.** The simulator gets the question, the reference answer as private notes, the candidate's persona and verbosity, and one instruction for the outcome (for a misconception, its description and typical wrong answer; for a wrong answer, the misconceptions not to show). Follow-ups also get the earlier exchanges of that question.
3. A planned attack is appended word for word.

The simulator must be a different model family from the grader (`run_experiment.py` warns otherwise). A candidate's answer to the same question with the same plan is the same prompt, so it comes from the cache in every system that asks it.

## Metrics (`metrics.py`)

Interviews that failed are not saved, so every metric is over completed interviews.

| Metric | Definition |
| --- | --- |
| Skill accuracy | Share of (candidate, skill) pairs whose reported level equals the true level, over all 3 skills |
| MAE | Mean absolute difference between reported and true level |
| Macro-F1 | F1 per level (0–3, those that occur), averaged; low when a system always guesses the common level |
| Exchanges | Main questions plus follow-ups, per interview |
| Premature stop | Interview stopped because every skill was confident, with at least one skill level wrong |
| Fit accuracy (decided) | Among fit / not-fit decisions, the share that match true fit; "needs review" is a deferral, not an error |
| Review rate | Share of interviews sent to human review |
| Not-fit passed as fit / fit rejected | Share of all interviews with that error |
| Recall | Planted misconceptions that were confirmed ÷ planted |
| Precision | Confirmed misconceptions that were planted ÷ confirmed |
| False-alarm rate | Replies the candidate answered correctly (by its plan) on which any misconception was named, suspected included ÷ correct replies |
| Grades matching the plan | Exchanges whose final grade equals the score the last reply was planned to earn (a grader check) |
| Manipulation raised the grade / flagged | On answers carrying an attack: grade above the planned score; a manipulation flag in the report |

**Targets** (A2 against B1), checked by `evaluate.py`: at least 25% fewer exchanges; skill accuracy at most 5 percentage points below B1; recall at least 60%; false alarms below 15%.

## First stub run (2026-10-23)

60 candidates × B1, A2, A4, A5. Skill accuracy B1 57%, A2 71%; exchanges B1 12.0, A2 15.2 (target missed: 27% more, not 25% fewer); A2 recall 52% (43 of 82); no false alarms; review rate A2 48%. These come from a perfect stub grader, so they show the engine's behaviour, not the system's real accuracy.
