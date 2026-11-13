# Evaluation protocol for the reviewed bank (Step 2)

Nilkanta Karak, 13 November 2026. Builds on [the baseline manifest](evaluation_baseline.md) (Step 1, commit `c297b3f`). The machine-readable copy of the matrix, run names, allocation and budget is [experiment_matrix.json](experiment_matrix.json). `tests/test_experiment_matrix.py` checks that file against the real presets and candidate files.

This document **declares** what will be compared, how it will be measured, and how the human checks are scored. It does not run any experiment or claim any result. Anything it marks as *to implement* is Step 4 (Souradeep) or Step 5 (Nilkanta) work. Changes after the pilot need a new version number (section 7).

## 1. Conditions

All conditions use the same reviewed bank (135 questions, 95 misconceptions, hashes in `evaluation_baseline.json`), the same 3-vote grader, the same models, and seed 0. The table lists only how each condition differs from the historical `A2` preset. `tests/test_experiment_matrix.py` checks this against `shared.interfaces.SYSTEMS`.

| Condition | Preset | Differs from A2 | Status | Role |
| --- | --- | --- | --- | --- |
| **B1** | `B1` | `mode=fixed`, `stop_rule=fixed_count` (12), `diagnosis=off`, `generic_follow_up=False`, `probe_missed_topics=False` | exists | RQ1 baseline |
| **A2-rev** | `A2Q` for now | `misconception_confirmation_unit=question` | exists; Step 4 may extend it | Revised full system, main method for RQ1 and RQ2 |
| **A4** | `A4` | `diagnosis=single`, `diagnosis_votes=1`, `misconception_confirmations=1` | exists | RQ2 ablation: no candidate-belief package |
| **ADAPT12** | to add | `stop_rule=fixed_count` (12), `diagnosis=off`, `generic_follow_up=False` | to implement (Step 5); identical to the `adaptive12` control in `report/make_figures.py` | RQ1 equal-exchange control |
| **A5** | `A5` | `retrieval=job-blind` | exists (applied to A2-rev at freeze, see below) | Job-aware retrieval |
| A2 | `A2` | none | exists | Optional: confirmation-rule and bank comparison |
| A2M | to add | see section 1.2 | to design (Step 5, with Souradeep) | Optional: memory-specific control |
| A1 | `A1` | `stop_rule=fixed_count` (12) | exists | Not run live: 12 main questions plus follow-ups is not equal-exchange with B1 |
| A3 | `A3` | grader model | exists | Not run: no fine-tuned grader exists |

**A2-rev naming.** The revised full system is whatever Souradeep hands over after Step 4. Today that is the `A2Q` preset. If Step 4 changes behaviour beyond the confirmation unit, it gets its own preset name and `A2Q` stays as it is. A5 must be the frozen A2-rev with `retrieval=job-blind`, not today's `A5` (which is historical A2 with job-blind retrieval). Step 5 adds the matching preset, so that A5 against A2-rev changes exactly one setting.

**ADAPT12** asks exactly 12 main questions, chosen adaptively, with no diagnosis and no follow-ups. It therefore has exactly 12 exchanges, like B1. Apart from adaptive choice, the only difference from B1 is `probe_missed_topics` (on in ADAPT12, off in B1), which only reorders questions within the chosen skill and level. That is part of "adaptive choice" here. On the 60 frozen stub candidates it gave 12.0 exchanges and 72.2% skill accuracy, against 60.6% for B1. That is a plumbing check only, not a result.

### 1.1 What A4 actually changes

The report says A4 "keeps everything else". The code (`engine/followups.py`, `engine/selection.py`) shows five behavioural differences from A2, all of which change together:

1. **One diagnosis vote instead of three.** Its `vote_share` is always 1.0.
2. **Only the main answer is diagnosed.** A2 diagnoses every follow-up reply too.
3. **Follow-ups.** A4 asks at most one, and it is always the generic "Could you explain your reasoning?", asked when the main-answer cause is misconception, misread or slip. A2 asks up to two targeted follow-ups written by the diagnoser, and skips them once the misconception is confirmed.
4. **Confirmation from a single winning reply.** No misconception is ever in the "suspected" state.
5. **Selection never targets suspected misconceptions.** Because of item 4, the tracker's suspected set is always empty, so ranking rule (a), "prefer a question listing a suspected misconception", never fires. Rule (b), weak-topic probing, still applies.

A4 keeps the scorecard, stopping rule, cap, grader and its 3 votes, the regrade of exchanges that had a follow-up, job-aware retrieval, and the verdict rule.

**Consequence for claims.** A2-rev against A4 measures the whole **diagnosis and candidate-belief package**: votes, targeted follow-ups, diagnosis of follow-ups, cross-reply corroboration, suspected-misconception probing, and (for A2-rev) the distinct-question rule. It cannot attribute a difference to *persistent memory* alone. With `misconception_confirmations=1`, counting by reply or by question gives the same result, so A4 is identical under either unit.

### 1.2 A2M: matched-memory control (optional)

A2M is needed only for a claim that **cross-question belief memory** helps. Without it, the report limits the RQ2 claim to the combined package. A2M is defined against historical A2, so that exactly one thing changes: the tracker forgets between main questions.

- Votes, targeted follow-ups, follow-up diagnosis and the 2-reply rule are as in A2, but a misconception can only be confirmed by replies to **one** main question.
- Nothing carries to the next question: selection gets no suspected set, and a misconception confirmed earlier does not suppress later follow-ups.
- The report lists every misconception confirmed on any question, with the same evidence fields.

The comparison is **A2 against A2M**. A2-rev cannot be the reference here, because the question unit cannot confirm anything when memory is reset. Implementation touches the tracker and selection (Souradeep's modules). It is agreed after Step 4 and run only if the budget at freeze allows (section 8).

## 2. Comparisons and endpoints

Each comparison is **paired on candidates**: it uses the candidates with a saved result in *both* of its conditions. Today's `metrics.paired()` intersects every system in the run folder, so one incomplete condition would shrink all comparisons. Step 5 adds pairing per comparison. Intervals are 95% percentile bootstrap intervals over candidates: 2,000 draws, seed 0, the same draw for both conditions, as in `metrics.target_intervals`.

| ID | Question | Conditions | Primary endpoint | Decision rule | Secondary |
| --- | --- | --- | --- | --- | --- |
| **C1** | RQ1 | A2-rev vs B1 | Skill accuracy difference; exchanges per interview | Non-inferior if the lower bound of the accuracy-difference interval is above −5 points; exchange target: at least 25% fewer (already missed historically, reported as is) | MAE, macro-F1, fit accuracy when decided, review rate, false fit and false reject, premature stops |
| **C2** | RQ1 at equal length | ADAPT12 vs B1 | Skill accuracy difference at exactly 12 exchanges | Adaptive choice helps if the interval is above 0 | MAE, verdicts |
| **C3** | RQ2 | A2-rev vs A4 | Misconception recall and precision against planted, and false-alarm rate (each as a difference with interval) | Report all three plus length together; no single winner metric | Human-supported precision on the annotated sample (section 5), exchanges, interviewer LLM calls |
| C4 | Retrieval | A5 vs A2-rev | Recall; required-topic coverage | Descriptive | Skill accuracy |
| C5 | Confirmation rule | A2 vs A2-rev | Precision, recall, exchanges, **and skill accuracy** | Descriptive | Number of questions selected because they list a suspected misconception |
| C6 | Memory | A2 vs A2M | As C3 | Only with A2M run | |
| C7 | Bank change | historical `midsub` B1/A2 vs reviewed B1/A2 | Reported side by side | Descriptive only: not paired, original transcripts unavailable | |

C1 to C3 are the primary comparisons; C4 to C7 are exploratory and carry no new claims. A result whose interval crosses its threshold is "not settled", as in `docs/part4.md`.

**Metric definitions** are those in `src/part4/metrics.py` and `docs/part4.md`, unchanged. Two additions are declared now:

- **RQ2 is counted on reported findings.** For both A2-rev and A4, recall and precision count misconceptions the report marks `confirmed`. As a secondary measure, both are also computed on *any named* misconception (suspected included), so A2-rev is not judged only by its stricter threshold.
- **Premature stops** reuse Kushal's `premature_counts` and `premature_breakdown` from `report/make_figures.py`, with denominators added: wrong levels ÷ skills assessed, per skill and per true level, with intervals. Counts alone (five/four/five) do not show equal error rates.

**Stub warning for C5 (found while preparing this protocol).** On the 60 frozen candidates with stubs, `A2Q` reached 67.8% skill accuracy against 78.3% for `A2`, and under-estimated 38 levels against 20. The cause: under the question rule a misconception stays "suspected" longer, so selection picked 33 questions *because they list a suspected misconception* (A2: 0). A candidate holding it scores 0 on each of those, which pulls the ability estimate down. Ground-truth levels ignore misconceptions. This is stub behaviour, not a live result, but it is a real interaction between corroboration and the scorecard. C5 must report skill accuracy, and Souradeep should check it in Step 4.

## 3. Candidates, pairing and allocation

| Population | Ids | Seed | Size | Used for | Status |
| --- | --- | --- | --- | --- | --- |
| Evaluation | `CAND-001`–`CAND-060` | 2026 (existing files, hashed in `evaluation_baseline.json`) | 60 | Final runs only (Step 10). Never used for tuning or annotation. | Frozen; do not regenerate |
| Annotation source | `VAL-A01`–`VAL-A12` | 3101 | 12 (one balanced block) | Step 3 annotation sample | To generate (needs Step 5 support, section 7) |
| Pilot | `VAL-P01`–`VAL-P12` | 3102 | 12 (one balanced block) | Step 8 tuning and pilot | To generate |

The evaluation candidates were planted from the 94-misconception catalog. `py.concurrency.async-preempts` (added in the review) is never planted in them, so it can only appear as a non-planted finding there. The validation populations are drawn from the current 95-entry catalog and may plant it.

**Final-run allocation (Step 10).** Within every block of 12 evaluation candidates, the candidate with 0-based index `i` has job `i % 3` and `k = (i % 12) // 3`. Here k0 = even/fit, k1 = uneven/fit, k2 = even/not-fit, k3 = uneven/not-fit. **Nilkanta runs k ∈ {0, 3}; Souradeep runs k ∈ {1, 2}**, in every block, and each person runs *every* selected condition on their own candidates.

| Share | Candidates | Jobs (DA/BE/DE) | Even/uneven | Fit/not fit | Planted | Controls | Attacks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nilkanta | 30 | 10/10/10 | 15/15 | 15/15 | 39 | 3 | CAND-023, CAND-038 |
| Souradeep | 30 | 10/10/10 | 15/15 | 15/15 | 43 | 3 | CAND-008, CAND-053 |

Every block also splits 3/3 on profile and on fit. The full id lists are in `experiment_matrix.json`. `--share 1/2` would **not** give this balance: it takes every second candidate, which in block 1 gives one share 4 even and 2 uneven, and puts all four attacks in one share. Use explicit ids instead (`--ids` or an id-list file, Step 5). Same-key `--keys` sharing within one person's share is fine.

## 4. Run names and exclusions

Run folder: `results/<version>-<purpose>/`. The version is `rv1` for the reviewed bank at the Step 9 freeze. Any change to behaviour-affecting code, prompts, bank, catalog, candidates or models makes `rv2`, and the affected comparisons are repeated.

| Run | Population | Conditions |
| --- | --- | --- |
| `midsub` | historical, 24 | B1, A2. Reserved: never written to, results unavailable |
| `rv1-annot` | VAL-A | A2 (historical rule, which produces the most extra claims to label) |
| `rv1-pilot` | VAL-P | B1, A2-rev, A4, ADAPT12 (plus any stopping variant from Step 6) |
| `rv1-main` | CAND-001–060 | Frozen matrix (section 8) |
| `rv1-sqlverify` | paired subset, chosen at freeze | A2-rev with SQL verification on and off (Step 7) |

**Exclusions**, declared before any data is seen:

1. An interview ending in `stop_reason="error"` is not saved (current runner behaviour). Retry it up to 3 times under the same frozen setup. If it still fails, exclude that candidate *only from comparisons involving that condition*, and list it in `results/<run>/exclusions.json` with the error.
2. A quota stop is a resume, not an exclusion.
3. A completed interview with a non-empty `errors` list (for example a topic-extractor fallback) is **kept** and flagged.
4. No exclusion based on outcomes. Attack candidates and no-misconception controls stay in every metric; safety is also reported separately.
5. A result whose manifest does not match the run's frozen manifest is not used. Rerun it under a new run name; never mix it in.
6. Validation candidates never enter evaluation metrics, and evaluation candidates are never looked at during tuning.

## 5. Human checks: annotation protocol

Two tasks, both labelled by Nilkanta and Souradeep **independently**. Neither sees the other's labels until both sheets are submitted and frozen (file hashes recorded). Initial labels are kept unchanged after adjudication. Templates are in [annotation/](annotation/).

**Preparer and blinding.** A teammate who is not a rater (Kushal or Srinjoy) should generate the sheets and keep the answer keys. If that isn't possible, the preparer (Souradeep, Step 3) generates the sheets with a script that writes the key to a separate file that neither rater opens until both have submitted. Question ids encode the level (`SQL-L2-007`), so sheets use opaque item codes. The bank is in the repo, so blinding ultimately depends on raters not looking items up; the report states this as a limitation.

### 5.1 Task D: blind difficulty check (spec section 14)

- **Sample:** 7 questions per skill, simple random from the 45 of that skill (seed 1401), 21 in total, presented in one shuffled order as `D01`–`D21`.
- **Shown:** question text, code, tables, model answer, key points. **Hidden:** id, level, topic tags.
- **Label:** level 1, 2 or 3 by the definitions in `docs/part1_spec.md` section 5, plus an optional note.
- **Scoring:**
  - quadratic-weighted Cohen's kappa between the raters;
  - raw agreement;
  - each rater against the assigned level.
- **Spec rule:** if a skill's assigned level disagrees with *both* raters on more than 2 of its 7 questions, re-review that skill's levels. Any re-levelling is a separately versioned bank change, which makes `rv2`.

### 5.2 Task A: answer annotation

**Sample.** 30 replies from `rv1-annot` (seed 1402), stratified by the simulator's plan and the system's output:

| Stratum | Replies |
| --- | --- |
| Planned correct | 6 |
| Planned partial or wrong, system named no misconception | 6 |
| Planned planted misconception | 6 |
| System named a misconception that was **not planted** ("extra claim") | 8 |
| Planned misread or slip | 4 |

- **Skills:** 10 per skill as far as the strata allow.
- **Spread:** at most 3 replies per interview, and at least 6 of the 30 are follow-up replies.
- **Short strata:** if a stratum has too few replies, fill from the next one in table order, and record that.

**Shown** per item:

- the question (text, code, tables), its rubric, key points and listed misconceptions (descriptions);
- the reply;
- for a follow-up, the earlier replies of that exchange.

**Hidden** in phase 1: the system's grade and diagnosis, the plan and outcome, and the candidate key.

**Phase 1 labels (blind):**

| Field | Values | Rule |
| --- | --- | --- |
| `grade` | 0, 1, 2 | Rubric (spec section 7); a listed misconception shown is 0 |
| `cause` | correct, gap, misconception, misread, slip | `correct` only with grade 2 |
| `belief` | a misconception id from the question's list, `none`, or `other` | The belief the reply *actually expresses*; `other` = an unlisted wrong belief, described in `note` |
| `quote` | exact substring of the reply | Required when `belief` is not `none` |
| `sure` | yes, no | Used for the disagreement analysis only |

**Phase 2 labels** come after phase 1 is frozen, still independent. For each item where the system named a misconception, show that id and description and label:

- `claim_supported`: yes, partly or no;
- `claim_quote`: the exact substring that supports it.

**Phase 3, adjudication.** The raters meet, agree one label per field, record why, and keep the original sheets.

**Phase 4, comparison with the system.** This is done only after adjudication, and each is reported separately:

- **Human–human agreement:** phases 1 and 2.
- **Grader:** final grade against the adjudicated `grade`.
- **Diagnoser:** the share of named claims judged supported (*human-supported precision*). Split it four ways:
  - planted and supported;
  - planted and unsupported;
  - not planted but supported: the simulator expressed an unplanted belief, so the ground truth is incomplete;
  - not planted and unsupported: the diagnoser stretched.
- **Simulator adherence:** the adjudicated `grade`/`belief` against the plan's `expected_score`/`misconception_id`.

Planted-belief recall is reported from the run metrics, separately from human support.

### 5.3 Agreement scoring and acceptance

| Label | Statistic | Primary |
| --- | --- | --- |
| Task D level | Quadratic-weighted kappa; raw agreement | yes |
| `grade` | Quadratic-weighted kappa; raw agreement | yes |
| `cause` | Unweighted Cohen's kappa over 5 categories | yes |
| `belief` | Kappa on the binary "a listed misconception is expressed"; exact-id agreement reported beside it | yes |
| `claim_supported` | Kappa on binary yes vs (partly or no); the 3-way kappa reported beside it | yes |

Report each with the marginal label counts and a 95% bootstrap interval over items (2,000 draws, seed 0). With about 30 items and skewed labels, kappa is unstable, so raw agreement is always shown next to it.

**Acceptance** (the threshold already declared in `report/report.tex`): a primary label is acceptable when the kappa point estimate is at least **0.6**. The interval is reported but is not the criterion.

**Contingency** (as declared in the report):

- **Below 0.6:** treat the labelling guide as ambiguous first. Rewrite it, settle the disagreements together, and label a **fresh** sample of 30 (seed 1403) with the revised guide. Re-labelling the same items does not count.
- **Raters agree with each other but not with the model:** that is a model error. Report its error rate next to every accuracy figure.
- **Fresh sample still below 0.6:** report that label as unreliable, and do not use it to support a claim.

Also use a fresh sample after any diagnosis-prompt or model change (Step 4 or the pilot). A sample that guided a change cannot also validate it.

## 6. Evidence fields evaluation needs (contract with Souradeep)

**Already produced** (commit `c297b3f`), and enough for C1–C3 on planted truth:

- per finding: `id`, `confirmed`, `evidence_turns`, `vote_share`, `confirmation_unit`, `confirmation_count`, `required_confirmations`, `evidence[]`;
- per `evidence[]` record: `question_id`, `turn`, `reply_index`, `reply_text`, `vote_share`;
- per-reply `diagnosis` on turns and follow-ups (needed for the false-alarm rate).

A4 writes the same fields, because the tracker records evidence on every diagnosed reply, so A2-rev and A4 are scored by identical code.

**Requested from Step 4.** All fields are optional so that old results stay readable; evaluation treats a missing field as "not produced":

| Field | Where | Needed for |
| --- | --- | --- |
| `status`: `confirmed`, `suspected` or `disputed` | finding (keep the boolean `confirmed`) | Counting disputed findings separately from suspected ones |
| `supporting_quote`, plus `quote_start` and `quote_end` | each `evidence[]` record | Checking that the quote is an exact substring of `reply_text`; comparing it with the human `quote` |
| `contradicting[]`, same record shape plus `reason` | finding | Contradiction analysis; precision among findings with and without contradictions |
| Diagnoser prompt version or hash | `config` or the manifest | Telling A2 and A2-rev apart if only the prompt changes |

**Requested from Step 6** (stopping): at the end of the interview, per skill, the posterior mass inside the reported level band, next to the existing `estimate`, `se`, `level` and `meets_minimum`. This lets evaluation study boundary errors without re-running the scorecard.

Agreeing this contract is the handoff. Neither of us edits the other's modules until it is agreed (assessment section 8).

## 7. Runner and reporting support needed (Step 5, Nilkanta)

These are not implemented yet; they are listed so the order of work is clear.

1. **Run manifest.** Write `results/<run>/manifest.json` before the first interview:
   - code commit and dirty flag;
   - bank, catalog, topics and jobs hashes, and the hash of each candidate file;
   - each condition's `config.to_dict()`;
   - provider, models and simulator model;
   - prompt file hashes (`part3/prompts.py`, `part4/simulator.py`);
   - seeds, operator and share.

   Refuse to add to a run whose manifest differs.
2. **Explicit candidate selection.** `--ids` or an id-list file, plus a `--candidates-dir` for the validation populations.
3. **Validation candidates.** A `make_candidates` id prefix, so they can be `VAL-A..`/`VAL-P..`. `load_results` must accept that prefix (today it matches `CAND-*.json` only).
4. **Presets** for ADAPT12 and job-blind A2-rev (and A2M if it is run), each with a test that the configuration differs from its reference only as declared.
5. **Per-comparison pairing**, the comparisons in section 2 run through `evaluate.py`, and premature-stop statistics with denominators.
6. **Annotation scoring script** (Step 3): weighted and unweighted kappa, raw agreement, bootstrap intervals, phase 4 tables.
7. **Explicit bank root for the report:** `make_figures.py` must not silently use the working-tree bank for historical sweeps (assessment section 7.F.2).

`rv1-annot` needs items 1–3 before Step 3 can start. These three are the first Step 5 tasks, done once Souradeep accepts this protocol.

## 8. Budget and recommended scope

Interviewer-side LLM calls per interview are given by the report's formula:

> 1 topic call + grader votes × main questions + grader votes × questions with follow-ups + diagnosis votes × diagnosed replies

The simulator makes one call per exchange. B1 and ADAPT12 are exact. The others take their stub run on the 60 evaluation candidates and scale it by the live/stub ratio of historical A2 (104 / 74.3 calls ≈ 1.4). They are **estimates**.

| Condition | Stub calls | Estimated live calls | Simulator calls | 60 candidates |
| --- | ---: | ---: | ---: | ---: |
| B1 | 37 | 37 | 12 | 2,220 + 720 |
| ADAPT12 | 37 | 37 | 12 | 2,220 + 720 |
| A2-rev (`A2Q`) | 76.2 | ≈107 | ≈22 | ≈6,400 + 1,300 |
| A4 | 54.1 | ≈75 (at most about 106) | ≈18 | ≈4,500 + 1,100 |
| A5 | 72.7 | ≈102 | ≈21 | ≈6,100 + 1,300 |
| A2 | 74.3 | 104 (historical) | 21 | 6,240 + 1,260 |
| A2M | — | ≈104 | ≈21 | ≈6,200 + 1,300 |

These are upper bounds. The disk cache reuses identical grader calls on the same answer across conditions, so the real cost is lower.

| Scope | Interviews | Interviewer calls | Simulator calls |
| --- | ---: | ---: | ---: |
| `rv1-annot` (A2 on 12) | 12 | ≈1,250 | ≈250 |
| `rv1-pilot` (4 conditions on 12) | 48 | ≈3,100 | ≈770 |
| **Main, recommended:** B1, A2-rev, A4, ADAPT12, A5 | 300 | ≈21,500 | ≈5,100 |
| plus A2 and A2M on blocks 1–2 (24 candidates, exploratory) | +48 | +≈5,000 | +≈1,000 |
| plus A2 and A2M on all 60 | +120 | +≈12,500 | +≈2,600 |

**Recommendation.** Run the 300-interview main matrix. Add A2 and A2M on the first 24 candidates only if the pilot shows the budget holds. Otherwise drop C6 and keep the RQ2 claim to the combined package. The final choice is frozen at Step 9, and the scope is not expanded midway.

The historical run used free tiers over four accounts plus about $1 of paid Gemini (report section 5). Per-call prices are not estimated here: check the provider's current pricing at freeze and record it in the manifest.

## 9. Handoff to Souradeep

- **Behavioural definitions:** sections 1, 1.1 and 1.2. Please confirm that A2-rev is `A2Q` plus your Step 4 changes, and whether those changes get a new preset name.
- **Evidence fields:** section 6. Please confirm or amend before either of us edits `followups.py`, `report.py` or the schema.
- **Minimum comparisons:** C1–C3 (section 2). C5 needs skill accuracy because of the stub interaction above.
- **For Step 3:** you prepare the samples and provenance (sections 5.1 and 5.2); I write the scoring script (section 7, item 6).
