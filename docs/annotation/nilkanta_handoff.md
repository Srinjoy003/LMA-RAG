# Nilkanta: Step 3 difficulty-check handoff

The blind difficulty-check generator is ready. Its instructions are in [README.md](README.md); the agreed study protocol is [evaluation_protocol.md](../evaluation_protocol.md), section 5.

## 1. Pull and inspect the implementation

```bash
git pull --ff-only
```

Files to review:

- `scripts/prepare_difficulty.py`: preparation command.
- `src/part4/difficulty.py`: sampling, rendering, blank sheets, and private provenance.
- `tests/test_difficulty.py`: sampling, blinding, reproducibility, and overwrite protection.
- `docs/annotation/README.md`: preparation and distribution instructions.

The full test suite passed with **498 tests** when this work was prepared. To verify locally:

```bash
python -m pytest
```

No new API calls or dependencies are needed to prepare the difficulty check. Generation does not modify the bank or the 60 evaluation candidates.

## 2. Receive the same packet Souradeep uses

Generated packets, rating sheets, and keys are gitignored. Pulling retrieves the preparation code, not Souradeep's local generated packet.

Souradeep will send a recipient-only archive with exactly:

- `packet.md`
- `instructions.md`
- `ratings/nilkanta.csv`

Use that packet for this round. It contains 21 existing reviewed-bank questions, seven each from Python, SQL, and data structures, sampled with seed 1401 and shuffled to opaque codes `D01`–`D21`. No new questions were generated. Original IDs, assigned levels, and topic tags are hidden. Reference answers and key points are intentionally shown under the agreed protocol.

Do not regenerate a different sample, inspect the hidden key, or look questions up in the bank while rating. Prefer a non-rater such as Kushal or Srinjoy to hold the key and collect submissions. Do not request Souradeep's ratings before submitting your own.

## 3. Complete your independent ratings first

Read the level definitions in `docs/part1_spec.md`, section 5. For each item, assess difficulty for a junior technical candidate and fill `level` with 1, 2, or 3. Add an optional `note`; keep the `item` and `rater` columns unchanged.

- Level 1: basics.
- Level 2: everyday application.
- Level 3: tricky cases and trade-offs.

Use the full skill-specific definitions, not just these short summaries. You are rating question difficulty, not marking candidate answers or writing solutions.

Submit the completed CSV to the agreed preparer. Preserve the original file. Both independent submissions must be frozen and hashed before key-based comparison or discussion of specific ratings. Do not commit filled ratings or private keys during the blind phase.

## 4. Implement scoring against the agreed file contract

Use synthetic fixtures to develop/test the scoring code before your real ratings are frozen. The generator produces CSV columns `item,rater,level,note`. The private key is a JSON list with `item`, `question_id`, `skill`, and `assigned_level`. The private manifest records the starting bank hashes, generator hash, seed, initial output hashes, and key hash.

The scorer should reject missing/duplicate/unknown items, invalid levels, mismatched packets, and unfilled sheets. Use item codes to join rows, not row order. Preserve input sheets; write scores and adjudication separately.

After both real submissions are frozen, compute:

- Quadratic-weighted Cohen's kappa and raw agreement between raters.
- Each rater's agreement with the assigned bank levels.
- Marginal label counts and uncertainty intervals as specified in the protocol.
- Per-skill counts where **both** raters disagree with the assigned level. More than two of seven triggers re-review for that skill.

Use the protocol's predeclared kappa threshold of 0.6 and its remediation procedure. Report undefined kappa explicitly rather than treating it as a passing score. Initial blank-sheet hashes are provenance, not hashes of completed submissions: record those separately.

## 5. Handoff back to Souradeep

Push the scoring implementation, meaningful tests, and usage instructions. During the blind phase, keep the key, completed sheets, and real-item scoring output private. After scoring is permitted, share the results and proposed next actions without altering either original submission.

The 30-answer annotation exercise is separate. It still needs the validation-population and safe-run support declared in Step 2. Do not use the final 60 evaluation candidates as a substitute for the annotation population. Full live evaluation and later tool integration are not prerequisites for completing this difficulty check.
