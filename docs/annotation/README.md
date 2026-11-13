# Annotation sheets

Rules and scoring: [evaluation_protocol.md](../evaluation_protocol.md), section 5. Templates hold the header row only. The preparer fills `item`; each rater copies the file, fills their own columns, and submits it unchanged after the deadline.

| File | One row per | Columns filled by the rater |
| --- | --- | --- |
| `difficulty_sheet.template.csv` | item `D01`–`D21` | `level` (1, 2 or 3), `note` |
| `answer_labels.template.csv` | item and phase (`1` blind, `2` claim check) | Phase 1: `grade` (0/1/2), `cause` (correct/gap/misconception/misread/slip), `belief` (misconception id, `none` or `other`), `quote` (exact substring of the reply), `sure` (yes/no). Phase 2: `claim_supported` (yes/partly/no), `claim_quote`. `claim_shown` is filled by the preparer in phase 2. |

- Items use opaque codes; the key from item code to question id, level, plan and system output is held by the preparer.
- Submitted sheets are hashed and never edited. Adjudicated labels go in a separate file with a reason per change.
- Filled sheets and keys are kept out of git until both raters have submitted.

## Step 3: prepare the blind difficulty check

Ask a non-rater (preferably Kushal or Srinjoy) to run this from the repository root:

```bash
python scripts/prepare_difficulty.py
```

The generator uses seed **1401**, selects **seven questions per skill** from ID-sorted bank inputs, then shuffles all 21 into `D01`–`D21`. It makes no API calls, assigns no ratings, and changes no bank or candidate files.

Default output, entirely under the gitignored `results/` directory:

```text
results/annotation/difficulty-v1/
    packet.md
    instructions.md
    ratings/souradeep.csv
    ratings/nilkanta.csv
    private/key.json
    private/manifest.json
```

The packet includes question text, code, tables, model answer, and key points. Original IDs, assigned levels, and topic tags are not included. Actual code language and conceptual content can still suggest difficulty; this is metadata blinding, not a guarantee that an experienced reviewer cannot recognise an item.

**Distribution:** send each rater `packet.md`, `instructions.md`, and only their own CSV. Supply both with the level definitions in `docs/part1_spec.md`, section 5. Do not distribute the whole generated directory: it contains the hidden key and the other person's sheet. Do not look up packet items in the bank or discuss ratings before submitting.

The private key maps item codes to original IDs, skill, and assigned level. The private manifest records seed, sampling algorithm, source commit, hashes of all question input files, generator hash, initial public-file hashes, and key hash. Keep it with the preparer until both raters have submitted. On submission, preserve and hash each completed CSV separately; the initial-sheet hashes in the manifest are not hashes of the completed ratings.

Outputs have a local `.gitignore` that ignores the generated contents, even when `--out` selects another location in the repository. The private directory has owner-only permissions on systems that support them. This prevents accidental ordinary staging; it does not enforce blinding against somebody with access to the source bank or filesystem.

The command refuses an existing output directory, so rerunning cannot overwrite ratings. Do not change the seed or retry to cherry-pick a different sample. If rebuilding an identical copy for distribution, use a fresh output path:

```bash
python scripts/prepare_difficulty.py --out results/annotation/difficulty-v1-copy
```

Identical source contents, generator, seed, and sample size reproduce the same packet, key, and initial sheets. The source commit is provenance; the input hashes also record uncommitted bank contents. `--root` selects another bank root; `--seed` and `--per-skill` are available for explicitly documented future protocols, not silent overrides of this check.

### Handoff and remaining work

Souradeep provides the generator, tests, and these instructions. Both Souradeep and Nilkanta independently fill levels 1–3 and optional notes. After submissions are frozen, Nilkanta's scoring task uses the key to compute quadratic-weighted kappa, raw agreement, and agreement against assigned levels.

The protocol requires re-reviewing a skill's levels if both raters disagree with the assigned level on more than two of its seven questions. Preserve original submissions and keep adjudication separate. Rating, scoring, and any subsequent bank changes are not performed by this preparation command.

The separate 30-answer check still needs fresh validation-run support and answer sampling. This command does not generate candidate interviews or use final evaluation candidates for annotation.
