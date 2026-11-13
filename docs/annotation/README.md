# Annotation sheets

Rules and scoring: [evaluation_protocol.md](../evaluation_protocol.md), section 5. Templates hold the header row only. The preparer fills `item`; each rater copies the file, fills their own columns, and submits it unchanged after the deadline.

| File | One row per | Columns filled by the rater |
| --- | --- | --- |
| `difficulty_sheet.template.csv` | item `D01`–`D21` | `level` (1, 2 or 3), `note` |
| `answer_labels.template.csv` | item and phase (`1` blind, `2` claim check) | Phase 1: `grade` (0/1/2), `cause` (correct/gap/misconception/misread/slip), `belief` (misconception id, `none` or `other`), `quote` (exact substring of the reply), `sure` (yes/no). Phase 2: `claim_supported` (yes/partly/no), `claim_quote`. `claim_shown` is filled by the preparer in phase 2. |

- Items use opaque codes; the key from item code to question id, level, plan and system output is held by the preparer.
- Submitted sheets are hashed and never edited. Adjudicated labels go in a separate file with a reason per change.
- Filled sheets and keys are kept out of git until both raters have submitted.
