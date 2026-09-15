# Upload gate

`scripts/check_upload_gate.py` must exit 0 before DiverseRNA is published.

**Nothing in this repository can upload anything.** There is no publish script,
and no code path imports `huggingface_hub` or calls the Hub API — deliberately.
This is preparation only: the shards, the card, the fingerprinting tool and this
gate. Writing an upload script is the *last* step, after the gate opens, and it
should call the gate first and refuse on non-zero.

**The gate does not currently pass, and that is the correct state.** Seven of
the nine constituent blocks are unattributed, no block has a licence, and the
two blocks that *are* identified — GTEx and TCGA — both carry redistribution
conditions. Publishing DiverseRNA is a legal-review item, not an engineering
one.

## Conditions

| # | condition | why |
|---|---|---|
| 1 | every block `CONFIRMED`, or carrying a signed waiver | you cannot licence what you cannot name |
| 2 | every block has a non-empty `license` and `citation` | the constituent datasets have their own terms and their own authors |
| 3 | shard checksums verify, and Σnnz and ΣX over the shards equal the source | proves the export did not silently drop or alter data |
| 4 | no `TBD` / `TODO` / `UNRESOLVED` in the card outside `## Known issues` | an unfinished card published is an unfinished card forever |
| 5 | `LICENSE_REVIEW.md` names a human approver and a date | someone accountable has actually looked |
| 6 | the repo-wide banned-token grep is clean | the old project name must not ship |

## Waivers

A block may be waived rather than confirmed, if someone is prepared to sign for
it. Add to `sources/waivers.json`:

```json
[
  {
    "block": 3,
    "waived_by": "Name <email>",
    "date": "2026-09-10",
    "rationale": "Confirmed by the corpus author as an internal dataset cleared for release; see <ticket>."
  }
]
```

A waiver is a statement that a human takes responsibility for that block's
provenance. It is not a way to make the gate quiet.

## Resolving the unattributed blocks

Blocks 0–6 have no exact row-count match against the candidate registry, which
is what a **subsetted** source looks like: the block is a sample of something
larger, so its row count matches nothing. Three routes, in increasing strength:

1. **Widen the registry.** `--candidates <dir>` over more locations. Cheap, and
   it is what identified GTEx and TCGA.
2. **Gene-set Jaccard.** Compare each block's supported-gene set against
   candidates' gene axes after stripping Ensembl versions. Distinguishes a
   whole-transcriptome source from a targeted panel, and survives subsetting.
3. **Row-level hashing.** Reindex a candidate onto the 44,728-gene axis and hash
   its rows against the block's. **A single hit proves identity.** This is the
   only axis that upgrades "consistent with" to "is", and the only one worth
   running once a shortlist exists.

The person who assembled the corpus is the cheapest route of all, and should be
asked first.
