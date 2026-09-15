# Porting tools

One-off tools used to build this repo out of the research tree it replaces.
They are **not** part of Occam and are not shipped in the wheel.

They are the only files here that name the old project, because their whole job
is to read it, so `tests/test_rename_hygiene.py` excludes this directory. Keep
it that way: anything that belongs in the shipped package does not belong here.

| file | what it does |
|---|---|
| `capture_goldens.py` | Run from the source repo **before** any rename, to capture the numerical fixtures in `golden/`. Blocking prerequisite for the whole port. |
| `port_file.py` | Copies one file across, applying the import rewrites and the rename rules. Ordered longest-pattern-first so `train_X_decoder` is rewritten before `train_X`. |

Both are read-only with respect to the source repo.
