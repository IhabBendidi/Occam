#!/usr/bin/env python
"""Watch every launched competitor chain and say, in one table, what is actually wrong.

WHAT THIS IS FOR
    Four competitor arms (TxPert, STATE, CellFlow, PerturbDiff) each run a three-leg chain
    -- train -> export -> score -- under Slurm, for 3-20 h. The failures that have cost this
    port the most time are all SILENT:

      * a job vanishes from `squeue` with no log line saying why (OOM, preemption, my own
        `pkill`, a cancelled dependency);
      * a run trains happily for hours while EVERY validation raises inside a
        `try/except: continue`, so the defect only surfaces at the end as a
        `ModelCheckpoint` failure on a missing metric;
      * a chain's later legs sit `DependencyNeverSatisfied` forever after leg 1 failed.

    Reading `squeue` alone catches none of those: a job that is RUNNING is not a job that is
    WORKING. This script therefore reports, per arm, the Slurm state AND whether validation
    metrics are actually landing AND whether the job disappeared without finishing.

WHAT IT DOES NOT DO
    It never launches, cancels or resubmits anything (RULE 0). It only reads.

EXIT CODE
    0 = every tracked chain is either healthy or finished.
    1 = at least one arm needs attention. Use this to drive an until-loop.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "runs" / "competitors" / "watch.json"

# A validation has LANDED when one of these appears in the log. Each arm logs its own
# selection metric under its own key; matching the metric (not just "validation") is the
# point, because a run that enters validation and raises inside it still prints the former.
VAL_PATTERNS = {
    "txpert":      r"val/celleval/selection_score|selection_score=",
    "state":       r"val/celleval/selection_score|selection_score=",
    "cellflow":    r"\[selection\]|selection_score",
    "perturbdiff": r"validation/celleval/selection_score|selection_score",
}
# Anything here means the run is broken even if Slurm still calls it RUNNING.
FAIL_PATTERNS = (
    r"^Traceback", r"CUDA out of memory", r"Killed", r"srun: error",
    r"Signals\.SIG", r"oom-kill", r"DUE TO TIME LIMIT",
    # the swallowed-validation shape: the except branch logging its own failure
    r"validation .*failed", r"celleval .*failed", r"skipping validation",
)
FAIL_RE = re.compile("|".join(FAIL_PATTERNS), re.MULTILINE | re.IGNORECASE)


def sacct(jobid: str) -> tuple[str, str]:
    """(State, ExitCode) for a job id, from sacct -- which still knows it after it leaves."""
    out = subprocess.run(
        ["sacct", "-j", str(jobid), "-n", "-P", "-X", "-o", "State,ExitCode"],
        capture_output=True, text=True).stdout.strip()
    if not out:
        return ("UNKNOWN", "")
    st, _, ec = out.splitlines()[0].partition("|")
    return (st.strip(), ec.strip())


def in_queue() -> dict[str, str]:
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%T"],
        capture_output=True, text=True).stdout
    q = {}
    for ln in out.splitlines():
        jid, _, st = ln.partition("|")
        q[jid.strip()] = st.strip()
        # an array element 123_0 also answers for its parent 123
        base = jid.split("_", 1)[0]
        q.setdefault(base, st.strip())
    return q


def newest_log(arm: str, jobid: str) -> Path | None:
    pats = [f"*{jobid}*.out", f"*{jobid}*.err"]
    hits = [p for pat in pats for p in (REPO / "runs" / "slurm").glob(pat)]
    hits = [p for p in hits if p.stat().st_size > 0]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def read_logs(arm: str, jobid: str) -> str:
    txt = []
    for pat in (f"*{jobid}*.out", f"*{jobid}*.err"):
        for p in (REPO / "runs" / "slurm").glob(pat):
            try:
                txt.append(p.read_text(errors="replace"))
            except OSError:
                pass
    return "\n".join(txt)


def classify(arm: str, jobid: str, queued: dict[str, str]) -> dict:
    st_q = queued.get(str(jobid)) or queued.get(str(jobid).split("_")[0])
    st_acct, exitcode = sacct(jobid)
    logs = read_logs(arm, jobid)
    n_val = len(re.findall(VAL_PATTERNS.get(arm, r"selection_score"), logs))
    fail = FAIL_RE.search(logs)
    log = newest_log(arm, jobid)
    age_min = (time.time() - log.stat().st_mtime) / 60 if log else None

    if st_q in {"PENDING"}:
        verdict, why = "PENDING", st_acct
    elif st_q in {"RUNNING", "CONFIGURING", "COMPLETING"}:
        if fail:
            verdict, why = "⛔ BROKEN-WHILE-RUNNING", fail.group(0)[:60]
        elif age_min is not None and age_min > 90:
            verdict, why = "⚠️ STALLED", f"log silent {age_min:.0f} min"
        else:
            verdict, why = "RUNNING", f"{n_val} validation(s) landed"
    elif st_acct.startswith("COMPLETED"):
        verdict, why = "✅ COMPLETED", f"{n_val} validation(s) landed"
    elif st_acct == "UNKNOWN" and st_q is None:
        verdict, why = "⛔ VANISHED", "not in squeue and unknown to sacct"
    else:
        verdict, why = f"⛔ {st_acct}", (fail.group(0)[:60] if fail else f"exit {exitcode}")

    return {"arm": arm, "jobid": jobid, "slurm": st_acct, "verdict": verdict,
            "detail": why, "validations": n_val,
            "log": str(log.relative_to(REPO)) if log else None,
            "log_age_min": round(age_min, 1) if age_min is not None else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", nargs="*", metavar="ARM=JOBID",
                    help="add chains to the ledger, e.g. state=18097723 cellflow=18097800")
    ap.add_argument("--quiet", action="store_true", help="table only when something is wrong")
    a = ap.parse_args()

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tracked = json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    for item in (a.track or []):
        arm, _, jid = item.partition("=")
        tracked.setdefault(arm, [])
        if jid not in tracked[arm]:
            tracked[arm].append(jid)
    if a.track:
        LEDGER.write_text(json.dumps(tracked, indent=1, sort_keys=True))
        print(f"[watch] tracking {sum(len(v) for v in tracked.values())} job(s) "
              f"across {len(tracked)} arm(s) -> {LEDGER.relative_to(REPO)}")

    if not tracked:
        print("[watch] nothing tracked yet. Add with --track state=<jobid> ...")
        return 0

    queued = in_queue()
    rows = [classify(arm, jid, queued) for arm in sorted(tracked) for jid in tracked[arm]]
    bad = [r for r in rows if r["verdict"].startswith(("⛔", "⚠️"))]

    if not (a.quiet and not bad):
        w = max(len(r["arm"]) for r in rows)
        print(f"{'arm':<{w}}  {'job':<12} {'verdict':<24} {'val':>3}  detail")
        for r in rows:
            print(f"{r['arm']:<{w}}  {r['jobid']:<12} {r['verdict']:<24} "
                  f"{r['validations']:>3}  {r['detail']}")
    for r in bad:
        print(f"\n--- {r['arm']} {r['jobid']}: {r['verdict']} ---\n  {r['detail']}")
        if r["log"]:
            print(f"  log: {r['log']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
