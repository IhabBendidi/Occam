#!/usr/bin/env python
"""Build GenePT embeddings over the full 18,216-gene perturbation vocabulary.

The all-source models combine these embeddings with GO, STRING, and ESM2.
This creates a separate asset from the benchmark-only GenePT table used by
competitor bridges and checks that shared gene vectors agree.

SOURCES
  full   : GenePT v2 release, GenePT_gene_protein_embedding_model_3_text (133,736 keys x 3,072)
  theirs : PerturbDiff's 2,024-key pickle -- only for its 15 alias keys absent from the full
           release; their value wins on a shared key (identical anyway), exactly the rule of
           scripts/build_genept_for_occam.py, so the benchmark symbols get the same vectors.

RETIRED-NAME HANDLING (a vocabulary symbol with no GenePT key)
  The vocabulary mixes retired and current HGNC symbols (HARS / HARS1, H2AFZ / H2AZ1). For a
  missing symbol g, candidate aliases are tried in this order, first one with a GenePT key wins:
    1. the current symbol of the unique gene whose NCBI synonyms list g   (g is retired)
    2. the NCBI synonyms of the gene whose current symbol is g           (g is current)
    3. the current symbol of g's Ensembl ID under Occam's own map (assets/gene_ids + the
       benchmark gene axis)
  Alias source: obs['symbol'/'synonyms'] of the genbio-ai foundation-models-perturbation gene
  table (NCBI gene info, 23,122 protein-coding genes). When the alias is itself a vocabulary
  gene, the two nodes share one vector (same gene); those pairs are listed in the sidecar.

    python scripts/build_genept_full_vocab.py [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = ("/rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/GenePT/genept_embeddings/"
        "extracted/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle.")
THEIRS = ("/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/assets_dl/perturb_data/"
          "gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl")
EXISTING = os.path.join(REPO, "data", "assets", "genept_occam_rhj_fewshot.pkl")
ALIASES = ("/rxrx/scratch/sandbox/ihab.bendidi/foundation-models-perturbation/benchmark/"
           "data/gene_embeddings/ESM2_(D=1280).h5ad")
VOCAB = os.path.join(REPO, "data", "graphs", "master_gene_set_sorted.csv")
BENCH = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot")
OCCAM_IDMAP = os.path.join(REPO, "assets", "gene_ids", "ensembl_to_symbol.tsv")
OUT = os.path.join(REPO, "data", "assets", "genept_full_occam_rhj_fewshot.pkl")
DIM = 3072


def _sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _symbol_to_ensembl():
    out = {}
    if os.path.isfile(OCCAM_IDMAP):
        df = pd.read_csv(OCCAM_IDMAP, sep="\t")
        for e, s in zip(df["ensembl_id"].astype(str), df["gene_symbol"].astype(str)):
            out.setdefault(s, e)
    ax = os.path.join(BENCH, "gene_axis.csv")
    if os.path.isfile(ax):
        df = pd.read_csv(ax)
        for e, s in zip(df["ensembl_gene_id"].astype(str), df["gene_name"].astype(str)):
            out.setdefault(s, e)
    return out


def _alias_tables(path):
    import anndata as ad

    obs = ad.read_h5ad(path, backed="r").obs
    sym = obs["symbol"].astype(str).values
    syn = obs["synonyms"].astype(str).values
    ens = np.asarray(obs.index, dtype=str)
    retired_to_current, current_to_syns = {}, {}
    for s, field in zip(sym, syn):
        if not s or s == "nan":
            continue
        names = [x.strip() for x in field.split("|")] if field and field != "nan" else []
        current_to_syns.setdefault(s, []).extend(names)
        for n in names:
            retired_to_current.setdefault(n, set()).add(s)
    ens_to_sym = {e: s for e, s in zip(ens, sym) if s and s != "nan"}
    return retired_to_current, current_to_syns, ens_to_sym


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--theirs", default=THEIRS)
    ap.add_argument("--aliases", default=ALIASES)
    ap.add_argument("--vocab", default=VOCAB)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(EXISTING):
        raise SystemExit(f"refusing to overwrite {EXISTING} (used by the competitor bridge)")

    full = pickle.load(open(args.full, "rb"))
    theirs = pickle.load(open(args.theirs, "rb"))
    src = {str(k): v for k, v in full.items()}
    src.update({str(k): v for k, v in theirs.items()})     # their value wins (identical anyway)
    vocab = list(pd.read_csv(args.vocab)["gene_name"].astype(str))
    vocab_set = set(vocab)
    r2c, c2s, e2s = _alias_tables(args.aliases)
    s2e = _symbol_to_ensembl()

    table, route, pairs = {}, {}, []
    for g in vocab:
        if g in src:
            table[g], route[g] = src[g], "direct"
            continue
        cands = []
        cur = r2c.get(g, set())
        if len(cur) == 1:
            cands.append(("retired->current", next(iter(cur))))
        cands += [("current->synonym", s) for s in c2s.get(g, [])]
        if s2e.get(g) in e2s:
            cands.append(("ensembl->current", e2s[s2e[g]]))
        for how, alias in cands:
            if alias != g and alias in src:
                table[g], route[g] = src[alias], how
                pairs.append({"symbol": g, "alias": alias, "route": how,
                              "alias_in_vocab": alias in vocab_set})
                break
    table = {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in table.items()}
    bad = [k for k, v in table.items() if v.shape[0] != DIM]
    if bad:
        raise SystemExit(f"{len(bad)} keys not {DIM}-dim, e.g. {bad[:3]}")

    # the benchmark symbols must get exactly the vectors of the existing asset
    existing = pickle.load(open(EXISTING, "rb"))
    worst, n_cmp = 0.0, 0
    for k, v in existing.items():
        if k in table:
            worst = max(worst, float(np.abs(table[k] - np.asarray(v, dtype=np.float32)).max()))
            n_cmp += 1
    print(f"[genept-full] value check vs existing asset on {n_cmp} keys: max abs diff {worst:.3e}")
    if worst > 1e-6:
        raise SystemExit("disagrees with data/assets/genept_occam_rhj_fewshot.pkl; refusing")

    cov = {"vocab": {"n": len(vocab), "present": len(table)}}
    bp = pd.read_csv(os.path.join(BENCH, "perturbations.csv"))
    for split in ["all"] + sorted(bp["split"].unique()):
        sub = bp if split == "all" else bp[bp["split"] == split]
        genes = sorted(set(sub["gene_name"].astype(str)))
        miss = [g for g in genes if g not in table]
        cov[f"rhj_fewshot_{split}"] = {"n": len(genes), "present": len(genes) - len(miss),
                                       "missing": miss}
    routes = {k: sum(1 for v in route.values() if v == k)
              for k in ("direct", "retired->current", "current->synonym", "ensembl->current")}
    for k, v in cov.items():
        print(f"  coverage {k:28s} {v['present']:>6}/{v['n']:<6}"
              + (f" missing {v['missing'][:10]}" if v.get("missing") else ""))
    print(f"  routes {routes}; alias shared with another vocab node: "
          f"{sum(p['alias_in_vocab'] for p in pairs)}")
    if args.dry_run:
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(table, fh, protocol=4)
    meta = {
        "built": "scripts/build_genept_full_vocab.py",
        "date": datetime.date.today().isoformat(),
        "format": "pickle dict {gene_symbol: np.float32[3072]}; absent key = no GenePT vector",
        "dim": DIM, "n_keys": len(table), "sha256": _sha256(args.out),
        "vocab": os.path.relpath(args.vocab, REPO), "vocab_sha256": _sha256(args.vocab),
        "sources": {
            "full_release": {"path": args.full, "n_keys": len(full), "sha256": _sha256(args.full)},
            "perturbdiff_pickle": {"path": args.theirs, "n_keys": len(theirs),
                                   "sha256": _sha256(args.theirs)},
            "alias_table": {"path": args.aliases, "sha256": _sha256(args.aliases),
                            "what": "obs symbol/synonyms (NCBI gene info) of genbio-ai "
                                    "foundation-models-perturbation ESM2_(D=1280).h5ad"},
            "occam_ensembl_map": os.path.relpath(OCCAM_IDMAP, REPO),
        },
        "merge_rule": "full release ∪ PerturbDiff pickle, their value on a shared key",
        "retired_name_handling": {
            "order": ["retired->current (unique NCBI synonym owner)",
                      "current->synonym (NCBI synonyms of g)",
                      "ensembl->current (Occam Ensembl map)"],
            "routes": routes,
            "aliased": pairs,
        },
        "value_check_vs_existing_asset": {"path": os.path.relpath(EXISTING, REPO),
                                          "n_keys": n_cmp, "max_abs_diff": worst},
        "coverage": cov,
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"[genept-full] wrote {os.path.relpath(args.out, REPO)} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB) sha256 {meta['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
