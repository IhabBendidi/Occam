#!/usr/bin/env python
"""Build the ESM2 protein embeddings used by the all-source virtual-cell models.

OUTPUT (same shape as the GenePT asset data/assets/genept_occam_rhj_fewshot.pkl)
    data/assets/esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl   dict {gene_symbol: float32[1280]}
    ... .pkl.json                                           provenance + coverage report
Genes with no protein vector are ABSENT from the dict; FrozenEmbeddingModel gives them a zero row
plus one shared learned "missing" vector.

TWO SOURCES
-----------
--source precomputed (default, CPU, ~1 min)
    Re-keys an existing ESM2 per-gene table. Found offline on 2026-09-23:
      genbio-ai/foundation-models-perturbation (HF dataset, commit d67ba210...),
      gene_embeddings/ESM2_(D=1280).h5ad -- 23,122 protein-coding Ensembl genes x 1280.
    Dim 1280 = esm2_t33_650M_UR50D; model + pooling are not recorded in the file (inferred).
    Rows are keyed by Ensembl ID; symbols come from obs['symbol']. Symbol -> row resolution:
      1. direct symbol match. Several Ensembl IDs per symbol (alt-haplotype copies): if the rows
         are identical take them; else prefer the ID Occam's own map assigns the symbol
         (assets/gene_ids/ensembl_to_symbol.tsv); else the lowest ENSG.
      2. via Occam's Ensembl map: symbol -> ENSG (assets/gene_ids + the benchmark gene axis) ->
         row (catches symbols the table spells differently).
      3. via obs['synonyms']: only if exactly one row lists the symbol.
    Routes 2-3 recover RETIRED HGNC symbols (HARS -> HARS1, H2AFZ -> H2AZ1, HIST1H2AB -> H2AC4).
    The vocabulary holds some genes under both names, so such a pair shares one protein vector
    (it is one protein); the pairs are listed in the report.
    Each route's count goes in the report.

--source compute (GPU; scripts/slurm/build_esm2_embeddings.sbatch)
    Runs ESM2 itself. Needs (neither is on the cluster as of 2026-09-23):
      --fasta : UniProt human reference proteome, one canonical protein per gene
                (UP000005640_9606.fasta). Header `GN=<symbol>` gives the gene; when a symbol has
                several entries, Swiss-Prot (sp|) beats TrEMBL (tr|), then the longest.
      weights : fair-esm (`esm.pretrained.<model>()`) or transformers `facebook/<model>` (HF cache
                or network).
    Mean pooling over residue tokens (BOS/EOS/pad excluded). Proteins longer than 1022 residues
    are split into 1022-residue windows and the window means are averaged weighted by length.

    python scripts/build_esm2_embeddings.py                       # precomputed
    python scripts/build_esm2_embeddings.py --source compute --fasta UP000005640_9606.fasta
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
VOCAB = os.path.join(REPO, "data", "graphs", "master_gene_set_sorted.csv")
BENCH = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot")
OCCAM_IDMAP = os.path.join(REPO, "assets", "gene_ids", "ensembl_to_symbol.tsv")
PRECOMPUTED = os.path.join(REPO, "data", "gene_embeddings", "ESM2_(D=1280).h5ad")
PRECOMPUTED_PROVENANCE = {
    "hf_dataset": "genbio-ai/foundation-models-perturbation",
    "hf_commit": "d67ba21019814bb55dbdb448b3b84d18bbc693b8",
    "file": "gene_embeddings/ESM2_(D=1280).h5ad",
    "model": "esm2_t33_650M_UR50D (inferred from dim 1280; not recorded in the file)",
    "pooling": "not recorded in the file (per-protein mean pooling assumed)",
}
DEFAULT_MODEL = "esm2_t33_650M_UR50D"


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_vocab(path=VOCAB):
    return list(pd.read_csv(path)["gene_name"].astype(str))


def _symbol_to_ensembl():
    """Occam's own symbol -> Ensembl map (encoder vocab map + the benchmark gene axis)."""
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


# ---------------------------------------------------------------------------------------------
# source = precomputed
# ---------------------------------------------------------------------------------------------
def from_precomputed(h5ad_path, vocab):
    import anndata as ad

    a = ad.read_h5ad(h5ad_path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = X.astype(np.float32)
    ens = np.asarray(a.obs_names, dtype=str)
    sym = a.obs["symbol"].astype(str).values
    syn = a.obs["synonyms"].astype(str).values if "synonyms" in a.obs else None
    ens_row = {e: i for i, e in enumerate(ens)}
    by_sym = {}
    for i, s in enumerate(sym):
        if s and s != "nan":
            by_sym.setdefault(s, []).append(i)
    s2e = _symbol_to_ensembl()
    vocab_set = set(vocab)

    table, route = {}, {}
    n_dup_identical = n_dup_occam = n_dup_lowest = 0
    for g in vocab:
        rows = by_sym.get(g)
        if not rows:
            continue
        if len(rows) == 1:
            r = rows[0]
        elif all(np.array_equal(X[rows[0]], X[k]) for k in rows[1:]):
            r, n_dup_identical = rows[0], n_dup_identical + 1
        elif s2e.get(g) in ens_row and ens_row[s2e[g]] in rows:
            r, n_dup_occam = ens_row[s2e[g]], n_dup_occam + 1
        else:
            r, n_dup_lowest = min(rows, key=lambda k: ens[k]), n_dup_lowest + 1
        table[g], route[g] = X[r], "symbol"

    shared = []                                       # (old symbol, current symbol) both in vocab
    for g in vocab:                                   # 2. via Occam's Ensembl map
        if g in table:
            continue
        e = s2e.get(g)
        if e in ens_row:
            table[g], route[g] = X[ens_row[e]], "ensembl"
            if sym[ens_row[e]] in vocab_set:
                shared.append((g, sym[ens_row[e]]))

    if syn is not None:                               # 3. unambiguous synonym
        syn_rows = {}
        for i, field in enumerate(syn):
            if field and field != "nan":
                for s in field.split("|"):
                    syn_rows.setdefault(s.strip(), set()).add(i)
        for g in vocab:
            if g in table:
                continue
            rows = syn_rows.get(g, set())
            if len(rows) == 1:
                r = next(iter(rows))
                table[g], route[g] = X[r], "synonym"
                if sym[r] in vocab_set:
                    shared.append((g, sym[r]))

    info = {
        "source": "precomputed",
        "source_path": h5ad_path,
        "source_sha256": _sha256(h5ad_path),
        "source_shape": list(X.shape),
        "source_provenance": PRECOMPUTED_PROVENANCE,
        "dup_symbol_resolution": {"identical_rows": n_dup_identical,
                                  "occam_ensembl_map": n_dup_occam, "lowest_ensg": n_dup_lowest},
        "routes": {k: sum(1 for v in route.values() if v == k)
                   for k in ("symbol", "ensembl", "synonym")},
        # The master gene set carries BOTH retired and current HGNC symbols for some genes
        # (HARS and HARS1, H2AFZ and H2AZ1, ...). The data perturb under the retired name, so
        # the retired symbol gets the current symbol's protein vector: same protein.
        "shared_with_current_symbol": {"n": len(shared), "pairs": shared},
    }
    return table, info


# ---------------------------------------------------------------------------------------------
# source = compute
# ---------------------------------------------------------------------------------------------
def read_uniprot_fasta(path):
    """{gene_symbol: sequence}, canonical choice: Swiss-Prot over TrEMBL, then longest."""
    best = {}

    def _commit(header, seq):
        if header is None:
            return
        gn = None
        for tok in header.split():
            if tok.startswith("GN="):
                gn = tok[3:]
        if not gn:
            return
        rank = (1 if header.startswith(">sp|") else 0, len(seq))
        if gn not in best or rank > best[gn][0]:
            best[gn] = (rank, seq)

    header, parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                _commit(header, "".join(parts))
                header, parts = line, []
            elif line:
                parts.append(line)
    _commit(header, "".join(parts))
    return {g: s for g, (_, s) in best.items()}


def _load_esm(model_name, device):
    """(embed_fn, backend). embed_fn(list[str]) -> list[np.ndarray] of per-residue means."""
    import torch

    try:
        import esm  # fair-esm

        model, alphabet = getattr(esm.pretrained, model_name)()
        model = model.eval().to(device)
        conv = alphabet.get_batch_converter()
        n_layers = model.num_layers

        @torch.no_grad()
        def embed(seqs):
            _, _, toks = conv([(str(i), s) for i, s in enumerate(seqs)])
            toks = toks.to(device)
            rep = model(toks, repr_layers=[n_layers])["representations"][n_layers]
            return [rep[i, 1:len(s) + 1].float().mean(0).cpu().numpy() for i, s in enumerate(seqs)]

        return embed, "fair-esm"
    except ImportError:
        from transformers import AutoTokenizer, EsmModel

        tok = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        model = EsmModel.from_pretrained(f"facebook/{model_name}").eval().to(device)

        @torch.no_grad()
        def embed(seqs):
            b = tok(seqs, return_tensors="pt", padding=True).to(device)
            h = model(**b).last_hidden_state
            out = []
            for i, s in enumerate(seqs):
                out.append(h[i, 1:len(s) + 1].float().mean(0).cpu().numpy())
            return out

        return embed, "transformers"


def from_compute(fasta, vocab, model_name, device, batch_tokens=4096, window=1022):
    import torch  # noqa: F401

    seqs = read_uniprot_fasta(fasta)
    embed, backend = _load_esm(model_name, device)
    todo = [g for g in vocab if g in seqs]
    # windows: (gene, piece, length); long proteins -> length-weighted mean of window means
    pieces = []
    for g in todo:
        s = seqs[g]
        for k in range(0, len(s), window):
            pieces.append((g, s[k:k + window]))
    pieces.sort(key=lambda p: len(p[1]))
    acc = {g: [] for g in todo}
    i, nb = 0, 0
    while i < len(pieces):
        # pieces are sorted by length: grow the batch while (count x longest) fits the budget
        n = 1
        while i + n < len(pieces) and (n + 1) * (len(pieces[i + n][1]) + 2) <= batch_tokens:
            n += 1
        batch = pieces[i:i + n]
        for (g, s), v in zip(batch, embed([s for _, s in batch])):
            acc[g].append((len(s), v))
        i, nb = i + n, nb + 1
        if nb % 100 == 0:
            print(f"  embedded {i}/{len(pieces)} windows", flush=True)
    table = {}
    for g, parts in acc.items():
        w = np.asarray([p[0] for p in parts], dtype=np.float64)
        v = np.stack([p[1] for p in parts]).astype(np.float64)
        table[g] = ((w[:, None] * v).sum(0) / w.sum()).astype(np.float32)
    info = {
        "source": "compute",
        "fasta": os.path.abspath(fasta),
        "fasta_sha256": _sha256(fasta),
        "n_fasta_genes": len(seqs),
        "model": model_name,
        "backend": backend,
        "pooling": "mean over residue tokens (BOS/EOS/pad excluded); >1022 aa: length-weighted "
                   "mean of 1022-residue window means",
        "canonical_rule": "Swiss-Prot over TrEMBL, then longest, per GN= symbol",
    }
    return table, info


# ---------------------------------------------------------------------------------------------
def coverage_report(table, vocab):
    rep = {"vocab": {"n": len(vocab), "present": sum(g in table for g in vocab)}}
    p = os.path.join(BENCH, "perturbations.csv")
    if os.path.isfile(p):
        bp = pd.read_csv(p)
        for split in ["all"] + sorted(bp["split"].unique()):
            sub = bp if split == "all" else bp[bp["split"] == split]
            genes = sorted(set(sub["gene_name"].astype(str)))
            miss = [g for g in genes if g not in table]
            rep[f"rhj_fewshot_{split}"] = {"n": len(genes), "present": len(genes) - len(miss),
                                           "missing": miss[:50]}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["precomputed", "compute"], default="precomputed")
    ap.add_argument("--precomputed", default=PRECOMPUTED)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--vocab", default=VOCAB)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    vocab = load_vocab(args.vocab)
    if args.source == "precomputed":
        table, info = from_precomputed(args.precomputed, vocab)
        model = DEFAULT_MODEL
    else:
        if not args.fasta:
            ap.error("--source compute needs --fasta (UniProt human reference proteome)")
        table, info = from_compute(args.fasta, vocab, args.model, args.device, args.batch_tokens)
        model = args.model
    out = args.out or os.path.join(REPO, "data", "assets", f"{model}_occam_rhj_fewshot.pkl")
    dims = {v.shape[0] for v in table.values()}
    assert len(dims) == 1, dims
    rep = coverage_report(table, vocab)
    meta = {
        "built": "scripts/build_esm2_embeddings.py",
        "date": datetime.date.today().isoformat(),
        "dim": int(next(iter(dims))),
        "n_keys": len(table),
        "format": "pickle dict {gene_symbol: np.float32[dim]}; absent key = no protein vector",
        "vocab": os.path.relpath(args.vocab, REPO),
        "vocab_sha256": _sha256(args.vocab),
        "coverage": rep,
        **info,
    }
    print(json.dumps({k: meta[k] for k in ("dim", "n_keys")}, indent=None))
    for k, v in rep.items():
        print(f"  coverage {k:28s} {v['present']:>6}/{v['n']:<6}")
    if "routes" in info:
        print(f"  routes {info['routes']}  dup-resolution {info['dup_symbol_resolution']}")
    if args.dry_run:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(table, fh, protocol=4)
    meta["sha256"] = _sha256(out)
    with open(out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"wrote {out}  sha256 {meta['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
