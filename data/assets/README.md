# data/assets — per-gene embedding tables

Frozen per-gene vectors used by the perturbation model (and by the PerturbDiff competitor).
On the Hub under `assets/` in `bendidiihab/Occam`; they belong here, in `data/assets/`:

    hf download bendidiihab/Occam --include 'assets/*' --local-dir /tmp/dl && \
        mkdir -p data/assets && mv /tmp/dl/assets/* data/assets/

Configs reference them by these relative paths, so nothing else needs setting. Each `.pkl` has a
`.pkl.json` sidecar with its sha256, source, and coverage; the loader
(`build_pert_embedding_matrix`, `src/occam/vcell/adjacency_map.py`) refuses a table whose
sha256 does not match its sidecar.

| file | dim | genes | used by | rebuild |
|---|---|---|---|---|
| `genept_full_occam_rhj_fewshot.pkl` | 3072 | 17,843 / 18,216 of the master gene set | Occam GenePT arm, ALL-sources default (`pert_embedding_path(s)`) | `python scripts/build_genept_full_vocab.py` |
| `esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl` | 1280 | 18,149 / 18,216 | Occam ESM2 arm, ALL-sources default | `python scripts/ablation/build_esm2_embeddings.py` (precomputed source, CPU) |
| `genept_occam_rhj_fewshot.pkl` | 3072 | 2,376 (benchmark perturbations only) | PerturbDiff (GenePT) competitor on rhj | `competitors/PerturbDiff/build_genept_for_occam.py` |
| `genept_occam_norman_combo.pkl` | 3072 | Norman perturbations | PerturbDiff competitor on Norman | same |

The two `*_occam_rhj_fewshot` Occam tables are keyed by gene symbol over the 18,216-gene master
set (`data/graphs/master_gene_set_sorted.csv`), which rhj_fewshot and norman_combo share, so
the same files serve both benchmarks.

Provenance (full detail in each sidecar):
- **GenePT**: the GenePT v2 release (133,736 gene-symbol keys, text-embedding-3-large, 3072-d),
  plus PerturbDiff's pickle for 14 alias keys; retired symbols filled through NCBI synonyms /
  the Ensembl map (42 genes).
- **ESM2**: precomputed ESM2 (1280-d, esm2_t33_650M) per-gene protein embeddings from the
  Hugging Face dataset `genbio-ai/foundation-models-perturbation` @ `d67ba210`
  (`ESM2_(D=1280).h5ad`); the pooling is not recorded by the source.

Licensing of both sources is not recorded here; the Hub repo is private (see the README).
