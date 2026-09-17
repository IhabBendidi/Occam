# TxPert (GSPP) — Perturbation Prediction Pipeline

## What this is

TxPert (codebase name: GSPP) predicts single-cell gene expression changes caused by genetic perturbations (CRISPRi knockdowns). It uses a perturbation model (GNN or adjacency map) on knowledge graphs to learn gene embeddings, combined with a basal-state model (MLP/VAE) for the unperturbed cell state. The model predicts the perturbed expression profile given a control cell and a perturbation target gene.

## Project structure

```
txpert/
├── main.py                            # Hydra-based entry point
├── configs/                           # Hydra YAML configs
│   ├── config.yaml                    # Default (GNN + STRING, single CT)
│   ├── config-simple-adj-map-reverse-only.yaml  # Adjacency map, 4 KGs
│   ├── config-x-cell.yaml            # XCell benchmark (GNN)
│   ├── config-x-cell-adj-map.yaml    # XCell benchmark (adj map, 4 KGs)
│   ├── config-x-cell-baseline.yaml   # XCell mean baseline
│   └── ...
├── gspp/                              # Main package
│   ├── predictor.py                   # Lightning module (train/eval/save)
│   ├── evaluation.py                  # Inference + metric computation
│   ├── metrics.py                     # Pearson delta, MSE, MAE, R2
│   ├── constants.py                   # Cell type paths, graph dicts
│   ├── data/
│   │   ├── datamodule.py              # PertDataModule (single CT)
│   │   ├── datamodule_hybrid.py       # HybridDataModuleWrapper (multi CT)
│   │   ├── data_utils.py             # Loading, normalization, filtering
│   │   └── graphmodule.py            # Graph loading/creation
│   └── models/
│       ├── gears_derivative.py        # Main GSPP model
│       ├── baselines.py              # Mean/experimental baselines
│       └── pert_models/
│           ├── simple_adjacency_map.py  # Best performing: adj row → MLP
│           ├── gnn.py                   # GATv2/GCN/GIN message passing
│           └── ...
├── run_model.sh                       # SLURM template (single run)
├── wandb_agent.sh                     # SLURM template (sweep agents)
└── data/                              # Local graph data
```

## How to run

```bash
# Basic training (Hydra config system)
python main.py --config-name=config-simple-adj-map-reverse-only \
    wandb.name=my_run \
    training.epochs=50

# Override specific params via Hydra CLI
python main.py --config-name=config-x-cell-adj-map \
    datamodule.test_cell_type=hepg2 \
    datamodule.train_cell_types=all \
    training.lr=0.001 \
    wandb.project=TxPert-Rebuttal

# SLURM
sbatch run_model.sh
```

## Data pipeline

1. Raw h5ad files loaded from `PUBLIC_CELL_TYPE_PATHS` (hardcoded in `gspp/constants.py`)
2. HVG selection (default: 5000 genes)
3. `normalize_total(target_sum=4000)` + `log1p`
4. Control/treatment separation
5. Train/val/test split (75/12.5/12.5 or predefined)
6. PyTorch Lightning DataModule with batch sampling

### Key data paths

```
BASE_PATH_WITH_FM = /rxrx/data/user/ihab.bendidi/outgoing/biomodality_scgpt/
  nadig_2024/nadig_2024_hepg2.h5ad
  nadig_2024/nadig_2024_jurkat.h5ad
  replogle_2022/replogle_2022_k562_e.h5ad
  replogle_2022/replogle_2022_rpe1_e.h5ad

XCell data dir: /rxrx/data/user/wilson.tu/outgoing/data_xcell/
  nadig_hepg2/, nadig_jurkat/, replogle_k562_essential/, replogle_rpe1_essential/
  Each contains: perturb_processed.h5ad, data_pyg/, splits/

Graph cache: /rxrx/data/valence/pathfinder/modules/gspp/graphs/
Auxiliary:   /rxrx/data/valence/pathfinder/modules/gspp/auxiliary/
Data cache:  /rxrx/data/valence/pathfinder/modules/gspp/cache_v2/
```

## Model architecture

```
Control cell (expression) → Basal MLP → z_ctrl (latent)
                                           |
Perturbation gene → Adj Map / GNN → e_pert |
                                           |
                              z_ctrl + e_pert → Decoder → predicted expression
```

- **Basal model**: MLP (default), VAE, or MoE. Maps expression → latent (hidden_dim=512, latent_dim=64).
- **Perturbation model**: Adjacency map (best) or GNN. Maps gene → embedding.
- **Combination**: Additive (`z_ctrl + e_pert`) — no FiLM or gated interaction.
- **Decoder**: Linear layer from latent to expression space.
- **Loss**: MSE on predicted vs true expression.

## Knowledge graphs

Available graph sources (all in `/rxrx/data/valence/pathfinder/modules/gspp/graphs/`):

| Graph | Type | Description |
|-------|------|-------------|
| go | Gene Ontology | Functional annotations |
| string | STRING PPI v11.5 | Protein-protein interactions |
| pxmap-top1 | Phenomics | RxRx phenotypic similarities |
| txmap-top1 | Transcriptomics | RxRx transcriptomic similarities |
| reactome | Reactome | Metabolic/signaling pathways |

Default processing: `reduce2perts=true`, `mode=top_20`, `norm_weights=true` for GO only.

---

# TASK: Reproduce Best Adjacency Map Config + XCell Benchmark

## Goal

Retrain the best txpert adjacency map configuration on the XCell benchmark (HEPG2-only test set) and produce prediction files compatible with the ARC Institute cell-eval package, so results can be directly compared against IhabNet (scMAE virtual cell).

## Best adjacency map configuration (from 1400+ sweep runs)

This is the best-performing txpert setup, validated across 18 sweeps on K562:

```yaml
model:
  pert_model:
    model_type: simple_adjacency_map
    hidden_dim: 128
    dropout: 0.2
    adj_deg: 1
    normalize: none
    only_reverse: true
    self_loops: false
    make_undirected: true   # in existing config, "undir_or_addcol: none" = only_reverse
    kg_list: [go, string, pxmap, txmap]

graph:
  graph_cfg:
    go:
      graph_type: go
      reduce2perts: true
      norm_weights: true
      mode: top_20
    string:
      graph_type: string
      reduce2perts: true
      norm_weights: false
      mode: top_20
    pxmap:
      graph_type: pxmap-top1
      reduce2perts: true
      norm_weights: false
      mode: top_20
    txmap:
      graph_type: txmap-top1
      reduce2perts: true
      norm_weights: false
      mode: top_20

training:
  lr: 0.001
  epochs: 50
  weight_decay: 0.0
  lr_scheduler:
    type: cosine
    warmup_epochs: 5

model:
  hidden_dim: 512
  latent_dim: 64
  dropout: 0.2
  use_batch_norm: true
  mse_weight: 1

datamodule:
  batch_size: 64
  target_sum: 4000
  hvg_number: 5000
  obsm_key: raw
```

### Key findings from sweeps

- **All 4 KGs >> any subset**: K562 test pearson_delta: 4-KG=0.639, go+string=0.618, string=0.611
- **adj_deg=1 >> 2 >> 3**: deg1=0.639, deg2=0.608, deg3=0.542. Higher hops add noise.
- **normalize=none best**: none=0.633, col=0.622, sym=0.621, row=0.617.
- **Adjacency map beats GNN**: adj_map best=0.639, GNN avg=0.555
- **only_reverse=true best**: only_reverse=0.639, addcol=0.633, undir=0.626

Reference: `config-simple-adj-map-reverse-only.yaml` is the closest existing config.

## XCell benchmark setup

XCell is the ARC Institute perturbation prediction benchmark. It evaluates on HEPG2 only with fixed perturbation splits.

### What XCell mode means for training

1. **Test cell type**: HEPG2 (not K562)
2. **Train cell types**: all (HEPG2 train perts + all other cell types)
3. **Gene set**: 6,642 XCell benchmark genes (subset of all genes)
4. **Fixed splits**: Predefined val (104 perts) and test (624 perts) for HEPG2
5. **All non-HEPG2 perturbations go to training**

### Existing XCell config

`config-x-cell-adj-map.yaml` already exists with the right adjacency map setup. Key settings:
```yaml
datamodule:
  test_cell_type: K562          # CHANGE to hepg2
  train_cell_types: all
  use_predefined_splits: true   # Uses splits from data_xcell/nadig_hepg2/splits/
  match_cntr: true              # Match controls by batch for eval
  avg_cntr: false               # Don't average controls
```

### Running XCell with the best adj map config

Use `config-x-cell-adj-map.yaml` as base, override test_cell_type to hepg2:

```bash
python main.py --config-name=config-x-cell-adj-map \
    datamodule.test_cell_type=hepg2 \
    datamodule.train_cell_types=all \
    wandb.name=xcell_best_adj_map_hepg2 \
    wandb.project=TxPert-Rebuttal \
    training.epochs=50
```

### Predefined splits location

When `use_predefined_splits=true`, splits are loaded from:
```
/rxrx/data/user/wilson.tu/outgoing/data_xcell/nadig_hepg2/splits/
  nadig_hepg2_simulation_1_0.75.pkl          # {train: [...], val: [...], test: [...]}
  nadig_hepg2_simulation_1_0.75_subgroup.pkl # subgroup annotations
```

**Important**: Verify these pickle files contain the same 624 test perturbations as the XCell benchmark. The canonical XCell splits are stored at:
```
/rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/scMAE-main/virtual_cell/xcell_splits.json
```
This JSON has: val (104 perts), test (624 perts), cell_line="HEPG2". Compare against the pickle to ensure splits match.

## Saving predictions for cell-eval comparison

After training, predictions must be saved as an h5ad file for evaluation with the ARC Institute `cell-eval` package. The prediction file must match a specific format.

### Required prediction file format

```
AnnData object:
  .X: float32 matrix (n_samples, 6642)  — log1p normalized expression
  .obs: DataFrame with single column "target_gene"
        - Perturbation names for test samples (e.g., "TTF2", "INO80B", ...)
        - "non-targeting" for control samples
  .var: DataFrame with gene symbol index (6642 XCell genes, exact order must match real file)
```

### Required real data file

The real HEPG2 test data file is at:
```
/rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/scMAE-main/virtual_cell/cell_eval_data/real_hepg2_xcell_test.h5ad
```
- Shape: (19312, 6642) — 19256 test pert profiles + 56 control profiles
- Pseudobulked by (batch, perturbation), normalize_total(target_sum=2000), log1p
- 624 test perturbations + non-targeting controls
- `.obs.target_gene`: perturbation name or "non-targeting"
- `.var.index`: 6642 XCell gene symbols

### How to produce the prediction h5ad

After training completes, you need to:

1. Run inference on the test set to get predicted expression profiles
2. Map predictions to the 6,642 XCell gene set (same gene order as real file)
3. Include pseudobulked controls from HEPG2 (same controls as in real file)
4. Save as h5ad with `target_gene` column in `.obs`

The prediction file must:
- Use the **same 6,642 genes** in the **same order** as the real file
- Have the **same set of perturbation names** (624 test perts + non-targeting controls)
- Be in **log1p float** format (NOT integer counts)
- Include **3 extra perturbations** if they're missing from the model's vocabulary: `ALG1L`, `C14orf178`, `NEDD8-MDP1` — fill these with the global mean prediction profile

### XCell gene set

The 6,642 XCell benchmark genes (gene symbols, one per line) are at:
```
/rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/scMAE-main/virtual_cell/xcell_gene_set.txt
```

### Running cell-eval

cell-eval requires Python >= 3.11. Install in a separate environment:
```bash
conda create -n celleval python=3.11
conda activate celleval
pip install cell-eval
```

Then run:
```bash
cell-eval \
    --pred /path/to/txpert_xcell_predictions.h5ad \
    --real /rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/scMAE-main/virtual_cell/cell_eval_data/real_hepg2_xcell_test.h5ad \
    --outdir /path/to/cell-eval-results
```

This produces `agg_results.csv` with metrics including:
- `pearson_delta` — main metric for comparison
- `mse`, `mae`, `mse_delta`, `mae_delta`
- `de_spearman_sig`, `de_direction_match`
- `pr_auc`, `roc_auc`
- `discrimination_score_l1/l2/cosine`
- `overlap_at_N` (N=50,100,200,500)

## IhabNet (scMAE) comparison reference

The competing model (IhabNet virtual cell) uses:
- A pretrained single-cell masked autoencoder as encoder (44,728 genes, all genes not HVGs)
- Same adjacency map perturbation model (ported from this codebase)
- FiLM interaction (cell-state-dependent perturbation effect, not additive)
- Pretrained decoder (from separate decoder training phase)
- Latent-space delta loss (MSE between predicted and true encoder embeddings)
- Pseudobulked training data (averaged by batch)

### IhabNet best config (sweep winner, run 0r8pwdki)

```
pert_model_type=adjacency_map
graph_types=[go, string, pxmap-top1]       # NOTE: only 3 KGs (no txmap)
graph_edge_filter_mode=top_5               # Much sparser than txpert's top_20
pert_latent_dim=64
lr=4.0e-4, decoder_lr=2.8e-4
decoder_lr_warmup_epochs=10
latent_loss_weight=0.0067
interaction_type=film
loss_type=smooth_l1, smooth_l1_beta=2
cosine_loss_weight=0.2
adj_hidden_dim=128, adj_dropout=0.2, adj_deg=1
adj_normalize=none, adj_only_reverse=True, adj_make_undirected=True
batch_size=1024, epochs=100, weight_decay=0.001
```

### IhabNet XCell results (preliminary)

From the eval_xcell.py output on the best checkpoint:
- **pearson_delta_mean: 0.5459** (HEPG2 only, 621 test perturbations)
- mae_delta: 0.0149
- gene_mse: 0.0101

Note: IhabNet's 621 test perts vs XCell's 624 — 3 perts (ALG1L, C14orf178, NEDD8-MDP1) are not in IhabNet's perturbation vocabulary and get filled with mean profiles.

### IhabNet eval script

The IhabNet XCell evaluation script is at:
```
/rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/scMAE-main/eval_xcell.py
```

Usage:
```bash
cd /rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/scMAE-main
python eval_xcell.py \
    --vcell_checkpoint vcell_results/.../checkpoints/best_model.pth \
    --pseudobulk
```

## Key differences: txpert vs IhabNet

| Aspect | txpert | IhabNet |
|--------|--------|---------|
| Encoder | MLP on raw/HVG expression (5k genes) | Pretrained MAE on ALL genes (44k) |
| Gene set | HVG intersection (5k) | All genes, XCell mask for eval (6.6k) |
| Normalization | target_sum=4000 | target_sum=2000 |
| Perturbation model | Adj map (best) or GNN | Same adj map (ported from txpert) |
| Interaction | Additive (z + e_pert) | FiLM (cell-state-dependent) |
| Decoder | Single linear layer | Deep residual decoder (pretrained) |
| Loss | MSE | SmoothL1 + cosine + latent delta |
| Training data | Single-cell | Pseudobulked by batch |
| Primary metric | pearson_delta | pearson_delta |

## SLURM template

For running on the cluster:
```bash
#!/bin/bash
#SBATCH --job-name=txpert_xcell
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --partition=hopper
#SBATCH --gpus-per-node=1
#SBATCH --output=log/xcell_%j.out
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict

python main.py --config-name=config-x-cell-adj-map \
    datamodule.test_cell_type=hepg2 \
    datamodule.train_cell_types=all \
    wandb.name=xcell_best_adj_map_hepg2 \
    wandb.project=TxPert-Rebuttal \
    training.epochs=50
```

## Gotchas

- Cell type names are case-sensitive: `hepg2` (lowercase) in txpert, `HEPG2` (uppercase) in IhabNet
- txpert uses `target_sum=4000`, IhabNet uses `target_sum=2000` — prediction files must be in the correct normalization space for cell-eval
- Graph cache is shared at `/rxrx/data/valence/pathfinder/modules/gspp/graphs/` — don't delete
- Predefined splits in `data_xcell/` may differ from the canonical XCell splits in `xcell_splits.json` — verify before comparing
- HVG selection (5000 genes) means txpert evaluates on fewer genes than the 6,642 XCell set
- `obsm_key=raw` means using raw expression as input (not foundation model embeddings)
