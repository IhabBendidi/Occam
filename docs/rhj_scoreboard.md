# `rhj_fewshot` split-half — the scoreboard

**Objective.** Predict a perturbation's effect in a cellular context where that perturbation was never measured, from a single control cell.

**This table's question.** Across our loss/DCOS arms, the four competitor ports, the post-hoc emission treatments and the model-independent baselines — what does each actually score, on the same instrument, with no metric omitted?

**Varied:** the arm. **Held fixed:** the 611-perturbation cohort, the 6,983-gene axis, the seeded per-split half-B control draw, cell-eval 0.6.0 / pdex 0.1.24.

**Directions:** `pearson_delta` ↑ · `mae` **↓ (it is an ERROR)** · `overlap_at_N` ↑ · `discrimination_score_l1` ↑ (chance 0.5).

⚠️ **`n` is load-bearing.** A pdex shared-memory race (F-infra1) made cell-eval return nothing for some (arm, split, mode) triples while the SLURM task still exited 0. **Any row with n < 5 is PARTIAL**, marked ⚠️, and its mean may move.

⚠️ **Never compare across the two calibration tables.** Calibration costs 0.008–0.068 `pearson_delta` and quadruples `mae` (F-loss3); the uncalibrated table is the reportable one.

## Baselines — model-independent anchors

Read every model number against these, never against 0.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| ctrlcell | 5 | **-0.0018** ± 0.0031 | 0.0296 | 0.0000 | 0.5038 |  |
| ctrlmean | 5 | **0.0016** ± 0.0028 | 0.0231 | 0.0081 | 0.5014 |  |
| meanpert | 5 | **0.1324** ± 0.0030 | 0.0290 | 0.0108 | 0.5037 |  |
| transfer | 5 | **0.2323** ± 0.0021 | 0.0299 | 0.0351 | 0.6404 |  |
| experimental | 5 | **0.3722** ± 0.0010 | 0.0257 | 0.1942 | 0.7821 |  |

## Competitors — retrained on our data, our axis, our splits

⚠️ None of these reproduces the method's published figure; see `docs/rhj_competitors_results.md` for each arm's deliberate deviations.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| **TxPert** (single-cell controls) | 5 | **0.3463** ± 0.0035 | 0.0227 | 0.0582 | 0.6704 | headline |
| TxPert mean-controls | 5 | **0.3343** ± 0.0044 | 0.0231 | 0.0589 | 0.6584 | A/B diagnostic |
| TxPert epoch-20 | 5 | **0.3038** ± 0.0037 | 0.0231 | 0.0486 | 0.6450 | A/B diagnostic |
| ~~TxPert original~~ | 5 | **0.3077** ± 0.0032 | 0.0228 | 0.0522 | 0.6458 | ⛔ WITHDRAWN (C28) |
| **PerturbDiff** (genept) | 5 | **0.2449** ± 0.0038 | 0.0287 | 0.0734 | 0.6650 | headline |
| **STATE** | 5 | **0.2362** ± 0.0019 | 0.0291 | 0.0849 | 0.6639 | headline |
| **CellFlow** | 5 | **0.2154** ± 0.0039 | 0.0251 | 0.0315 | 0.5436 | headline |

## Our arms — UNCALIBRATED (`model_nocal`), the reportable table

Loss composition and delta-cosine sweep, seed 42.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `D03` smooth_l1_nbw0_nbs1.0_dcos0.5 | 5 | **0.3686** ± 0.0021 | 0.0321 | 0.0718 | 0.7263 |  |
| `D02` smooth_l1_nbw0_nbs1.0_dcos0.25 | 5 | **0.3661** ± 0.0023 | 0.0277 | 0.0568 | 0.7180 |  |
| `L10` smooth_l1_nbw0_nbs1.0_dcos0.1 | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 |  |
| `L11` smooth_l1_nbw0_nbs1.0_dcos0.25 | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 |  |
| `L12` smooth_l1_nbw0_nbs1.0_dcos0.5 | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 |  |
| `L13` smooth_l1_nbw0_nbs1.0_dcos1.0 | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 |  |
| `L02` nb_smooth_l1_nbw0.1_nbs1.0_dcos0.0 | 5 | **0.3184** ± 0.0033 | 0.0211 | 0.0621 | 0.6483 |  |
| `L08` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.0 | 5 | **0.3149** ± 0.0028 | 0.0211 | 0.0652 | 0.6437 |  |
| `L14` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.25 | 5 | **0.3149** ± 0.0028 | 0.0211 | 0.0652 | 0.6437 |  |
| `L07` nb_smooth_l1_nbw0.25_nbs6.5_dcos0.0 | 5 | **0.3094** ± 0.0025 | 0.0212 | 0.0609 | 0.6412 |  |
| `L03` nb_smooth_l1_nbw0.25_nbs1.0_dcos0.0 | 5 | **0.2989** ± 0.0029 | 0.0214 | 0.0663 | 0.6440 |  |
| `L09` nb_smooth_l1_nbw0.75_nbs6.5_dcos0.0 | 5 | **0.2979** ± 0.0022 | 0.0216 | 0.0629 | 0.6451 |  |
| `L05` nb_smooth_l1_nbw0.75_nbs1.0_dcos0.0 | 5 | **0.2790** ± 0.0027 | 0.0226 | 0.0539 | 0.6472 |  |
| `L06` nb_smooth_l1_nbw0.9_nbs1.0_dcos0.0 | 5 | **0.2774** ± 0.0027 | 0.0221 | 0.0660 | 0.6409 |  |
| `L01` nb_nbw0_nbs1.0_dcos0.0 | 5 | **0.2766** ± 0.0027 | 0.0220 | 0.0651 | 0.6425 |  |
| `L04` nb_smooth_l1_nbw0.5_nbs1.0_dcos0.0 | 5 | **0.2740** ± 0.0029 | 0.0219 | 0.0600 | 0.6357 |  |

## Schedule arms — epochs x `eta_min`

F-sched1/2/3. `epochs` is the strongest lever in the campaign; `eta_min` is noise inside 1e-7…1e-4 and a real gain at 3e-4. The campaign BEST lives here.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `Y06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e150_em3e-4 | 5 | **0.4043** ± 0.0019 | 0.0215 | 0.0946 | 0.7288 | extended schedule (F-sched3) |
| `Y03` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e120_em3e-4 | 5 | **0.4015** ± 0.0018 | 0.0217 | 0.0919 | 0.7247 | extended schedule (F-sched3) |
| `W22` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e90_em1e-6 | 5 | **0.3985** ± 0.0025 | 0.0215 | 0.0868 | 0.7156 | epochs x eta_min (F-sched2) |
| `Y05` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e150_em1e-4 | 5 | **0.3983** ± 0.0023 | 0.0217 | 0.0931 | 0.7223 | extended schedule (F-sched3) |
| `W21` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e90_em1e-7 | 5 | **0.3981** ± 0.0018 | 0.0216 | 0.0855 | 0.7183 | epochs x eta_min (F-sched2) |
| `Y01` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e120_em1e-5 | 5 | **0.3978** ± 0.0028 | 0.0217 | 0.0889 | 0.7141 | extended schedule (F-sched3) |
| `Y04` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e150_em1e-5 | 5 | **0.3963** ± 0.0021 | 0.0217 | 0.0913 | 0.7176 | extended schedule (F-sched3) |
| `Y02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e120_em1e-4 | 5 | **0.3960** ± 0.0021 | 0.0223 | 0.0904 | 0.7221 | extended schedule (F-sched3) |
| `W23` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e90_em1e-5 | 5 | **0.3959** ± 0.0031 | 0.0216 | 0.0887 | 0.7142 | epochs x eta_min (F-sched2) |
| `W24` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e90_em1e-4 | 5 | **0.3957** ± 0.0022 | 0.0221 | 0.0889 | 0.7190 | epochs x eta_min (F-sched2) |
| `W19` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e60_em1e-5 | 5 | **0.3933** ± 0.0027 | 0.0217 | 0.0817 | 0.7047 | epochs x eta_min (F-sched2) |
| `W18` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e60_em1e-6 | 5 | **0.3921** ± 0.0030 | 0.0216 | 0.0819 | 0.7089 | epochs x eta_min (F-sched2) |
| `W20` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e60_em1e-4 | 5 | **0.3921** ± 0.0034 | 0.0216 | 0.0792 | 0.7068 | epochs x eta_min (F-sched2) |
| `W17` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e60_em1e-7 | 5 | **0.3898** ± 0.0026 | 0.0216 | 0.0811 | 0.7040 | epochs x eta_min (F-sched2) |
| `W09` smooth_l1_nbw0_nbs1.0_dcos0.01_e90_em1e-7 | 5 | **0.3842** ± 0.0033 | 0.0209 | 0.0816 | 0.6852 | epochs x eta_min (F-sched2) |
| `W11` smooth_l1_nbw0_nbs1.0_dcos0.01_e90_em1e-5 | 5 | **0.3805** ± 0.0023 | 0.0210 | 0.0814 | 0.6852 | epochs x eta_min (F-sched2) |
| `W12` smooth_l1_nbw0_nbs1.0_dcos0.01_e90_em1e-4 | 5 | **0.3805** ± 0.0023 | 0.0208 | 0.0799 | 0.6836 | epochs x eta_min (F-sched2) |
| `W08` smooth_l1_nbw0_nbs1.0_dcos0.01_e60_em1e-4 | 4 | **0.3800** ± 0.0021 | 0.0208 | 0.0765 | 0.6814 | epochs x eta_min (F-sched2) ⚠️ PARTIAL |
| `W16` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e30_em1e-4 | 5 | **0.3794** ± 0.0026 | 0.0229 | 0.0795 | 0.7017 | epochs x eta_min (F-sched2) |
| `W13` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e30_em1e-7 | 5 | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 | epochs x eta_min (F-sched2) |
| `W06` smooth_l1_nbw0_nbs1.0_dcos0.01_e60_em1e-6 | 5 | **0.3777** ± 0.0020 | 0.0208 | 0.0745 | 0.6750 | epochs x eta_min (F-sched2) |
| `W10` smooth_l1_nbw0_nbs1.0_dcos0.01_e90_em1e-6 | 5 | **0.3773** ± 0.0022 | 0.0208 | 0.0812 | 0.6791 | epochs x eta_min (F-sched2) |
| `W14` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e30_em1e-6 | 5 | **0.3769** ± 0.0017 | 0.0227 | 0.0767 | 0.7015 | epochs x eta_min (F-sched2) |
| `W05` smooth_l1_nbw0_nbs1.0_dcos0.01_e60_em1e-7 | 4 | **0.3754** ± 0.0031 | 0.0208 | 0.0763 | 0.6739 | epochs x eta_min (F-sched2) ⚠️ PARTIAL |
| `W07` smooth_l1_nbw0_nbs1.0_dcos0.01_e60_em1e-5 | 5 | **0.3749** ± 0.0021 | 0.0210 | 0.0708 | 0.6800 | epochs x eta_min (F-sched2) |
| `W01` smooth_l1_nbw0_nbs1.0_dcos0.01_e30_em1e-7 | 4 | **0.3683** ± 0.0018 | 0.0212 | 0.0721 | 0.6715 | epochs x eta_min (F-sched2) ⚠️ PARTIAL |
| `W15` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e30_em1e-5 | 4 | **0.3679** ± 0.0025 | 0.0231 | 0.0702 | 0.6936 | epochs x eta_min (F-sched2) ⚠️ PARTIAL |
| `W02` smooth_l1_nbw0_nbs1.0_dcos0.01_e30_em1e-6 | 5 | **0.3650** ± 0.0031 | 0.0214 | 0.0711 | 0.6663 | epochs x eta_min (F-sched2) |
| `W03` smooth_l1_nbw0_nbs1.0_dcos0.01_e30_em1e-5 | 5 | **0.3644** ± 0.0012 | 0.0213 | 0.0725 | 0.6716 | epochs x eta_min (F-sched2) |
| `W04` smooth_l1_nbw0_nbs1.0_dcos0.01_e30_em1e-4 | 5 | **0.3622** ± 0.0024 | 0.0214 | 0.0694 | 0.6701 | epochs x eta_min (F-sched2) |

## Regulariser arms — VICReg and the dispersion terms

F-vicreg2/3, F-disp1. VICReg helps at 30 epochs and hurts at 60–90; the dispersion terms hold per-cell spread at parity (mechanism verified in training).

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `X02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.0487g0.55_vcov0.0081-delta_z_e90_em1e-5 | 5 | **0.3943** ± 0.0021 | 0.0220 | 0.0843 | 0.7189 | VICReg x schedule compounding (F-vicreg3) |
| `X01` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.0487g0.55_vcov0.0081-delta_z_e60_em1e-5 | 5 | **0.3920** ± 0.0034 | 0.0217 | 0.0793 | 0.7079 | VICReg x schedule compounding (F-vicreg3) |
| `V09` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.0487g0.55_vcov0.0081-delta_z | 5 | **0.3843** ± 0.0016 | 0.0225 | 0.0808 | 0.7065 | VICReg latent terms (F-vicreg2) |
| `V10` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.0487g0.55_vcov0.00081-z_hat | 5 | **0.3773** ± 0.0031 | 0.0228 | 0.0771 | 0.7113 | VICReg latent terms (F-vicreg2) |
| `V03` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vcov0.0081-delta_z | 5 | **0.3761** ± 0.0019 | 0.0228 | 0.0746 | 0.6979 | VICReg latent terms (F-vicreg2) |
| `V07` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.0131g1.0-delta_z | 5 | **0.3754** ± 0.0036 | 0.0228 | 0.0757 | 0.7050 | VICReg latent terms (F-vicreg2) |
| `V01` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vcov0.000081-delta_z | 5 | **0.3748** ± 0.0020 | 0.0226 | 0.0745 | 0.7042 | VICReg latent terms (F-vicreg2) |
| `V08` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.0487g0.55_vcov0.00081-delta_z | 5 | **0.3744** ± 0.0027 | 0.0229 | 0.0749 | 0.7075 | VICReg latent terms (F-vicreg2) |
| `V05` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.00487g0.55-delta_z | 5 | **0.3743** ± 0.0021 | 0.0226 | 0.0752 | 0.6986 | VICReg latent terms (F-vicreg2) |
| `V04` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vcov0.00081-z_hat | 5 | **0.3736** ± 0.0029 | 0.0227 | 0.0778 | 0.6984 | VICReg latent terms (F-vicreg2) |
| `V02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vcov0.00081-delta_z | 5 | **0.3728** ± 0.0017 | 0.0230 | 0.0777 | 0.7047 | VICReg latent terms (F-vicreg2) |
| `V06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_vvar0.0487g0.55-delta_z | 5 | **0.3720** ± 0.0024 | 0.0226 | 0.0758 | 0.7070 | VICReg latent terms (F-vicreg2) |
| `Z03` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_rspread0.512_e90_em1e-5 | 4 | **0.3677** ± 0.0036 | 0.0227 | 0.0717 | 0.6721 | dispersion terms (F-disp1) ⚠️ PARTIAL |
| `Z05` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_gvar0.473_e90_em1e-5 | 5 | **0.3673** ± 0.0017 | 0.0223 | 0.0582 | 0.6798 | dispersion terms (F-disp1) |
| `Z04` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_rspread2.56_e90_em1e-5 | 5 | **0.3557** ± 0.0021 | 0.0225 | 0.0695 | 0.6548 | dispersion terms (F-disp1) |
| `Z06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_gvar2.37_e90_em1e-5 | 4 | **0.3453** ± 0.0019 | 0.0242 | 0.0589 | 0.6652 | dispersion terms (F-disp1) ⚠️ PARTIAL |
| `Z01` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_dconc13.09_e90_em1e-5 | 5 | **0.2024** ± 0.0016 | 0.0318 | 0.0185 | 0.5989 | dispersion terms (F-disp1) |
| `Z02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_dconc65.47_e90_em1e-5 | 5 | **0.1868** ± 0.0019 | 0.0374 | 0.0173 | 0.6032 | dispersion terms (F-disp1) |

## Loss-composition arms — latent, scale-match, split reconstruction

F-latent4/5/6, F-loss7/8, F-recon3/4. These produced the three bases M03/M06/Q02.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `Q05` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lcos0.0059-delta_z | 5 | **0.3786** ± 0.0028 | 0.0226 | 0.0761 | 0.7078 | latent losses on M06 (F-latent4/6) |
| `Q02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044 | 5 | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 | latent losses on M06 (F-latent4/6) |
| `Q10` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.044_lcos0.0059-delta_z | 5 | **0.3772** ± 0.0017 | 0.0223 | 0.0768 | 0.6961 | latent losses on M06 (F-latent4/6) |
| `Q04` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lcos0.00059-delta_z | 4 | **0.3764** ± 0.0023 | 0.0223 | 0.0693 | 0.6921 | latent losses on M06 (F-latent4/6) ⚠️ PARTIAL |
| `Q09` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.044_lcos0.059-delta_z | 5 | **0.3755** ± 0.0015 | 0.0221 | 0.0683 | 0.6931 | latent losses on M06 (F-latent4/6) |
| `Q01` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.00044 | 5 | **0.3725** ± 0.0024 | 0.0227 | 0.0794 | 0.7008 | latent losses on M06 (F-latent4/6) |
| `Q03` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.044 | 4 | **0.3721** ± 0.0028 | 0.0229 | 0.0755 | 0.6968 | latent losses on M06 (F-latent4/6) ⚠️ PARTIAL |
| `Q08` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_lcos0.0059-delta_z | 5 | **0.3717** ± 0.0018 | 0.0227 | 0.0742 | 0.6992 | latent losses on M06 (F-latent4/6) |
| `Q11` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.00044_lcos0.00059-delta_z | 5 | **0.3708** ± 0.0024 | 0.0226 | 0.0697 | 0.7016 | latent losses on M06 (F-latent4/6) |
| `Q06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lcos0.059-delta_z | 4 | **0.3690** ± 0.0023 | 0.0230 | 0.0720 | 0.7022 | latent losses on M06 (F-latent4/6) ⚠️ PARTIAL |
| `M06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363 | 5 | **0.3681** ± 0.0023 | 0.0232 | 0.0703 | 0.7047 | dcos/delta_mag scale-match (F-loss7/8) |
| `Q07` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lcos0.048-z_hat | 4 | **0.3681** ± 0.0020 | 0.0225 | 0.0663 | 0.6890 | latent losses on M06 (F-latent4/6) ⚠️ PARTIAL |
| `M03` smooth_l1_nbw0_nbs1.0_dcos0.01 | 5 | **0.3679** ± 0.0017 | 0.0212 | 0.0722 | 0.6715 | dcos/delta_mag scale-match (F-loss7/8) |
| `M04` smooth_l1_nbw0_nbs1.0_dcos0.02 | 5 | **0.3642** ± 0.0026 | 0.0217 | 0.0703 | 0.6704 | dcos/delta_mag scale-match (F-loss7/8) |
| `G23` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitdelta_only-group_mean-1.0-3.0 | 4 | **0.3628** ± 0.0019 | 0.0238 | 0.0734 | 0.7108 | split reconstruction (F-recon3/4) ⚠️ PARTIAL |
| `M05` smooth_l1_nbw0_nbs1.0_dcos0.05 | 5 | **0.3624** ± 0.0021 | 0.0228 | 0.0693 | 0.6921 | dcos/delta_mag scale-match (F-loss7/8) |
| `T02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.000044_lcos0.000059-delta_z | 5 | **0.3617** ± 0.0023 | 0.0235 | 0.0714 | 0.7026 | latent losses at 0.01x (F-latent5) |
| `G10` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitdelta_only-group_mean-1.0-1.0 | 5 | **0.3607** ± 0.0026 | 0.0247 | 0.0691 | 0.7138 | split reconstruction (F-recon3/4) |
| `T01` smooth_l1_nbw0_nbs1.0_dcos0.01_lmse0.000157_lcos0.000097-delta_z | 5 | **0.3595** ± 0.0027 | 0.0214 | 0.0700 | 0.6653 | latent losses at 0.01x (F-latent5) |
| `G05` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitdelta_only-group_mean-1.0-1.0 | 5 | **0.3582** ± 0.0025 | 0.0227 | 0.0711 | 0.6979 | split reconstruction (F-recon3/4) |
| `G19` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitcmean-cell-1.0-3.0 | 5 | **0.3576** ± 0.0022 | 0.0239 | 0.0661 | 0.7074 | split reconstruction (F-recon3/4) |
| `G03` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitcmean-cell-1.0-1.0 | 5 | **0.3575** ± 0.0023 | 0.0224 | 0.0663 | 0.6871 | split reconstruction (F-recon3/4) |
| `G21` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitdelta_only-group_mean-1.0-3.0 | 5 | **0.3562** ± 0.0021 | 0.0220 | 0.0699 | 0.6830 | split reconstruction (F-recon3/4) |
| `M02` smooth_l1_nbw0_nbs1.0_dcos0.005 | 5 | **0.3553** ± 0.0030 | 0.0212 | 0.0598 | 0.6624 | dcos/delta_mag scale-match (F-loss7/8) |
| `G04` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitcmean-group_mean-1.0-1.0 | 5 | **0.3540** ± 0.0022 | 0.0225 | 0.0689 | 0.6931 | split reconstruction (F-recon3/4) |
| `G26` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitraw-cell-1.562-0.735 | 5 | **0.3532** ± 0.0025 | 0.0244 | 0.0654 | 0.7057 | split reconstruction (F-recon3/4) |
| `G22` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitdelta_only-group_mean-1.0-10.0 | 5 | **0.3529** ± 0.0018 | 0.0215 | 0.0692 | 0.6725 | split reconstruction (F-recon3/4) |
| `G18` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitcmean-cell-1.0-10.0 | 5 | **0.3529** ± 0.0026 | 0.0216 | 0.0678 | 0.6694 | split reconstruction (F-recon3/4) |
| `G17` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitcmean-cell-1.0-3.0 | 5 | **0.3522** ± 0.0025 | 0.0219 | 0.0648 | 0.6789 | split reconstruction (F-recon3/4) |
| `G09` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitcmean-group_mean-1.0-1.0 | 5 | **0.3512** ± 0.0027 | 0.0249 | 0.0600 | 0.7038 | split reconstruction (F-recon3/4) |
| `G08` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitcmean-cell-1.0-1.0 | 5 | **0.3511** ± 0.0027 | 0.0247 | 0.0636 | 0.7127 | split reconstruction (F-recon3/4) |
| `G06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitraw-cell-1.0-1.0 | 5 | **0.3499** ± 0.0025 | 0.0242 | 0.0653 | 0.7129 | split reconstruction (F-recon3/4) |
| `G24` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitdelta_only-group_mean-1.0-10.0 | 5 | **0.3481** ± 0.0027 | 0.0243 | 0.0563 | 0.6841 | split reconstruction (F-recon3/4) |
| `M01` smooth_l1_nbw0_nbs1.0_dcos0.002 | 5 | **0.3457** ± 0.0022 | 0.0211 | 0.0658 | 0.6522 | dcos/delta_mag scale-match (F-loss7/8) |
| `G07` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitraw-group_mean-1.0-1.0 | 4 | **0.3456** ± 0.0019 | 0.0237 | 0.0634 | 0.7020 | split reconstruction (F-recon3/4) ⚠️ PARTIAL |
| `G01` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitraw-cell-1.0-1.0 | 5 | **0.3424** ± 0.0033 | 0.0248 | 0.0632 | 0.6932 | split reconstruction (F-recon3/4) |
| `G14` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitraw-cell-1.0-3.0 | 5 | **0.3423** ± 0.0022 | 0.0242 | 0.0587 | 0.7010 | split reconstruction (F-recon3/4) |
| `G20` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitcmean-cell-1.0-10.0 | 4 | **0.3415** ± 0.0024 | 0.0242 | 0.0604 | 0.7037 | split reconstruction (F-recon3/4) ⚠️ PARTIAL |
| `G16` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitraw-cell-1.0-30.0 | 5 | **0.3399** ± 0.0025 | 0.0252 | 0.0555 | 0.7072 | split reconstruction (F-recon3/4) |
| `G25` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitraw-cell-1.562-0.735 | 5 | **0.3362** ± 0.0032 | 0.0244 | 0.0595 | 0.6760 | split reconstruction (F-recon3/4) |
| `G02` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitraw-group_mean-1.0-1.0 | 5 | **0.3282** ± 0.0024 | 0.0240 | 0.0590 | 0.6922 | split reconstruction (F-recon3/4) |
| `G15` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_rsplitraw-cell-1.0-10.0 | 5 | **0.3248** ± 0.0029 | 0.0260 | 0.0559 | 0.7006 | split reconstruction (F-recon3/4) |
| `G11` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitraw-cell-1.0-3.0 | 5 | **0.3231** ± 0.0030 | 0.0248 | 0.0555 | 0.6778 | split reconstruction (F-recon3/4) |
| `G12` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitraw-cell-1.0-10.0 | 5 | **0.2930** ± 0.0035 | 0.0304 | 0.0577 | 0.6109 | split reconstruction (F-recon3/4) |
| `G13` smooth_l1_nbw0_nbs1.0_dcos0.01_rsplitraw-cell-1.0-30.0 | 5 | **0.2699** ± 0.0036 | 0.0371 | 0.0501 | 0.5740 | split reconstruction (F-recon3/4) |

## Our arms — CALIBRATED (`model_cal`), same checkpoints

Shown because it was measured, NOT because it should be used.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `D02` smooth_l1_nbw0_nbs1.0_dcos0.25 | 5 | **0.2993** ± 0.0011 | 0.1018 | 0.0290 | 0.8103 |  |
| `D03` smooth_l1_nbw0_nbs1.0_dcos0.5 | 5 | **0.2980** ± 0.0013 | 0.0929 | 0.0217 | 0.7759 |  |
| `L02` nb_smooth_l1_nbw0.1_nbs1.0_dcos0.0 | 5 | **0.2789** ± 0.0024 | 0.0917 | 0.0240 | 0.7575 |  |
| `L08` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.0 | 5 | **0.2771** ± 0.0012 | 0.0868 | 0.0229 | 0.7554 |  |
| `L14` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.25 | 5 | **0.2771** ± 0.0012 | 0.0868 | 0.0229 | 0.7554 |  |
| `L10` smooth_l1_nbw0_nbs1.0_dcos0.1 | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 |  |
| `L11` smooth_l1_nbw0_nbs1.0_dcos0.25 | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 |  |
| `L12` smooth_l1_nbw0_nbs1.0_dcos0.5 | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 |  |
| `L13` smooth_l1_nbw0_nbs1.0_dcos1.0 | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 |  |
| `L09` nb_smooth_l1_nbw0.75_nbs6.5_dcos0.0 | 5 | **0.2709** ± 0.0030 | 0.0896 | 0.0244 | 0.7628 |  |
| `L06` nb_smooth_l1_nbw0.9_nbs1.0_dcos0.0 | 5 | **0.2690** ± 0.0017 | 0.0912 | 0.0275 | 0.7602 |  |
| `L01` nb_nbw0_nbs1.0_dcos0.0 | 5 | **0.2625** ± 0.0014 | 0.0901 | 0.0245 | 0.7711 |  |
| `L03` nb_smooth_l1_nbw0.25_nbs1.0_dcos0.0 | 5 | **0.2612** ± 0.0021 | 0.0624 | 0.0208 | 0.7728 |  |
| `L07` nb_smooth_l1_nbw0.25_nbs6.5_dcos0.0 | 5 | **0.2612** ± 0.0020 | 0.0867 | 0.0322 | 0.7612 |  |
| `L05` nb_smooth_l1_nbw0.75_nbs1.0_dcos0.0 | 5 | **0.2585** ± 0.0017 | 0.0919 | 0.0284 | 0.7703 |  |
| `L04` nb_smooth_l1_nbw0.5_nbs1.0_dcos0.0 | 5 | **0.2372** ± 0.0043 | 0.0842 | 0.0262 | 0.7754 |  |

## Post-hoc emission treatments on `s313`

⚠️ split 00 only for the NB rows (n=1).

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| none (`model_nocal`) | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 | the default |
| calibration only (`model_cal`) | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 | control-fit quantile |
| NB emission + `exact_colsum` | 1 | **0.2027** | 0.0276 | 0.0961 | 0.6393 | fair theta, lib 17,500 ⚠️ PARTIAL |
| NB emission alone | 1 | **0.1775** | 0.0331 | 0.0614 | 0.6475 | fair theta, lib 17,500 ⚠️ PARTIAL |
| NB emission + `exact_colsum` | 1 | **0.2478** | 0.0295 | 0.0314 | 0.6511 | ⛔ BROKEN theta/lib (F-emit4) ⚠️ PARTIAL |
| NB emission alone | 1 | **0.1611** | 0.0490 | 0.0079 | 0.6700 | ⛔ BROKEN theta/lib (F-emit4) ⚠️ PARTIAL |

