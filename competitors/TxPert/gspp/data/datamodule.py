from typing import Literal, Union, List, Tuple, Dict
import os
import sys
from sklearn.model_selection import KFold
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import lightning as L
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import joblib
import multiprocessing as mp
from scipy.sparse import csr_matrix, issparse, hstack
from loguru import logger

# TODO, refactor to use cs after merge
import gspp.constants as cs

from gspp.constants import (
    PUBLIC_CELL_TYPES,
    PUBLIC_CELL_TYPE_DATASETS,
    CONTROL,
    CELL_TYPE,
    CONDITION_NAME,
    CONDITION,
    DOSE,
    CONTROL_LABEL,
    BATCH,
    BATCH_RAW,
    GENE_NAME,
    IS_CONTROL,
    MOL_FEATURIZER_DICT,
)

from gspp.data.data_utils import (
    collate_fn,
    calculate_de_for_eval,
    normalize_and_log_transform,
    _standardize_obs_format,
)

logger.remove()
logger.add(sys.stdout, level="INFO")


class PertDataModule(L.LightningDataModule):
    """
    Datamodule for the perturbation prediction task. This datamodule loads the perturbation data and control data.

    Args:
        batch_size (int): Batch size for the dataloader
        match_cntr (bool): Whether to match the control based on the batch
        data_dir (str): Directory where the raw data is stored
        cache_dir (str): Directory where the processed data is stored
        xtraining_strategy (str): Default to "train_together_implicit" for single cell type
        train_cell_types (List or str): List of cell types to train on
        test_cell_type (str): Cell type to test on
        combine_strategy (str): How to combine datasets
        gene_prior (List): List of genes to include in the dataset (relevant when added data does not fully intersect with public data)
        hvg_number (int): Number of highly variable genes to use
        target_sum (int): Target sum for normalization
        min_cells (int): Minimum number of cells for a perturbation to be included
        min_de_genes (int): Minimum number of DE genes for a perturbation to be included
        knockdown_threshold (float): Threshold for knockdown
        use_predefined_splits (bool): Whether to use predefined splits (taken from GEARS)
        num_additional_cell_types (int): Number of additional cell types added to training data
        auxiliary_dir (str): Directory where auxiliary data is stored
        embed_cntr (bool): Whether to embed control samples
        test_split (int): Test split index for k-fold cross
        use_cache (bool): Whether to use cache
        debug (bool): Whether to run in debug mode
        n_jobs_preprocessing (int): Number of processes when preprocessing
    """

    def __init__(
        self,
        batch_size: int = 64,
        match_cntr: bool = False,
        avg_cntr: bool = False,
        data_dir: str = None,
        cache_dir: str = str(Path(__file__).resolve().parents[2] / "cache"),
        auxiliary_dir: str = None,
        xtraining_strategy: str = "train_together_explicit",
        train_cell_types: Union[str, List[str]] = "all",
        val_cell_type: Union[str, None] = None,
        test_cell_type: Union[str, None] = None,
        combine_strategy: str = "intersect",
        gene_prior: List[str] = None,
        hvg_number: int = 5000,
        target_sum: int = 4000,
        min_cells: int = 25,
        min_de_genes: int = 50,
        knockdown_threshold: float = -0.3,
        use_predefined_splits: bool = True,
        num_additional_cell_types: int = 0,
        embed_cntr: bool = True,
        fp_types: List[str] = ["ecfp"],
        test_split: int = 0,
        use_cache: bool = True,
        debug: bool = False,
        n_jobs_preprocessing: int = 1,
        obsm_key: str = cs.ObsmKey.SCGPT,
        xcell: bool = False,
        xcell_gene_set_path: str = None,
        xcell_splits_path: str = None,
        xcell_raw_hepg2_path: str = None,
        xcell_v2: bool = False,
        xcell_v2_data_path: str = None,
        zarr: bool = False,
        zarr_manifest_path: str = None,
    ):
        """
        Initialize the PertDataModule with various parameters.
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        # Upstream always had a GEARS-style `data_dir` of per-cell-type h5ad folders. The
        # zarr benchmark has no such root, so it is optional now.
        self.data_path = Path(data_dir) if data_dir is not None else None
        self.cache_dir = cache_dir
        self.xtraining_strategy = xtraining_strategy
        self.train_cell_types = train_cell_types
        self.val_cell_type = val_cell_type
        self.test_cell_type = test_cell_type
        # Molecular perturbations are out of scope for this arm: the benchmark is CRISPRi-only.
        # Upstream built SMILES featurizers here and eagerly torch.load'ed a molgps fingerprint
        # pickle from another user's `outgoing/`; both are gone.
        self.molecular_perts = False
        self.smiles_featurizer = None

        self.combine_strategy = combine_strategy
        self.gene_prior = gene_prior
        self.hvg_number = hvg_number
        self.target_sum = target_sum
        self.min_cells = min_cells
        self.min_de_genes = min_de_genes
        self.knockdown_threshold = knockdown_threshold
        self.use_predefined_splits = use_predefined_splits
        self.num_additional_cell_types = num_additional_cell_types
        self.auxiliary_dir = auxiliary_dir
        self.embed_cntr = embed_cntr
        self.test_split = test_split
        self.use_cache = use_cache
        self.debug = debug
        self.obsm_key = obsm_key
        self.n_jobs_preprocessing = n_jobs_preprocessing
        self.match_cntr = match_cntr
        self.avg_cntr = avg_cntr
        self.mol_fp_dim = None

        self.xcell = xcell
        self.xcell_gene_set_path = xcell_gene_set_path
        self.xcell_splits_path = xcell_splits_path
        self.xcell_raw_hepg2_path = xcell_raw_hepg2_path
        self.xcell_genes = None
        if self.xcell and self.xcell_gene_set_path:
            with open(self.xcell_gene_set_path) as f:
                self.xcell_genes = [line.strip() for line in f if line.strip()]
            logger.info(f"XCell mode: loaded {len(self.xcell_genes)} genes from {self.xcell_gene_set_path}")

        self.xcell_v2 = xcell_v2
        self.xcell_v2_data_path = xcell_v2_data_path
        self._xcell_v2_hepg2_exclude = set()

        # Canonical hooke-tx zarr source (see gspp/data/zarr_loader.py). Shares the xcell_v2
        # code paths downstream -- no val split, HepG2 perturbation holdout -- but reads the
        # zarr datasets STATE and PerturbDiff use and keeps their normalization.
        self.zarr = zarr
        self.zarr_manifest_path = zarr_manifest_path
        self._zarr_manifest = None
        self._zarr_split = None
        if self.zarr:
            if not self.zarr_manifest_path:
                raise ValueError("datamodule.zarr=true requires datamodule.zarr_manifest_path")
            self.xcell_v2 = True  # reuse the no-val / HepG2-holdout wiring

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_prepared = False
        self._initialize_gene_sets()
        self._validate_and_set_cell_types()

    def _validate_and_set_cell_types(self) -> None:
        """
        Validate and set the provided cell types against the public cell types.
        """
        if self.test_cell_type not in PUBLIC_CELL_TYPES:
            raise ValueError(
                f"test_cell_type must be one of {PUBLIC_CELL_TYPES}, got {self.test_cell_type}"
            )

        if isinstance(self.train_cell_types, str) and self.train_cell_types != "all":
            self.train_cell_types = [self.train_cell_types]

        if not (
            self.train_cell_types == "all"
            or all(ct in PUBLIC_CELL_TYPES for ct in self.train_cell_types)
        ):
            raise ValueError(
                f"train_cell_types must be 'all' or a subset of {PUBLIC_CELL_TYPES}, got {self.train_cell_types}"
            )

        temp_train_cell_types = set(PUBLIC_CELL_TYPES)
        temp_train_cell_types.remove(self.test_cell_type)
        if self.val_cell_type is not None:
            temp_train_cell_types.remove(self.val_cell_type)
        
        self.train_cell_types = (
            [ct for ct in temp_train_cell_types if not ct.startswith(("mol-", "doubles-")) and ct not in ["K562_adamson", "K562_dixit"]]
            if self.train_cell_types == "all"
            else self.train_cell_types
        )
        self.train_cell_types.sort(reverse=True)

        self.total_cell_types = set(self.train_cell_types)
        self.total_cell_types.add(self.test_cell_type)

    def _initialize_gene_sets(self) -> None:
        """
        Initialize gene sets for perturbation data.
        """
        id2gene = dict(zip((_df := pd.read_csv(cs.DATA_DIR / "master_gene_set_sorted.csv"))['gene_id'], _df['gene_name']))

        id2pert = pd.read_csv(
            cs.DATA_DIR / "gears_gene_set.csv",
            index_col=0,
        ).to_dict()["0"]

        if self.embed_cntr:
            id2pert.update({-1: CONTROL_LABEL})

        self.id2pert = {k: v for k, v in id2pert.items()}
        self.pert2id = {v: k for k, v in id2pert.items()}

        self.id2gene = {k: v for k, v in id2gene.items()}
        self.gene2id = {v: k for k, v in id2gene.items()}
        
    def get_dataset_identifier(self):
        """
        Returns a consistent string identifier for the dataset configuration.
        This can be used for cache paths, checkpoint directories, and other places
        where the dataset configuration needs to be represented as a string.
        """
        test_val = f"{self.test_cell_type}_{self.val_cell_type}" if self.val_cell_type is not None else self.test_cell_type
        xcell_tag = "_xcell_v2" if self.xcell_v2 else ("_xcell" if self.xcell else "")
        cache_path = f'{test_val}_{self.xtraining_strategy}_{self.combine_strategy}_{self.hvg_number}_{self.target_sum}_{"-".join(self.train_cell_types)}_prior-gene-{self.gene_prior != None}_{self.obsm_key}{xcell_tag}'

        return cache_path
    
    def prepare_data(self) -> None:
        self.cache_path = f'{self.cache_dir}/{self.get_dataset_identifier()}'

        if self.data_prepared:
            return

        cache_file = f"{self.cache_path}/combined.h5ad"

        if (
            os.path.exists(cache_file)
            and not self.debug
            and self.use_cache
        ):
            logger.info(f"adata cached, loading from {cache_file}")
            self.adata = sc.read_h5ad(cache_file)
            
        else:
            logger.info(f"adata not cached, loading and processing data")
            self.adata = self._combine_datasets()

            embedding = None
            if self.obsm_key != cs.ObsmKey.RAW:
                embedding = self.adata.obsm[self.obsm_key]
                # TODO, confirm directly setting adata.var here does not break anything
                # self.adata.var = pd.DataFrame(index=[f"feature_{i}" for i in range(embedding.shape[1])])
            
            one_hot_headers, one_hot_encoders = self._one_hot_encode()
            self._add_one_hot_FM_embeddings(one_hot_headers, one_hot_encoders, embedding)
            
            self._save_cache(cache_file)

        self._split_data()

        # Featurize molecules if needed
        if self.molecular_perts:
            self._featurize_smiles()

        self.data_prepared = True

    def _one_hot_encode(self) -> Union[None, Tuple[np.ndarray, pd.Index]]:
        """
        One hot encode the cell types.
        """
        one_hot_headers, one_hot_encoders = [], []

        if self.xtraining_strategy == "train_together_explicit":
            ct_one_hot = pd.get_dummies(self.adata.obs[CELL_TYPE], dtype=int)
            for i in range(self.num_additional_cell_types):
                ct_one_hot[f"additional_category_{i}"] = 0
            one_hot_headers.append(ct_one_hot.columns)
            one_hot_encoders.append(ct_one_hot.values)
            logger.info("One hot encoding the cell types")

        return one_hot_headers, one_hot_encoders

    def _add_one_hot_FM_embeddings(
        self, one_hot_headers: List[pd.Index], one_hot_encoders: List[np.ndarray], embedding: np.ndarray
    ) -> None:
        """
        Append the one hot encoded cell types to the AnnData object.
        """
        if self.xtraining_strategy == "train_together_explicit":
            self.adata = sc.AnnData(
                X=hstack([self.adata.X] + one_hot_encoders).tocsr(),
                obs=self.adata.obs,
                var=pd.concat([self.adata.var] + [pd.DataFrame(index=header) for header in one_hot_headers]),
                uns=self.adata.uns,
            )
        if self.obsm_key != cs.ObsmKey.RAW:  # TODO, refactor so handling of obsm and var for embeddings are in one spot
            assert embedding is not None, "if this fails, there's a code consistency problem"
            self.adata.obsm[self.obsm_key] = np.hstack([embedding] + one_hot_encoders)

    def _save_cache(self, cache_file: str) -> None:
        """
        Save the processed data to cache.

        Args:
            cache_file (str): Path to the cache file.
        """
        if not self.debug and not self.xcell_v2:
            os.makedirs(self.cache_path, exist_ok=True)
            if os.path.exists(cache_file):
                os.remove(cache_file)
            self.adata.write(cache_file)
            logger.info(f"adata saved to {cache_file}")

    def _split_data(self) -> None:
        """
        Split the data into control and treatment datasets.
        """
        self._split_control_data()
        self._process_treatment_data()
        self._set_input_output_dimensions()
        self._load_pharos_data()

    def _featurize_smiles(self) -> None:
        """
        Featurize molecules.
        """
        self.smiles2vec = {}
        for perts in self.conditions:
            assert perts[-1] == "ctrl", "Last entry must be `ctrl`"
            for smiles in perts[:-1]:
                if smiles not in self.smiles2vec.keys() and len(smiles) > 0:
                    self.smiles2vec[smiles] = {
                        key: torch.from_numpy(featurizer(smiles)).float()
                        for key, featurizer in self.smiles_featurizer.items()
                    }

        self.mol_fp_dim = {
            key: val.shape for key, val in next(iter(self.smiles2vec.values())).items()
        }

    def _smiles_to_molgps_fp(self, smiles: str, molgps_id: str) -> torch.Tensor:
        """
        Featurize molecules using molgps.
        """
        smiles_idx = self.molgps_fps["smiles"]["test"].index(smiles)
        return self.molgps_fps["fps"]["test"][molgps_id][smiles_idx,:].numpy()

    def _split_control_data(self) -> None:
        """
        Split the data into control and treatment datasets.
        """
        # TODO: Needs future fixing
        if self.molecular_perts:
            train_control_mask = (self.adata.obs[CONTROL] == 1)

            self.control_data_train = self.adata[train_control_mask]
            self.control_data_val = self.control_data_train
            self.control_data_test = self.adata[train_control_mask]     # TODO: Anything wrong with having the same controls here?
        else:
            train_control_mask = (self.adata.obs[CONTROL] == 1) & (
                self.adata.obs[CELL_TYPE].isin(
                    self.train_cell_types
                    if self.xtraining_strategy == "fine_tuning"
                    else self.train_cell_types + [self.test_cell_type] + [self.val_cell_type]
                )
            )

            self.control_data_train = self.adata[train_control_mask]

            if (self.xcell or self.xcell_v2) and self.val_cell_type is None:
                self.control_data_val = self.adata[
                    (self.adata.obs[CONTROL] == 1)
                    & (self.adata.obs[CELL_TYPE] == self.test_cell_type)
                ]
            elif self.val_cell_type is not None:
                self.control_data_val = self.adata[
                    (self.adata.obs[CONTROL] == 1)
                    & (self.adata.obs[CELL_TYPE] == self.val_cell_type)
                ]
            else:
                self.control_data_val = self.control_data_train

            self.control_data_test = self.adata[
                (self.adata.obs[CONTROL] == 1)
                & (self.adata.obs[CELL_TYPE] == self.test_cell_type)    # TODO: Split by cell type only for genetic perts atm (see above)
            ]
        
        self.control_data = self.control_data_test

        self.control_train_condition_names = self.control_data_train.obs[CONDITION_NAME]
        self.control_val_condition_names = None if self.val_cell_type is None else self.control_data_val.obs[CONDITION_NAME]
        self.control_test_condition_names = self.control_data_test.obs[CONDITION_NAME]

    def _process_treatment_data(self) -> None:
        """
        Process the treatment data by filtering and setting conditions.
        """
        self.treatment_data = self.adata[self.adata.obs[CONTROL] == 0]

        # `.astype(str)` is required, not cosmetic: CONDITION is a categorical, and under
        # pandas >=3 `categorical.str.split("+")` returns the *repr* of each list
        # ("['ABCB7', 'ctrl']") rather than the list. Iterating that yields characters, so the
        # pert2id membership check below rejected every row and left treatment_data empty.
        temp_conditions = self.treatment_data.obs[CONDITION].astype(str).str.split("+").tolist()
        if(self.molecular_perts):
            # fix for SMILES strings
            temp_conditions = [["+".join(c[:-1]), c[-1]] for c in temp_conditions]
        
        if self.molecular_perts:
            self.condition_names = self.treatment_data.obs[CONDITION_NAME]
            self.conditions = pd.Series([p for p in temp_conditions])
            self.conditions.index = self.treatment_data.obs.index
        else:
            pert_idxs = [
                [self.pert2id[p] for p in item if p in self.pert2id]
                for item in temp_conditions
            ]

            filtered_idx = [
                all(p in self.pert2id for p in item) for item in temp_conditions
            ]

            self.condition_names = self.treatment_data.obs[CONDITION_NAME][filtered_idx]
            filtered_conditions = self.treatment_data.obs[CONDITION][
                ~np.array(filtered_idx)
            ].unique()
            self.treatment_data = self.treatment_data[filtered_idx]
            self.conditions = pd.Series(pert_idxs)[filtered_idx]
            self.conditions.index = self.treatment_data.obs.index

            logger.info(f"Filtered out the following perturbations: {filtered_conditions}")


    def _set_input_output_dimensions(self) -> None:
        """
        Set the input and output dimensions for the data.
        """
        # used to set dimensions (with our without cell type onehot encoding)
        self.cell_types = np.unique(self.adata.obs[CELL_TYPE])
        self.number_of_cell_types = len(self.cell_types)

        dim_adjustment = self.number_of_cell_types if "train_together_explicit" == self.xtraining_strategy else 0

        self.unique_doses = self.adata.obs[DOSE].unique().tolist() if self.molecular_perts else None
        self.num_doses = len(self.unique_doses) if self.molecular_perts else None
        self.adata_output_dim = self.adata.X.shape[1] - dim_adjustment
        
        if self.obsm_key != cs.ObsmKey.RAW:
            self.input_dim = self.adata.obsm[self.obsm_key].shape[1]
            self.output_dim = self.adata.obsm[self.obsm_key].shape[1] - dim_adjustment
        else :
            self.input_dim = self.adata.X.shape[1]
            self.output_dim = self.adata.X.shape[1] - dim_adjustment
        
        logger.info(f"adata_output_dim :  {self.adata_output_dim}")

        self.adata = self.adata[:, : self.adata_output_dim].to_memory()

    def _load_pharos_data(self) -> None:
        """
        Load Pharos data and update the AnnData object.

        Optional: ``uns["pharos"]`` is only read by the ``pearson_delta_kl*`` breakdowns, which
        are registered ``slow=True`` and therefore computed only under ``slow_benchmark: true``.
        With no ``auxiliary_dir`` configured (the zarr benchmark sets none) this is a no-op.
        """
        if not self.auxiliary_dir:
            logger.info("auxiliary_dir not set: skipping Pharos knowledge levels")
            return

        dat = pd.read_csv(Path(self.auxiliary_dir) / "pharos_mean_rank.csv")
        dat = dat.sort_values("mean_rank")
        thresholds = [dat.shape[0] // x for x in [8, 4, 2, 1]]
        res = {}
        for i, gene in enumerate(dat["Symbol"]):
            for j in range(len(thresholds) - 1, -1, -1):
                if thresholds[j] >= i:
                    res[gene] = j

        knowledge_levels = np.array([res.get(v, -1) for v in self.adata.var_names])
        self.adata.uns["pharos"] = knowledge_levels
        logger.info(
            "knowledge_level_counts:",
            pd.Series(self.adata.uns["pharos"]).value_counts(),
        )

    def _load_xcellv2_data(self) -> sc.AnnData:
        """
        Load the merged replogle_nadig.h5ad for XCellv2 mode.
        Subsets to 2,000 HVG genes, renames columns, normalizes.
        """
        logger.info(f"XCellv2: loading merged data from {self.xcell_v2_data_path}")
        adata = sc.read_h5ad(self.xcell_v2_data_path)

        hvg_mask = adata.var['highly_variable'].values
        adata = adata[:, hvg_mask].copy()
        logger.info(f"XCellv2: subset to {adata.shape[1]} HVG genes")

        adata.obs[GENE_NAME] = adata.obs['gene'].astype(str)
        adata.obs[BATCH_RAW] = adata.obs['gem_group'].astype(str)

        cl_map = {'k562': 'K562', 'rpe1': 'RPE1'}
        adata.obs[CELL_TYPE] = adata.obs['cell_line'].map(lambda x: cl_map.get(x, x))

        adata.obs.loc[adata.obs['is_control'].astype(bool), GENE_NAME] = 'non-targeting'
        adata.obs[IS_CONTROL] = adata.obs['is_control'].astype(bool)

        adata.var[GENE_NAME] = adata.var.index.values
        adata.var_names_make_unique()

        adata.obs.index = [str(i) for i in range(len(adata))]
        adata.obs.index.name = None

        adata = normalize_and_log_transform(adata, self.target_sum)
        adata = _standardize_obs_format(adata, GENE_NAME)

        logger.info(f"XCellv2: loaded {adata.shape[0]} cells, {adata.shape[1]} genes, "
                     f"cell lines: {sorted(adata.obs[CELL_TYPE].unique())}")
        return adata

    def _extend_pert_vocabulary(self, adata: sc.AnnData) -> None:
        """Add benchmark perturbations that are missing from ``data/gears_gene_set.csv``.

        ``pert2id`` is a *fixed* 9,853-symbol vocabulary shipped with upstream TxPert, not
        something derived from the data. On this benchmark 33 of the 2,023 perturbed genes are
        absent from it -- mostly HGNC renames the GEARS list predates (HARS1<-HARS, LARS1<-LARS,
        MARS1<-MARS, RARS1<-RARS, WARS1<-WARS, H2AZ1<-H2AFZ, ...) -- and **9 of those are in the
        380-perturbation HepG2 test set**. ``_process_treatment_data`` drops cells whose
        perturbation is outside the vocabulary, so leaving this alone would score TxPert on 371
        perturbations while STATE / PerturbDiff / hooke-tx score 380.

        Appending them keeps the scored set identical across arms. The new nodes enlarge the
        adjacency matrix by ~0.3%; they pick up knowledge-graph edges wherever the current
        symbol is known (STRING carries all 33) and are isolated rows where it is not.
        """
        observed = set(adata.obs.loc[~adata.obs[IS_CONTROL].astype(bool), GENE_NAME].unique())
        # Multi-gene perturbations are encoded "A+B"; the vocabulary is per single gene.
        singles = {s for name in observed for s in str(name).split("+")}

        missing = sorted(singles - set(self.pert2id))
        if not missing:
            logger.info("pert vocabulary already covers all benchmark perturbations")
            return

        next_id = max(self.pert2id.values()) + 1
        for offset, symbol in enumerate(missing):
            self.pert2id[symbol] = next_id + offset
            self.id2pert[next_id + offset] = symbol

        logger.info(
            "pert vocabulary extended by {} symbol(s) absent from gears_gene_set.csv "
            "({} -> {} entries): {}",
            len(missing), len(self.pert2id) - len(missing), len(self.pert2id),
            missing if len(missing) <= 40 else missing[:40] + ["..."],
        )

    def _combine_datasets(self) -> sc.AnnData:
        """
        Combine datasets based on the specified strategy.

        Upstream also supported reading per-cell-type h5ad files from a hardcoded registry and
        combining them with HVG selection / imputation (`_standard_adata_reading_processing`,
        `_combine_with_method`). Those paths and their data are unavailable here, so xcell_v2 --
        a single pre-merged, pre-HVG'd file -- is the only supported source. The zarr loader
        lands alongside it as a second branch.
        """
        if self.zarr:
            return self._load_zarr_data()

        if self.xcell_v2:
            return self._load_xcellv2_data()

        raise NotImplementedError(
            "This TxPert copy supports the canonical zarr source (datamodule.zarr=true) or the "
            "pre-merged xcell_v2 h5ad (datamodule.xcell_v2=true). The upstream multi-h5ad "
            "loading path was removed; see competitors/TxPert/README_HOOKE_TX.md."
        )

    def _load_zarr_data(self) -> sc.AnnData:
        """Load the hooke-tx canonical zarr datasets named in the manifest TOML.

        Deliberately does NOT normalize: ``X.zarr`` is already ``log1p(target_sum=1e4)`` and
        both ../STATE and ../PerturbDiff train on that scale untouched, so re-normalizing to
        TxPert's usual ``target_sum`` would make the arms incomparable. HVG selection is also
        skipped -- the datasets are already restricted to the benchmark's 2,000 genes.
        """
        from gspp.data.zarr_loader import (
            GeneIDMapper,
            load_manifest,
            load_split_perturbations,
            load_zarr_datasets,
        )

        logger.info(f"zarr: loading manifest {self.zarr_manifest_path}")
        self._zarr_manifest = load_manifest(self.zarr_manifest_path)

        adata = load_zarr_datasets(
            self._zarr_manifest,
            gene_mapper=GeneIDMapper(),
            # gspp's cell-type vocabulary (PUBLIC_CELL_TYPES) is lower-case for the Nadig lines
            # and upper-case for the Replogle ones; the zarr uses display casing throughout.
            cell_line_rename={"HepG2": "hepg2", "Jurkat": "jurkat"},
        )

        # Resolve the index-based split into perturbation names before the AnnData is filtered
        # any further, so row_in_dataset still lines up with the split file.
        self._zarr_split = load_split_perturbations(self._zarr_manifest, adata)

        self._extend_pert_vocabulary(adata)

        adata = _standardize_obs_format(adata, GENE_NAME)
        logger.info(
            f"zarr: {adata.shape[0]} cells, {adata.shape[1]} genes, "
            f"cell lines {sorted(adata.obs[CELL_TYPE].unique())}"
        )
        return adata






    def setup(self, stage: Union[str, None] = None) -> None:
        """
        Setup the data module for different stages (fit, validate, test, fine-tune).

        Args:
            stage (str): The stage to setup (fit, validate, test, fine-tune).
        """
        logger.info(f"Setting up in stage {stage}")

        if stage == "fit" or stage is None:
            self._setup_fit()
        elif stage == "validate":
            self._setup_validate()
        elif stage == "test":
            self._setup_test()
        elif stage == "fine_tune":
            self._setup_fine_tune()
            
    def _calc_de_mask(self, stage: str) -> pd.Series:
        """
        Calculate differentially expressed genes mask for evaluation.
        
        Args:
            stage (str): The stage to calculate DE mask for.
        
        Returns:
            pd.Series: The differentially expressed genes
        """
        if self.molecular_perts:
            test_de_mask = self.adata.obs[CELL_TYPE].isin(self.adata.obs[CELL_TYPE].unique())
        else:
            test_de_mask = self.adata.obs[CELL_TYPE] == self.test_cell_type
        
        if stage == "fit":
            if self.xcell or self.xcell_v2:
                train_cell_type_de_mask = self.adata.obs[CELL_TYPE].isin(
                    self.train_cell_types + [self.test_cell_type]
                )
            elif not self.molecular_perts:
                train_cell_type_de_mask = self.adata.obs[CELL_TYPE].isin(
                    self.train_cell_types
                )
            else:
                train_cell_type_de_mask = self.adata.obs[CELL_TYPE].isin(
                    self.adata.obs[CELL_TYPE].unique()
                )

            train_de_mask = (
                self.adata.obs[CONDITION].isin(self.condition_group["train"]).values
            ) & train_cell_type_de_mask

            if (self.xcell or self.xcell_v2) and self.val_cell_type is None:
                val_cell_type_de_mask = self.adata.obs[CELL_TYPE] == self.test_cell_type
            elif self.val_cell_type is not None:
                val_cell_type_de_mask = self.adata.obs[CELL_TYPE] == self.val_cell_type
            else:
                val_cell_type_de_mask = train_cell_type_de_mask

            val_de_mask = (
                self.adata.obs[CONDITION].isin(self.condition_group["val"]).values
            ) & val_cell_type_de_mask
            
            return ((test_de_mask) | (train_de_mask) | (val_de_mask))
        elif stage == "test":
            return test_de_mask

    def _setup_fit(self) -> None:
        """
        Setup the data module for the fit stage.
        """
        if self.molecular_perts:
            self.train_cell_type_mask = self.treatment_data.obs[CELL_TYPE].isin(self.treatment_data.obs[CELL_TYPE].unique())     # TODO: Currently throws all cell-types together for molecular perts
            self.test_cell_type_mask = self.train_cell_type_mask
        else:
            if self.xcell or self.xcell_v2:
                self.train_cell_type_mask = self.treatment_data.obs[CELL_TYPE].isin(
                    self.train_cell_types + [self.test_cell_type]
                )
            else:
                self.train_cell_type_mask = self.treatment_data.obs[CELL_TYPE].isin(
                    self.train_cell_types
                )

            self.val_cell_type_mask = (
                self.treatment_data.obs[CELL_TYPE] == self.val_cell_type
            ) if self.val_cell_type is not None else None

            self.test_cell_type_mask = (
                self.treatment_data.obs[CELL_TYPE] == self.test_cell_type
            )

        if self.use_predefined_splits:
            self._load_predefined_splits()
        else:
            self.condition_group, self.subgroup = (
                self.get_condition_groups_and_subgroups(
                    all_train_perts = np.unique(
                        self.treatment_data[self.train_cell_type_mask]
                        .obs[CONDITION]
                        .values
                    ),
                    test_perts = np.unique(
                        self.treatment_data[self.test_cell_type_mask]
                        .obs[CONDITION]
                        .values
                    ),
                    val_perts = None if self.val_cell_type is None else np.unique(
                        self.treatment_data[self.val_cell_type_mask]
                        .obs[CONDITION]
                        .values
                    ),
                    test_size=0.25,
                    split=self.test_split,
                )
            )

        self._create_datasets()

        cache_file = (
            f"{self.cache_path}/de_adata_fit_predefined.h5ad"
            if self.use_predefined_splits
            else f"{self.cache_path}/de_adata_fit.h5ad"
        )

        de_mask = self._calc_de_mask("fit")
        de_adata = self._load_or_calculate_de_adata(cache_file, de_mask)
        self.adata.uns.update(de_adata.uns)

    def _load_predefined_splits(self) -> None:
        """
        Load predefined splits for the data.
        """
        if self.zarr:
            self._load_zarr_splits()
            return
        if self.xcell_v2:
            self._load_xcell_v2_splits()
            return
        if self.xcell:
            self._load_xcell_splits()
            return

        if not os.path.exists(
            self.data_path
            / PUBLIC_CELL_TYPE_DATASETS[self.test_cell_type]
            / f"splits/{PUBLIC_CELL_TYPE_DATASETS[self.test_cell_type]}_simulation_1_0.75.pkl"
        ):
            raise ValueError("Predefined splits not found")
        self.condition_group = joblib.load(
            self.data_path
            / PUBLIC_CELL_TYPE_DATASETS[self.test_cell_type]
            / f"splits/{PUBLIC_CELL_TYPE_DATASETS[self.test_cell_type]}_simulation_1_0.75.pkl"
        )
        self.subgroup = joblib.load(
            self.data_path
            / PUBLIC_CELL_TYPE_DATASETS[self.test_cell_type]
            / f"splits/{PUBLIC_CELL_TYPE_DATASETS[self.test_cell_type]}_simulation_1_0.75_subgroup.pkl"
        )
        logger.info("Using predefined splits")

    def _load_xcell_splits(self) -> None:
        """
        Load XCell benchmark splits from JSON.
        """
        import json
        with open(self.xcell_splits_path) as f:
            xcell_data = json.load(f)

        val_perts = [f"{g}+ctrl" for g in xcell_data["val"]]
        test_perts = [f"{g}+ctrl" for g in xcell_data["test"]]
        held_out = set(val_perts + test_perts)

        all_perts = set(np.unique(self.treatment_data.obs[CONDITION].values))
        train_perts = list(all_perts - held_out)
        train_perts.append(CONTROL_LABEL)

        self.condition_group = {
            "train": train_perts,
            "val": val_perts,
            "test": test_perts,
        }
        self.subgroup = {
            "test_subgroup": {
                "unseen_cell": [],
                "unseen_cell_pert": test_perts,
            }
        }
        logger.info(f"XCell splits: {len(train_perts)} train, {len(val_perts)} val, {len(test_perts)} test")

    def _load_xcell_v2_splits(self) -> None:
        """
        Load XCellv2 splits: test = intersection(xcell_test, hepg2_perts), no val.
        Train = ALL perts globally; test perts excluded from HEPG2 only via mask in _create_datasets.
        """
        import json
        with open(self.xcell_splits_path) as f:
            xcell_data = json.load(f)

        xcell_test_genes = set(xcell_data["test"])

        hepg2_mask = self.treatment_data.obs[CELL_TYPE] == self.test_cell_type
        hepg2_conditions = set(self.treatment_data[hepg2_mask].obs[CONDITION].unique())
        hepg2_perts = {c.replace("+ctrl", "") for c in hepg2_conditions if c != CONTROL_LABEL}

        test_perts_raw = sorted(hepg2_perts & xcell_test_genes)
        test_perts = [f"{g}+ctrl" for g in test_perts_raw]

        all_perts = list(set(self.treatment_data.obs[CONDITION].unique()) - {CONTROL_LABEL})
        all_perts.append(CONTROL_LABEL)

        self.condition_group = {
            "train": all_perts,
            "val": [],
            "test": test_perts,
        }

        self._xcell_v2_hepg2_exclude = set(test_perts)

        self.subgroup = {
            "test_subgroup": {
                "unseen_cell": [],
                "unseen_cell_pert": test_perts,
            }
        }
        logger.info(f"XCellv2 splits: {len(all_perts)} train (global), {len(test_perts)} test (HEPG2 only), val=0")

    def _load_zarr_splits(self) -> None:
        """Install the fold from the hooke-tx split file as gspp condition groups.

        Same shape as :meth:`_load_xcell_v2_splits` -- train is every perturbation globally and
        test is the held-out HepG2 set -- but the membership comes from the index-based split file
        rather than a name list, and is resolved in ``zarr_loader.load_split_perturbations`` (which
        also proves the sides are pairwise disjoint).

        ``_create_datasets`` then excludes the held-out perturbations from HepG2 *only*, so the
        very same perturbation measured in jurkat / k562 / rpe1 still trains -- that is the
        cross-cell-line generalization the benchmark is testing.

        Validation follows exactly the same rule, and for the same reason: hooke-tx, STATE and
        PerturbDiff all carve their validation set out of the HepG2 rows while leaving those
        perturbations in the other three cell lines, so anything else here would train TxPert on
        a different amount of data than the arms it is compared against. Both held-out groups
        therefore go into one ``_xcell_v2_hepg2_exclude`` set, and ``condition_group["train"]``
        deliberately still lists them -- the row mask, not the name list, is what holds them out.
        """
        if self._zarr_split is None:
            raise RuntimeError("zarr splits requested before _load_zarr_data() ran")

        # condition_group is matched against obs[CONDITION], which _standardize_obs_format
        # writes as "<symbol>+ctrl" for single-gene perturbations.
        def as_condition(names):
            return [n if "+" in n else f"{n}+{CONTROL_LABEL}" for n in names]

        test_perts = as_condition(self._zarr_split["test"])
        val_perts = as_condition(self._zarr_split.get("validation", []))
        observed = set(self.treatment_data.obs[CONDITION].unique())

        missing = [p for p in test_perts + val_perts if p not in observed]
        if missing:
            # Perturbations dropped by _process_treatment_data because they are outside the
            # pert2id vocabulary would silently shrink the scored set; _initialize_gene_sets
            # extends the vocabulary precisely to avoid this, so treat it as an error.
            raise ValueError(
                f"{len(missing)} held-out perturbation(s) from the split file are absent from the "
                f"data after vocabulary filtering (e.g. {missing[:5]}). The scored set would "
                "not match the other competitor arms."
            )

        all_perts = list(observed - {CONTROL_LABEL})
        all_perts.append(CONTROL_LABEL)

        self.condition_group = {"train": all_perts, "val": val_perts, "test": test_perts}
        self._xcell_v2_hepg2_exclude = set(test_perts) | set(val_perts)
        self.subgroup = {
            "test_subgroup": {"unseen_cell": [], "unseen_cell_pert": test_perts},
        }
        logger.info(
            f"zarr splits: {len(all_perts)} train (global), {len(val_perts)} val, "
            f"{len(test_perts)} test (both {self.test_cell_type} only)"
        )

    def _create_datasets(self) -> None:
        """
        Create datasets for training, validation, and testing.
        """
        train_indices = (
            self.treatment_data.obs[CONDITION]
            .isin(self.condition_group["train"])
            .values
        ) & self.train_cell_type_mask

        if self.xcell_v2:
            hepg2_test_mask = (
                (self.treatment_data.obs[CELL_TYPE] == self.test_cell_type) &
                self.treatment_data.obs[CONDITION].isin(self._xcell_v2_hepg2_exclude)
            ).values
            train_indices = train_indices & ~hepg2_test_mask

        if(train_indices.sum() == 0):
            raise ValueError("Training split is empty.")

        self.train_adata = self.treatment_data[train_indices]
        self.train_conditions = self.conditions[train_indices]
        self.train_condition_names = self.condition_names[train_indices]

        self.train_data = XcelltypePerturbDataset(
            self.train_adata,
            self.control_data_train,
            self.train_conditions,
            self.train_condition_names,
            self.device,
            train_cond_names=self.control_train_condition_names,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            smiles2vec=self.smiles2vec if self.molecular_perts else None,
            unique_doses=self.unique_doses,
            obsm_key=self.obsm_key,
        )

        # No validation group -> no validation dataset. This is the original xcell_v2 / zarr
        # behaviour (replogle_nadig_xcell.json ships `validation: []`) and is left byte-identical.
        # A populated group takes the normal path below, which for xcell_v2 means the held-out
        # perturbations of the TEST cell type -- the 4,825 HepG2 rows of the fewshot split.
        if self.xcell_v2 and not self.condition_group.get("val"):
            self.val_data = None
            self.val_adata = None
            self.val_conditions = None
            self.val_condition_names = None
        else:
            if self.xcell or self.xcell_v2:
                val_cell_type_mask = self.test_cell_type_mask
            elif self.val_cell_type is not None:
                val_cell_type_mask = self.val_cell_type_mask
            else:
                val_cell_type_mask = self.train_cell_type_mask

            val_indices = (
                self.treatment_data.obs[CONDITION].isin(self.condition_group["val"]).values
            ) & val_cell_type_mask
            if(val_indices.sum() == 0):
                raise ValueError("Validation split is empty.")
            logger.info(
                f"validation split: {int(val_indices.sum()):,} cells over "
                f"{len(self.condition_group['val'])} perturbations"
            )

            self.val_adata = self.treatment_data[val_indices]
            self.val_conditions = self.conditions[val_indices]
            self.val_condition_names = self.condition_names[val_indices]

            self.val_data = XcelltypePerturbDataset(
                self.val_adata,
                self.control_data_val,
                self.val_conditions,
                self.val_condition_names,
                self.device,
                match_cntr=self.match_cntr,
                avg_cntr=self.avg_cntr,
                smiles2vec=self.smiles2vec if self.molecular_perts else None,
                unique_doses=self.unique_doses,
                obsm_key=self.obsm_key,
            )

        test_mask = (
            self.treatment_data.obs[CONDITION].isin(self.condition_group["test"]).values
        ) & self.test_cell_type_mask
        if(test_mask.sum() == 0):
            raise ValueError("Test split is empty.")

        self.test_adata = self.treatment_data[test_mask]
        self.test_conditions = self.conditions[test_mask]
        self.test_condition_names = self.condition_names[test_mask]

        self.test_data = XcelltypePerturbDataset(
            self.test_adata,
            self.control_data_test,
            self.test_conditions,
            self.test_condition_names,
            self.device,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            smiles2vec=self.smiles2vec if self.molecular_perts else None,
            unique_doses=self.unique_doses,
            obsm_key=self.obsm_key,
        )


    def _setup_validate(self) -> None:
        """
        Setup the data module for the validation stage.
        """
        val_mask = (
            self.treatment_data.obs[CONDITION].isin(self.condition_group["val"]).values
        ) & self.val_cell_type_mask

        self.val_adata = self.treatment_data[val_mask]
        self.val_conditions = self.conditions[val_mask]
        self.val_condition_names = self.condition_names[val_mask]

        self.val_data = XcelltypePerturbDataset(
            self.val_adata,
            self.control_data_val,
            self.val_conditions,
            self.val_condition_names,
            self.device,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            smiles2vec=self.smiles2vec if self.molecular_perts else None,
            unique_doses=self.unique_doses,
            obsm_key=self.obsm_key,
        )

    def _setup_test(self) -> None:
        """
        Setup the data module for the test stage.
        """
        test_mask = (
            self.treatment_data.obs[CONDITION].isin(self.condition_group["test"]).values
        ) & self.test_cell_type_mask

        self.test_adata = self.treatment_data[test_mask]
        self.test_conditions = self.conditions[test_mask]
        self.test_condition_names = self.condition_names[test_mask]

        self.test_data = XcelltypePerturbDataset(
            self.test_adata,
            self.control_data_test,
            self.test_conditions,
            self.test_condition_names,
            self.device,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            smiles2vec=self.smiles2vec if self.molecular_perts else None,
            unique_doses=self.unique_doses,
            obsm_key=self.obsm_key,
        )

        cache_file = (
            f"{self.cache_path}/de_adata_test_predefined.h5ad"
            if self.use_predefined_splits
            else f"{self.cache_path}/de_adata_test.h5ad"
        )
        mask = self._calc_de_mask("test")
        de_adata = self._load_or_calculate_de_adata(cache_file, mask)
        self.adata.uns.update(de_adata.uns)

    def _setup_fine_tune(self) -> None:
        """
        Setup the data module for the fine-tuning stage.
        """
        self.train_data = XcelltypePerturbDatasetFineTune(
            self.control_data_test,
            self.control_test_condition_names,
            self.device,
        )

    def _load_or_calculate_de_adata(self, cache_file: str, mask: pd.Series) -> sc.AnnData:
        """
        Load or calculate differentially expressed genes for evaluation.

        Args:
            cache_file (str): Path to the cache file.
            indices (pd.Series): The indices to use for calculating DE genes.

        Returns:
            sc.AnnData: The differentially expressed genes AnnData object.
        """
        if os.path.exists(cache_file) and not self.debug and not self.xcell_v2:
            logger.info(f"de_adata cached, loading from {cache_file}")
            de_adata = sc.read_h5ad(cache_file)
        else:
            de_adata = self.adata[mask]
            de_adata = calculate_de_for_eval(de_adata, molecular_perts=self.molecular_perts)
            if not self.debug and not self.xcell_v2:
                de_adata.write(cache_file)
                logger.info(f"de_adata saved to {cache_file}")
        
        return de_adata

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for training data.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for validation data.

        Raises when there is no validation dataset. It used to return ``test_dataloader()``, which
        meant every ``val_*`` metric on the zarr / xcell_v2 arm was silently a TEST metric -- and
        anything selected on it (an epoch budget, a checkpoint, a hyperparameter) was selected with
        test knowledge. STATE shipped the identical fallback and it is fixed there the same way.

        Callers that legitimately have no validation split must check ``val_data is None`` rather
        than relying on this returning something.
        """
        if self.val_data is None:
            raise RuntimeError(
                "val_dataloader() called but no validation dataset exists. This split has an "
                "empty validation group; returning the test loader here would report test metrics "
                "as validation metrics. Use a manifest whose split file has a validation list "
                "(configs/manifests/ihabxcell_hepg2_fewshot_zarr.toml)."
            )
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for test data.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def fine_tune_train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for fine-tuning data.
        """
        return DataLoader(
            self.fine_tune_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def get_all_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return all DataLoaders (train, validation, test).
        """
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    def get_condition_groups_and_subgroups(
        self,
        all_train_perts: np.ndarray,
        test_perts: np.ndarray,
        val_perts: np.ndarray = None,
        test_size: float = 0.25,        # TODO: Make configurable (and smaller for mols)
        split: int = 0,
        val_unseen_fraction: float = 1.0,
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, np.ndarray]]]:
        """
        Setup train, val, test for seen & unseen perturbations; note this is in complement to the major reservation of a cell type for testing.

        Args:
            all_train_perts (list): All unique perturbations in the training cell types.
            test_perts (list): All unique perturbations in the test cell type.
            test_size (float, optional): Fraction for test (& val) set. Defaults to 0.25.
            split (int, optional): Split for K-fold cross validation. Defaults to 0.
            val_unseen_fraction (float, optional): Fraction of the validation set reserved for unseen perturbations. Defaults to 1.0.

        Returns:
            dict: Condition groups (train, val, test perturbations), subgroup (which test perturbations are seen/unseen).
        """
        # Split test perturbations into seen and unseen groups
        test_seen_perts, test_unseen_perts = self._split_test_perts(
            test_perts, test_size, split
        )
        
        if val_perts is not None:
            val_seen_perts, val_unseen_perts = self._split_test_perts(
                val_perts, test_size, split
            )

        # Remove unseen test perturbations from training perturbations
        train_perts = np.setdiff1d(all_train_perts, test_unseen_perts)

        # Split validation unseen perturbations from training perturbations
        if val_perts is None:
            val_unseen_perts = self._split_val_unseen_perts(
                train_perts, test_size, val_unseen_fraction
            )

        # Remove validation unseen perturbations from training perturbations
        train_perts = np.setdiff1d(train_perts, val_unseen_perts)

        # Select validation perturbations from training perturbations
        val_perts = self._select_val_perts(
            train_perts, test_size, val_unseen_perts, val_unseen_fraction
        )

        # Add control perturbation to training perturbations
        train_perts = np.append(train_perts, CONTROL_LABEL)
        val_perts = np.append(val_perts, CONTROL_LABEL)

        # Create condition groups for train, validation, and test sets
        condition_groups = {
            "train": list(train_perts),
            "val": list(val_perts),
            "test": list(test_perts),
        }

        test_subgroups = {
            "unseen_cell": test_seen_perts,
            "unseen_cell_pert": test_unseen_perts,
        }

        # Adjust condition groups and subgroups if training and test cell types are the same
        if (
            len(self.train_cell_types) == 1
            and self.train_cell_types[0] == self.test_cell_type
        ):
            condition_groups["test"] = list(test_unseen_perts)
            test_subgroups["unseen_cell"] = []
            
        return condition_groups, {"test_subgroup": test_subgroups}

    def _split_test_perts(
        self, test_perts: np.ndarray, test_size: float, split: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split test perturbations into seen and unseen groups.

        Args:
            test_perts (np.ndarray): All unique perturbations in the test cell type.
            test_size (float): Fraction for test set.
            split (int): Split for K-fold cross validation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Seen and unseen test perturbations.
        """
        kf = KFold(n_splits=int(1 / test_size), shuffle=True, random_state=42)
        test_seen_perts_idx, test_unseen_perts_idx = list(kf.split(test_perts))[split]
        return test_perts[test_seen_perts_idx], test_perts[test_unseen_perts_idx]

    def _split_val_unseen_perts(
        self, train_perts: np.ndarray, test_size: float, val_unseen_fraction: float
    ) -> np.ndarray:
        """
        Split validation unseen perturbations from training perturbations.

        Args:
            train_perts (np.ndarray): All unique perturbations in the training cell types.
            test_size (float): Fraction for test set.
            val_unseen_fraction (float): Fraction of the validation set reserved for unseen perturbations.

        Returns:
            np.ndarray: Validation unseen perturbations.
        """
        val_unseen_size = test_size * val_unseen_fraction
        kfv = KFold(n_splits=int(1 / val_unseen_size), shuffle=True, random_state=23)
        _, val_unseen_perts_idx = list(kfv.split(train_perts))[0]
        
        return train_perts[val_unseen_perts_idx]

    def _select_val_perts(
        self,
        train_perts: np.ndarray,
        test_size: float,
        val_unseen_perts: np.ndarray,
        val_unseen_fraction: float,
    ) -> np.ndarray:
        """
        Select validation perturbations from training perturbations.

        Args:
            train_perts (np.ndarray): All unique perturbations in the training cell types.
            test_size (float): Fraction for test set.
            val_unseen_perts (np.ndarray): Validation unseen perturbations.

        Returns:
            np.ndarray: Validation perturbations.
        """
        val_seen_size = test_size * (1 - val_unseen_fraction)
        val_perts_seen = np.random.RandomState(42).choice(
                train_perts, int(len(train_perts) * val_seen_size), replace=False
            )
        return np.append(val_perts_seen, val_unseen_perts)


class XcelltypePerturbDataset(Dataset):
    def __init__(
        self,
        adata: sc.AnnData,
        control_data: sc.AnnData,
        pert_conditions: pd.Series,
        condition_names: pd.Series,
        device: str = "cuda",
        train_cond_names: pd.Series = None,
        match_cntr: bool = False,
        avg_cntr: bool = False,
        smiles2vec: Dict[str, torch.Tensor] = None,
        unique_doses: List[str] = None,
        obsm_key: str = cs.ObsmKey.SCGPT,
    ):
        """
        Dataset for perturbation data.

        Args:
            adata (AnnData): The perturbation data
            control_data (AnnData): The control data
            pert_conditions (pd.Series): The perturbation conditions
            condition_names (pd.Series): The condition names
            device (str): The device to use
            train_cond_names (pd.Series): The condition names for the control data
            match_cntr (bool): Whether to match the control based on the batch
        """
        self.obsm_key = obsm_key

        self.device = device
        self.match_cntr = match_cntr
        self.avg_cntr = avg_cntr
        self.smiles2vec = smiles2vec
        if self.smiles2vec is not None:
            self.fp_dim = torch.cat(list(next(iter(self.smiles2vec.values())).values())).shape
            self.unique_doses = unique_doses

        self.control_data_raw_X = self._to_tensor(control_data.X)
        # TODO, this precludes using both X, and an embedding, but simple first; maybe make flexible
        self.pert_data = self._to_tensor(adata.X)
        if self.obsm_key == cs.ObsmKey.RAW:
            self.control_data = self._to_tensor(control_data.X)
        else:
            self.control_data = self._to_tensor(control_data.obsm[obsm_key])

        self.pert_conditions = pert_conditions
        self.condition_names = condition_names
        
        self.index = adata.obs.index.to_series()
        self.control_index = control_data.obs.index.to_series()
        
        self.treatment_cell_types = adata.obs[CELL_TYPE]
        self.treatment_cell_batches = adata.obs[BATCH]
                
        self.control_cell_types = control_data.obs[CELL_TYPE]
        self.control_cell_batches = control_data.obs[BATCH]
        
        if self.smiles2vec is not None:
            self.treatment_doses = adata.obs[DOSE]
            self.control_doses = control_data.obs[DOSE]

        self._initialize_all_treatment_data()
        self._initialize_control_data(train_cond_names, adata, control_data)
        
        if self.smiles2vec is not None:
            self._initialize_dose_enc()

    def _to_tensor(self, data: csr_matrix) -> torch.Tensor:
        """
        Convert sparse matrix to tensor.

        Args:
            data (csr_matrix): Sparse matrix data.

        Returns:
            torch.Tensor: Tensor data.
        """
        return torch.Tensor(data.toarray()).to(self.device)

    def _initialize_all_treatment_data(self) -> None:
        """
        Initialize all treatment data by combining control and treatment
        """
        
        self.all_cell_types = pd.concat(
            [self.treatment_cell_types, self.control_cell_types]
        )

        self.all_cell_batches = pd.concat(
            [self.treatment_cell_batches, self.control_cell_batches]
        )
        
        if self.smiles2vec is not None:
            self.all_doses = pd.concat(
                [self.treatment_doses, self.control_doses]
            )

    def _initialize_control_data(
        self,
        train_cond_names: pd.Series,
        adata: sc.AnnData,
        control_data: sc.AnnData
    ) -> None:
        """
        Initialize control data based on the matching mode.

        Args:
            train_cond_names (pd.Series): The condition names for the control data.
            adata (AnnData): The perturbation data.
            control_data (AnnData): The control data.
        """
        # Collect information for control matching
        self.control_ct_indices = {}  # For random matching on cell-type only
        self.control_ct_batch_indices = {}  # For random matching on cell-type & batch

        for ct in np.unique(self.all_cell_types).tolist():
            self.control_ct_batch_indices[ct] = {}

            ct_mask = adata.obs[CELL_TYPE] == ct

            # For random matching on cell-type only
            # if not self.match_cntr:
            ct_mask_cntr = control_data.obs[CELL_TYPE] == ct
            self.control_ct_indices[ct] = ct_mask_cntr.values.nonzero()[0]

            # TODO: Temporary fix because we add control later to training set
            available_batches = list(set(np.unique(adata[ct_mask].obs[BATCH]).tolist() + np.unique(control_data.obs[BATCH]).tolist()))

            # Collect indices of control data for each batch
            for batch in available_batches:
                ct_batch_mask_cntr = (control_data.obs[CELL_TYPE] == ct) & (
                    control_data.obs[BATCH] == batch
                )
                self.control_ct_batch_indices[ct][
                    batch
                ] = ct_batch_mask_cntr.values.nonzero()[0]
        
        # Extend the dataset with training data only if we don't use FM
        if train_cond_names is not None:
            self._extend_with_train_data(train_cond_names)

        self.control_cache = {}

    def _initialize_dose_enc(self) -> None:
        self.dose2enc = {
            dose: torch.eye(len(self.unique_doses))[i]
            for i, dose in enumerate(self.unique_doses)
        }

    def _extend_with_train_data(self, train_cond_names: pd.Series) -> None:
        """
        Extend the dataset with training data.

        Args:
            train_cond_names (pd.Series): The condition names for the control data.
        """
        self.pert_data = torch.cat([self.pert_data, self.control_data_raw_X])
        self.pert_conditions = pd.concat(
            [self.pert_conditions, pd.Series([['ctrl']] * self.control_data_raw_X.shape[0])]
        )
        self.condition_names = pd.concat([self.condition_names, train_cond_names])
        self.index = pd.concat([self.index, self.control_index])
        self.treatment_cell_types = self.all_cell_types
        self.treatment_cell_batches = self.all_cell_batches

        if self.smiles2vec is not None:
            self.treatment_doses = self.all_doses

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return self.pert_data.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, pd.Series, pd.Series, pd.Series, Union[str, None], pd.Series
    ]:
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple: Perturbation data, control data, perturbation condition, condition name, index, cell type, experimental batch.
        """
        control = self._get_control_data(idx)
        pert_data = self.pert_data[idx]

        p_emb = None

        if self.smiles2vec is not None:
            smiles = self.pert_conditions.iloc[idx][:-1]    # TODO: Fix in case of multiple molecular perts
            if len(smiles) > 0 and len(smiles[0]) > 0:
                p_emb = self.smiles2vec[smiles[0]]
                p_emb = torch.cat(list(p_emb.values()))
            else:
                p_emb = torch.zeros(self.fp_dim)

        return (
            pert_data,
            control,
            torch.cat([p_emb, self.dose2enc[self.treatment_doses.iloc[idx]]]) if p_emb is not None else torch.zeros(2), # TODO: control is just a dummy here
            self.pert_conditions.iloc[idx],
            self.condition_names.iloc[idx],
            self.treatment_doses.iloc[idx] if self.smiles2vec is not None else "1+1",
            self.index.iloc[idx],
            self.treatment_cell_types.iloc[idx],
            self.treatment_cell_batches.iloc[idx],
        )

    def _get_control_data(self, idx: int) -> torch.Tensor:
        """
        Get control data based on the matching mode.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: Control data.
        """
        cell_type = self.treatment_cell_types.iloc[idx]
        batch = self.treatment_cell_batches.iloc[idx]
        if self.match_cntr:
            if self.avg_cntr:
                control = self.control_data[self.control_ct_batch_indices[cell_type][batch],:].mean(0)
            else:
                control = self.control_data[
                    np.random.choice(self.control_ct_batch_indices[cell_type][batch])
                ]
        else:
            if self.avg_cntr:
                if cell_type not in self.control_cache.keys():
                    self.control_cache[cell_type] = self.control_data[self.control_ct_indices[cell_type],:].mean(0)
                
                control = self.control_cache[cell_type]
            else:
                control = self.control_data[
                    np.random.choice(self.control_ct_indices[cell_type])
                ]
        return control



class XcelltypePerturbDatasetFineTune(Dataset):
    def __init__(
        self, control_data: sc.AnnData, condition_names: pd.Series, device: str = "cuda"
    ):
        """
        Dataset for fine-tuning with control data.

        Args:
            control_data (AnnData): The control data.
            condition_names (pd.Series): The condition names.
            device (str): The device to use.
        """
        self.full_ctrl = torch.Tensor(control_data.X.toarray()).to(device)
        self.pert_condition = pd.Series([[-1]] * self.full_ctrl.shape[0])
        self.condition_names = condition_names
        self.index = control_data.obs.index.to_series()

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return self.full_ctrl.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, pd.Series, pd.Series, pd.Series, Union[str, None], Union[pd.Series, None]
    ]:
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple: Perturbation data, control data, perturbation condition, condition name, index, cell type, experimental batch.
        """
        return (
            self.full_ctrl[idx],
            self.full_ctrl[idx],
            self.pert_condition.iloc[idx],
            self.condition_names.iloc[idx],
            self.index.iloc[idx],
            None,
            None,
        )
