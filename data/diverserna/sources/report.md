# DiverseRNA source identification

Corpus: `txfm_ICLR_train_data_composite___raw_counts.h5ad`
Shape: 1,434,299 cells x 44,728 genes

## Structural fingerprint

| block | rows | row range | % nonzero | max value | integral | median row sum | genes supported |
|---|---|---|---|---|---|---|---|
| 0 | 502,080 | 0–502,079 | 7.0 | 977 | yes | 11,033 | 8,248 |
| 1 | 102,205 | 502,080–604,284 | 8.7 | 3,908 | yes | 16,528 | 15,432 |
| 2 | 99,300 | 604,285–703,584 | 6.8 | 5,801 | yes | 5,978 | 18,287 |
| 3 | 71,585 | 703,585–775,169 | 6.8 | 404,058 | yes | 10,514 | 17,572 |
| 4 | 88,399 | 775,170–863,568 | 7.4 | 8 | **NO** | 496 | 16,403 |
| 5 | 31,542 | 863,569–895,110 | 7.3 | 9 | **NO** | 3,004 | 20,693 |
| 6 | 504,929 | 895,111–1,400,039 | 7.3 | 7 | **NO** | 2,054 | 20,267 |
| 7 | 10,526 | 1,400,040–1,410,565 | 50.2 | 26,309,592 | yes | 60,686,904 | 35,708 |
| 8 | 23,733 | 1,410,566–1,434,298 | 38.3 | 27,590,498 | yes | 40,963,360 | 19,556 |

## Attribution

| block | rows | status | source | licence |
|---|---|---|---|---|
| 0 | 502,080 | UNRESOLVED | — | — |
| 1 | 102,205 | UNRESOLVED | — | — |
| 2 | 99,300 | UNRESOLVED | — | — |
| 3 | 71,585 | UNRESOLVED | — | — |
| 4 | 88,399 | UNRESOLVED | — | — |
| 5 | 31,542 | UNRESOLVED | — | — |
| 6 | 504,929 | UNRESOLVED | — | — |
| 7 | 10,526 | CONFIRMED | GTEx_homosapiens_bulk | — |
| 8 | 23,733 | CONFIRMED | 2 candidates | — |

## What this means

`CONFIRMED` blocks matched a candidate on exact row count with a corroborating gene count. `UNRESOLVED` blocks have no exact row match, which is what a subsetted source looks like; identifying those needs the gene-Jaccard and row-hash axes against a wider candidate registry.

**No block may be published without a licence.** See `data/diverserna/UPLOAD_GATE.md`.
