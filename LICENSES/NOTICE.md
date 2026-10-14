# Third-party notices

The framework license is in [LICENSE](../LICENSE). FedDrPlus retains its [MIT notice](FedDrPlus-MIT.txt); FedLog and Fed3R retain [GPL-3.0](GPL-3.0.txt). BTFL uses Apache-2.0. Dataset terms are separate from code licenses.

The recorded source revisions and implementation provenance are listed below.

| Implementation | Source / scope |
|---|---|
| FedAvg, FedRep, FedROD, FedLC, FedTHE, FedETF, pFedFDA, FedVLS, FedYoYo | Local framework adaptations of the cited methods; retain the benchmark's models/data/evaluation rather than upstream experiment runners |
| FedFA | [TailinZhou/FedFA](https://github.com/TailinZhou/FedFA), reviewed commit `e4aa421b32d89cf176f78d5e6eb49f66c589893a`; feature-anchor semantics |
| FedDrPlus | [curisam/FedDr_plus](https://github.com/curisam/FedDr_plus), dot regression + global feature distillation; MIT |
| BTFL | [ZhouYuCS/BTFL](https://github.com/ZhouYuCS/BTFL), commit `ada7b55c776164f9d11f0871f8abb83df249388e`; DLE and Bayesian gating; Apache-2.0 |
| HPFL | [tmlr-group/HPFL](https://github.com/tmlr-group/HPFL), commit `c21ebf2a7b03a03262afae218a98a427fb89fa9a`; method implementation without the upstream FedML runtime |
| FedLog | [rossyu/fedlog](https://github.com/rossyu/fedlog); body updates, sufficient statistics and MAP solve adapted from upstream; GPL-3.0 |
| pFedGM | Implementation of Hu & Ma, arXiv:2603.11620, with the declared full-label-space missing-prototype policy |
| CCVR, FedNCM, FedCGS | Classifier operators reimplemented for common frozen-backbone feature statistics; see the corresponding original papers |
| Fed3R | [Erosinho13/Fed3R](https://github.com/Erosinho13/Fed3R), commit `c2a8e5de59e4b86efaa351c241a4a3357bcdab34`; closed-form stage adapted from `src/fed3r.py`; GPL-3.0 |
| FedCOF | [dipamgoswami/FedCOF](https://github.com/dipamgoswami/FedCOF); class-mean covariance estimation and classifier operator |
| GHLDA | [FabioTur-dev/One-Shot-FL](https://github.com/FabioTur-dev/One-Shot-FL), reviewed revision `fd6daf2`; GH-LDA sufficient-statistic and shrinkage operator |
| FedRS reader | [dongdongzhaoUP/FedRS-Bench](https://github.com/dongdongzhaoUP/FedRS-Bench); NIID-1, semantic label mapping and image transforms |

The available source records do not establish a top-level license for HPFL, FedCOF or GH-LDA. Their distribution permissions remain unresolved in those records; no replacement license is assigned here.
