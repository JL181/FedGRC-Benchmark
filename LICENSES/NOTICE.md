# Third-party notices

The framework license is in [LICENSE](../LICENSE). FedDrPlus retains its [MIT notice](FedDrPlus-MIT.txt); FedLog and Fed3R retain [GPL-3.0](GPL-3.0.txt). BTFL uses Apache-2.0. Dataset terms are separate from code licenses.

Sources for the methods and data readers are listed below.

| Implementation | Source / scope |
|---|---|
| FedAvg, FedRep, FedROD, FedLC, FedTHE, FedETF, pFedFDA, FedVLS, FedYoYo | Local framework adaptations of the cited methods; retain the benchmark's models/data/evaluation rather than upstream experiment runners |
| FedFA | [TailinZhou/FedFA](https://github.com/TailinZhou/FedFA); feature anchors |
| FedDrPlus | [curisam/FedDr_plus](https://github.com/curisam/FedDr_plus), dot regression + global feature distillation; MIT |
| BTFL | [ZhouYuCS/BTFL](https://github.com/ZhouYuCS/BTFL); DLE and Bayesian gating; Apache-2.0 |
| HPFL | [tmlr-group/HPFL](https://github.com/tmlr-group/HPFL); personalized head selection |
| FedLog | [rossyu/fedlog](https://github.com/rossyu/fedlog); body updates, sufficient statistics and MAP solve adapted from upstream; GPL-3.0 |
| pFedGM | Implementation of Hu & Ma, arXiv:2603.11620, with the declared full-label-space missing-prototype policy |
| CCVR, FedNCM, FedCGS | Classifier operators reimplemented for common frozen-backbone feature statistics; see the corresponding original papers |
| Fed3R | [Erosinho13/Fed3R](https://github.com/Erosinho13/Fed3R); closed-form classifier; GPL-3.0 |
| FedCOF | [dipamgoswami/FedCOF](https://github.com/dipamgoswami/FedCOF); class-mean covariance estimation and classifier operator |
| GHLDA | [FabioTur-dev/One-Shot-FL](https://github.com/FabioTur-dev/One-Shot-FL); GH-LDA sufficient-statistic and shrinkage operator |
| FedRS reader | [dongdongzhaoUP/FedRS-Bench](https://github.com/dongdongzhaoUP/FedRS-Bench); NIID-1, semantic label mapping and image transforms |

The available source records do not establish a top-level license for HPFL, FedCOF or GH-LDA. Their distribution permissions remain unresolved in those records; no replacement license is assigned here.
