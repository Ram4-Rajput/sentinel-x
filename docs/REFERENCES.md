# Sentinel-X — Research References

Curated references grouped by the Sentinel-X component / phase each supports.
Links point to open-access sources (arXiv / official pages) where possible.
Content summarized for compliance with source licensing.

---

## Datasets (Phase 1)

- **CTU-13** (primary benchmark, scenario 11) — Garcia, Grill, Stiborek, Zunino,
  "An empirical comparison of botnet detection methods," *Computers & Security*, 2014.
  Canonical citation requested by the Stratosphere Lab.
  - PDF: https://ri.conicet.gov.ar/bitstream/11336/6772/2/CONICET_Digital_Nro.9220_A.pdf
  - Dataset: https://www.stratosphereips.org/datasets
- **CSE-CIC-IDS2018** — Canadian Institute for Cybersecurity (CIC/UNB).
  - Dataset: https://www.unb.ca/cic/datasets/ids-2018.html
- **UNSW-NB15** — Moustafa & Slay, 2015.
  - Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CICIoT2023** — Canadian Institute for Cybersecurity.
  - Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **NetFlow-standardized NIDS datasets** (cross-dataset generalization) — Sarhan et al.,
  "NetFlow Datasets for ML-based NIDS," 2020. https://arxiv.org/abs/2011.09144

## World-Model Core (Phase 4)

- Ha & Schmidhuber, "World Models," 2018 — encoder → latent → recurrent dynamics →
  forecasting formulation that Sentinel-X mirrors.
  - https://arxiv.org/abs/1803.10122
  - NeurIPS version: https://arxiv.org/abs/1809.01999

## Graph Encoder (Phase 4 — GAT / GraphSAGE stage)

- Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs"
  (GraphSAGE), 2017. https://arxiv.org/abs/1706.02216
- Veličković et al., "Graph Attention Networks" (GAT), 2018.
  https://arxiv.org/abs/1710.10903
- Lo et al., "E-GraphSAGE: A GNN-based Intrusion Detection System" — closest prior art;
  adapts GraphSAGE to NetFlow edge features for NIDS. https://arxiv.org/abs/2103.16329

## Temporal / Dynamic Graph Modeling (Phases 2 + 4)

- "A Comprehensive Survey of Dynamic Graph Neural Networks," 2024.
  https://arxiv.org/abs/2405.00476
- "Graph Neural Networks for Temporal Graphs," 2023. https://arxiv.org/abs/2302.01018
- "A Survey on Temporal Graph Representation Learning and Generative Modeling," 2022.
  https://arxiv.org/abs/2208.12126

## Uncertainty, Calibration, Novelty / OOD (Phase 6)

- Guo et al., "On Calibration of Modern Neural Networks" (temperature scaling), 2017.
  https://arxiv.org/abs/1706.04599
- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using
  Deep Ensembles," 2017. https://arxiv.org/abs/1612.01474
- Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and OOD Examples"
  (max softmax baseline), 2016. https://arxiv.org/abs/1610.02136
- Hendrycks et al., "Scaling Out-of-Distribution Detection for Real-World Settings," 2019.
  https://arxiv.org/abs/1911.11132

## Explainability + Counterfactual (Phase 8)

- Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks," 2019.
  https://arxiv.org/abs/1903.03894
- Lucic et al., "CF-GNNExplainer: Counterfactual Explanations for GNNs," 2022.
  https://arxiv.org/abs/2102.03322

## Attack Interpretation (Phase 7)

- MITRE ATT&CK — adversary tactics/techniques knowledge base.
  - https://attack.mitre.org
  - Enterprise techniques: https://attack.mitre.org/techniques/enterprise/

---

## BibTeX

```bibtex
@article{garcia2014ctu13,
  title   = {An empirical comparison of botnet detection methods},
  author  = {Garcia, Sebastian and Grill, Martin and Stiborek, Jan and Zunino, Alejandro},
  journal = {Computers \& Security},
  volume  = {45},
  pages   = {100--123},
  year    = {2014},
  publisher = {Elsevier}
}

@inproceedings{moustafa2015unswnb15,
  title     = {UNSW-NB15: a comprehensive data set for network intrusion detection systems},
  author    = {Moustafa, Nour and Slay, Jill},
  booktitle = {Military Communications and Information Systems Conference (MilCIS)},
  year      = {2015}
}

@misc{sarhan2020netflow,
  title         = {NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems},
  author        = {Sarhan, Mohanad and Layeghy, Siamak and Moustafa, Nour and Portmann, Marius},
  year          = {2020},
  eprint        = {2011.09144},
  archivePrefix = {arXiv}
}

@misc{ha2018worldmodels,
  title         = {World Models},
  author        = {Ha, David and Schmidhuber, J{\"u}rgen},
  year          = {2018},
  eprint        = {1803.10122},
  archivePrefix = {arXiv}
}

@inproceedings{hamilton2017graphsage,
  title     = {Inductive Representation Learning on Large Graphs},
  author    = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@inproceedings{velickovic2018gat,
  title     = {Graph Attention Networks},
  author    = {Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018}
}

@misc{lo2021egraphsage,
  title         = {E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT},
  author        = {Lo, Wai Weng and Layeghy, Siamak and Sarhan, Mohanad and Gallagher, Marcus and Portmann, Marius},
  year          = {2021},
  eprint        = {2103.16329},
  archivePrefix = {arXiv}
}

@inproceedings{guo2017calibration,
  title     = {On Calibration of Modern Neural Networks},
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2017}
}

@inproceedings{lakshminarayanan2017ensembles,
  title     = {Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
  author    = {Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@inproceedings{hendrycks2017baseline,
  title     = {A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks},
  author    = {Hendrycks, Dan and Gimpel, Kevin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017}
}

@inproceedings{ying2019gnnexplainer,
  title     = {GNNExplainer: Generating Explanations for Graph Neural Networks},
  author    = {Ying, Rex and Bourgeois, Dylan and You, Jiaxuan and Zitnik, Marinka and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}

@inproceedings{lucic2022cfgnnexplainer,
  title     = {CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks},
  author    = {Lucic, Ana and ter Hoeve, Maartje and Tolomei, Gabriele and de Rijke, Maarten and Silvestri, Fabrizio},
  booktitle = {International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2022}
}

@misc{mitreattack,
  title  = {MITRE ATT\&CK},
  author = {{The MITRE Corporation}},
  howpublished = {\url{https://attack.mitre.org}}
}
```
