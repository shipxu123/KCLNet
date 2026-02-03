# KCLNet Framework
Official implementation of "KCLNet: Electrically Equivalence-Oriented Graph Representation Learning for Analog Circuits" (Xu et al., AAAI 2026).

## Dataset
A download link of dataset is provided as following:
https://pan.quark.cn/s/f1e683e9358c

## Project Structure
- `arl/`: core library
  - `data/`: dataset loaders and graph builders
  - `model/`: GNN models and encoders (`backbone/`, `constrastive_encoders/`)
  - `solver/`: training and evaluation entry points
  - `loss/`, `eval/`, `utils/`, `optimizer/`, `lr_scheduler/`: supporting modules
- `requirements.txt`: pinned Python dependencies

## Setup
Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric (and compiled extensions like `torch-scatter`) must be installed separately to match your CUDA/CPU environment.

## Quickstart
The solver scripts are the main entry points. They accept `--config` (YAML) and `--phase`:

```
python arl/solver/pretrain_solver.py --config path/to/config.yaml --phase train
python arl/solver/graphcls_solver.py --config path/to/config.yaml --phase eval
```

We recommend keeping local configs under a `configs/` folder (not tracked) and passing the file explicitly.

## Config, Data, and Outputs
- Config files are YAML. The directory containing the config is treated as the run root.
- Runs create `checkpoints/`, `events/`, `results/`, and `log.txt` alongside the config.
- The dataset is not included in this repository; a download link will be provided separately.

## Citation
If you use this code, please cite the AAAI 2026 paper.

@inproceedings{xu2026kclnet,
  author    = {Xu, Peng and Li, Yapeng and Chen, Tinghuan and Ho, Tsung-Yi and Yu, Bei},
  title     = {KCLNet: Electrically Equivalence-Oriented Graph Representation Learning for Analog Circuits},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026},
  address   = {Singapore}
}