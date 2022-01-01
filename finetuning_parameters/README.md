This directory contains scripts, needed for tuning mask-rcnn model.

Steps for running tuning:

- edit params.yaml for adding/excluding types for lr, optimizers and schedulers
- run `python tuning_model.py` and specify needed parameters (use `-h` flag for it)
- look output logs and visualisations at `./logs` folder and look graphs at wandb.