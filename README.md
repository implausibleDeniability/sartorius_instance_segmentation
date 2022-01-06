# Kaggle Competition: Sartorius Cell Instance Segmentation

Solution of Sartorius Cell Instance Segmentation competition.

![train-data-visualisation](figures/45a1f06614f0-astro.png)
![train-data-visualisation](figures/508d39dcc9ef-cort.png)
![train-data-visualisation](figures/aff8fb4fc364-shsy5y.png)
*Types of cells from top to bottom: astro, cort, shsy5y*

## Contents
- **Mask R-CNN**
    The initial model we've used was Mask R-CNN. 
    The model was trained with **pytorch**.
    For the usage example, see the instructions in section below.
- **Cascade R-CNN**
    Cascade R-CNN was trained through **mmdetection** library.
    All necessary data transformation, model architecture, training and inference
    scripts are available at `mmdetection_training/`
- **Transfer Learning**
    The organizers of the dataset supply large dataset with various types of cells. 
    The dataset was used for transfer learning, see `src/pretrain.py`
- **Hyperparameters optimization**
    Hyperparameters were optimized with **optuna**. 
    Optimization scripts, config template and instructions are in `finetuning_parameters/`
- **Postprocessing optimization**
    Postprocessing of the NN output includes non-maximal suppression, removal of overlapping pixels, 
    and other transformations. Parameters for these transformations can be optimzed through 
    `thresholds_optimization.py`. Results are available in `thresolds/`

## How to run training script

1. Download data (see instructions below)
2. Rename `.env.example` file to `.env` and specify paths for dataset and weights storage
3. Run `pip install -r requirements.txt` for installing needed packages
4. Run training by command `python train_val_mask_rcnn.py --device cuda:0 --exp_name init-training`

## Additional information

### Links

- kaggle competition overview: [kaggle.com](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/)
- state of the project: [notion.so](https://www.notion.so/Kaggle-Instance-Segmentation-f5a291c7ffc34559927d2dedb8405c14)

### Troubleshooting

- `ModuleNotFoundError...` - add root folder of repository by running `export PYTHONPATH=$PYTHONPATH:/path/to/repo`

### Downloading data from competition

- `cd data/`
- `kaggle competitions download -c sartorius-cell-instance-segmentation`
- `unzip sartorius-cell-instance-segmentation.zip`

## Authors of repository

- Maxim Faleev, [github.com](https://github.com/implausibleDeniability)
- Shamil Arslanov, [github.com](https://github.com/homomorfism)
