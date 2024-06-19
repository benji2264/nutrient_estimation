# HS Nutrient Estimation

Code for nutrient estimation from hypercubes

## To Do

Immediate next steps:

- [ ] Add harvest 1 to training data (for now training + val is done on harvest 2)
- [ ] Implem LR scheduler (OneCycle or Cosine) instead of fixed LR
- [ ] Fix last formatting issues (e.g. wth is ```CA_0.png``` supposed to be??)
- [ ] Fix val metrics computation (e.g. R^2 is ]-inf; 0] rn for some reasons)
- [ ] Scale preds post training for fair comparisons with Dr. Sierra Young (for now targets are normalized during training: cf [here](https://github.com/benji2264/nutrient_estimation/blob/472a21831112e9f426b3ff8f3d655cb875babf75/src/hs_utils.py#L36))

## Installation

We recommend creating using conda for environment management.

```
conda create -n hs_nutrient python=3.9
conda activate hs_nutrient
```

Then, you can clone and install this repo.

```
git clone https://github.com/benji2264/nutrient_estimation.git
cd nutrient_estimation
pip install -e . 
```
## Visualize hypercubes

We provide a demo notebook to visualize .bin and .hdr files and save the correponding .jpg file for each wavelength and the whole RGB image. You may find this notebook at [notebooks/hs_visualization.ipynb](https://github.com/benji2264/nutrient_estimation/blob/main/notebooks/hs_visualization.ipynb).
