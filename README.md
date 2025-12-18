This repository contains the code for the project course I did at DTU focused on exploring the geometry of latent protein representation.

The project extract the proteins representation from the ESM2 model implementation in JAX which needs to be cloned from [here](https://github.com/irhum/esmjax). \
Due to storage constraints, the logs from the model training and all the experiments are available in [this google drive folder](https://drive.google.com/drive/folders/1p0c1Lp1JHkTiGF2BSWjClv2R8v7rlQCM?usp=drive_link).\
The google drive folder will be available until 01/03/2026.

Once downloaded the logs and cloned the esmjax repository the root directory should look like: 
* `configs/`: YAML files for geodesics and encoding comparisons
* `data/`: Dataset for training models and dataset infos
* `scripts/`: All scripts for training and geodesics
* `esmjax_clone/`: JAX implementation of ESM2
* `logs/`: Model states and experiment results
