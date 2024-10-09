# DiffCDD

This is the official implementation of the paper "DiffCDD: Exploring Covalent Drug Design With Reinforced Latent Diffusion Models".
There are two seperate folders, one implements DiffCDD-uncond and DiffCDD-cond, and the other one is DiffCDD-guide and DiffCDD-rl

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Develop](#develop)
  - [Dataset](#dataset)
  - [Inference](#inference)

## Installation

```bash
pip install -r requirements.txt
```

## Usage
For a more detailed usage instruction, please refer to the README.md in each separate folder.

## Develop
### Dataset
#### CrossDocked Dataset (Index by [3D-Generative-SBDD](https://github.com/luost26/3D-Generative-SBDD))

Download from the compressed package we provide <https://figshare.com/articles/dataset/crossdocked_pocket10_with_protein_tar_gz/25878871> (recommended). The alternative method is to obtain the files from the [3D-Generative-SBDD's index file](https://github.com/luost26/3D-Generative-SBDD/blob/main/data/README.md) and the [raw data for the CrossDocked2020 set](https://github.com/gnina/models/tree/master/data/CrossDocked2020). The script will re-fetch the required files.

```bash
tar xzf crossdocked_pocket10_with_protein.tar.gz
```

The following files are required to exist:

- `$sbdd_dir/split_by_name.pt`
- `$sbdd_dir/index.pkl`

Finally, run the script

```bash
python scripts/preprocess/crossdocked.py \
    --sbdd-dir <PATH TO crossdocked_pocket10_with_protein> \
    # --crossdocked-dir <PATH TO CrossDocked2020> # Not needed when using the recommended method
```


### Train
>The trained models are provided in the repository at 'DiffCDD/DiffCDD_uncond_cond/Models' and 'DiffCDD/DiffCDD_rl_guide/Models'.

### Inference
Please refer to the README.md in each separate folder.



