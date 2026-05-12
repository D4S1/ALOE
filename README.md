# VAE-Based Genotype-to-Phenotype Mapping in Follicular Lymphoma

This project uses variational autoencoders to model genotype-to-phenotype relationships in follicular lymphoma. It builds on [CaClust](https://github.com/szczurek-lab/CaClust), a probabilistic graphical model for single-cell genotyping, and extends it with a deep learning framework to better capture the functional consequences of somatic evolution in cancer.

In short: tumors are genetically messy, and we're trying to figure out how that genetic messiness maps to differences in cell behavior using VAEs.



## Setup

### Conda environment

The easiest way to get everything running is to recreate the conda environment from the provided `factmx_env.yml` file:

```bash
conda env create -f factmx_env.yml
conda activate factmx
```

> Requires [conda](https://docs.conda.io/en/latest/miniconda.html) or [mamba](https://mamba.readthedocs.io/) (recommended for faster installs — just replace `conda` with `mamba` above).

## Usage

### Training & evaluation

Model training and evaluation are done interactively in `run.ipynb`. Open it with:

```bash
jupyter notebook run.ipynb
```

The notebook walks through the full pipeline — data loading, model training, and evaluation of the learned genotype-to-phenotype mappings.