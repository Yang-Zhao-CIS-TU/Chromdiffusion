# ChromDiffusion

**ChromDiffusion: Refining Chromatin Structure Characterization Through Diffusion Models**

*Yang Zhao, Tomoya Kanno, Chong Li, and Xinghua Shi*  
**Manuscript status:** Under review

ChromDiffusion is a diffusion-based framework for refining predicted high-resolution Hi-C contact maps. It is designed to improve chromatin contact-map reconstruction by learning corrections to the outputs of existing Hi-C super-resolution models.

## Overview

Hi-C experiments measure spatial contacts between genomic loci, but generating high-resolution contact maps can be costly and data-intensive. ChromDiffusion builds on an existing super-resolution backbone and applies a diffusion-based refinement stage to its predicted contact maps.

<p align="center">
  <img src="Figures/chromdiff-fig-update.png" alt="ChromDiffusion Framework" width="850">
</p>

<p align="center">
  <em>Overview of the ChromDiffusion framework.</em>
</p>
The repository includes implementations and related resources for several Hi-C reconstruction backbones:

- **HiCARN**  (`HICARN/`)
- **HiCNN**  (`HICNN/`)
- **DeepHiC** (`deephic/`)
- **SRHiC** (`srhic/`)

ChromDiffusion is intended to complement these backbones rather than replace them.

## Framework

1. Prepare paired low- and high-resolution Hi-C contact-map patches.
2. Generate an initial high-resolution prediction using a backbone model.
3. Apply diffusion-based refinement to the backbone prediction.
4. Evaluate reconstructed contact maps against reference data.

<!-- Add the framework figure after it is finalized, for example:
![ChromDiffusion framework](assets/chromdiffusion_framework.png)
-->

## Repository Structure

```text
Chromdiffusion/
├── HiCARN/       # HiCARN-related code
├── HiCNN/        # HiCNN-related code
├── deephic/      # DeepHiC-related code
├── srhic/        # SRHiC-related code
└── benchmark/    # Benchmarking resources
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/shilab/Chromdiffusion.git
cd Chromdiffusion
```

The backbone directories contain model-specific implementations. Please refer to their scripts and configuration files for data preparation, training, and inference. A consolidated, verified quick-start workflow will be added in a future documentation update.


## Citation

If you use this repository, please cite the accompanying manuscript once its bibliographic details are available:

> Yang Zhao, Tomoya Kanno, Chong Li, and Xinghua Shi. *ChromDiffusion: Refining Chromatin Structure Characterization Through Diffusion Models*. Manuscript under review.

## Status

This repository is under active development. Documentation, reproducibility instructions, and release artifacts may be updated.
