# kLD-MRI: K-space latent diffusion for Accelerated MRI

This repository contains the official implementation of **kLD-MRI**, a framework that explores the application of *latent diffusion models directly in k‑space* for accelerated MRI reconstruction.  
It includes tools for dataset preprocessing, first‑stage autoencoder training, latent diffusion modeling, and the evaluation pipeline.

---

## 🌐 Overview

Conventional diffusion-based MRI reconstruction methods operate in the **image domain**, potentially losing access to the informative high‑frequency structure that naturally resides in k‑space.  
**kLD-MRI** investigates whether generative models can instead operate *directly in the frequency domain*, leveraging:

- A **k-space Autoencoder (K-AE)** to learn compact latent representations  
- A **latent diffusion model (LDenoiser)** trained on these representations  
- A **Consistency-Guidance Sampler (CGS)** to enforce partial data consistency  

The project provides a complete pipeline for creating latent datasets, training first‑stage models, and experimenting with diffusion-based reconstruction in latent k-space.

The pipeline:

1. Encode undersampled k-space  
2. Perform latent diffusion with CGS  
3. Decode to full k-space  
4. Apply inverse FFT for image reconstruction

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sandrosage/kspace-diffusion.git
cd kspace-diffusion
```

### 2. Install dependencies

```bash
conda env create -f environment.yaml
conda activate kdiff
```

### 3. Prepare the dataset

The project is based on the single-coil knee fastMRI dataset. It can be downloaded at the offical NYI fastMRI page [^1]

---
## 🧠 Training

### First-Stage: K-Space Autoencoders

We have implemented three different architectures:
- U-net
- Discrete Autoencoder (KspaceAutoencoder)
- Variational Autoencoder (KspaceAutoencoderKL)

The U-net follows the basic structure with the possibilty to remove the resiual skip connections and input normalization.
The Discrete Autoencoder is based on the Encoder and Decoder classes of the Hugging Face diffusers library. 
The Variational Autoencoder follows the imlementation of the first-stage model in the LDM paper with diffusers backbone architecture AutoencoderKL.

For the Discrete Autoencoder (AE) and the Variational Autoencoder there are also variants available that integrate the Structural Similarity Measure (SSIM) as part of the objective function.

- WeightedSSIMKspaceAutoencoder
- WeightedSSIMKspaceAutoencoderKL


#### Train a K-AE model:

So in order to train your K-AE model you have to specify your config file via `--config`. It follows the basic structure of YAML and all the config files can be found in the `cfg`. If you want to train your own model based on different hyperparameters, you can use these config files as templates.

```bash
python resume_train_first_stage.py --config cfg/<own_file>.yaml --id <run_id>
```

For the training you can specify another flag, namely `--id`. This is the run id of the wandb run. So if you have already trained a model, then you can resume the training with the corresponding run id. For the first training run, you can leave the *checkpoint_path* empty in the config file, but if you want to resume the training then you also have to specify the path to the model checkpoints here.

### Second-Stage: Latent Diffusion (LDenoiser)

The LDM is based on an unconditional UNet2DModel from the diffusers library. The diffusion process is scheduled by the DDPMScheduler again from the diffusers libarry.

#### Create the latent dataset

For the training of the LDM, you first have to create the dataset. Therefore you have to provide the path to the checkpoints of the desired K-AE model via `--ckpt`.
With the `--partition` flag, you can specify the partition (train, val, test) on which you want to generate the latent embeddings dataset.

```bash
python extract_latent.py --ckpt <path_to_checkpoints> --partition train
```

#### Train the LDM (LDenoiser) on K-AE embeddings:

So in order to train the LDM now, you have to specify the same flags as in the training configuration for the K-AE. The `--id` specifies the wandb run id and `--config` specifies the path to the config file for the LDM.

```bash
python resume_train_ldm.py --config cfg/<own_file>.yaml --id <run_id>
```

This model learns to denoise latent samples and approximate the distribution of fully sampled k-space.

---

## 📊 Evaluation

Evaluation metrics include:

- **PSNR**
- **SSIM**
- **NMSE**
- **LPIPS**

You can either evaluate the first-stage models using:

```bash
python test_first_stage.py --config test_cfg/<own_file>.yaml --undersampling -- accelerations 8 --mask_type equispaced
```

The `--config` flag determines the path to the evaluation configuration. Similar to the template in the training phase. In the `test_cfg/` directory, you can find some templates. Additionally, you can activate the `--undersampling` flag, so that the model is tested on undersampled k-space and not on fully sampled k-space. Therefore, further configurations have to be set:
- `accelerations`: (int) accelerations factor (4,8)
- `--mask_type`: (str) type of mask function (random, equispaced, etc.)

## 📘 Citation

If you use this repository, please cite the corresponding master’s thesis:

```
[Add citation once published]
```

---

## 👩‍⚖️ License
Copyright © Sandro Sage.
All rights reserved.
Please see the [license file](LICENSE) for terms.

---

## 🧩 Acknowledgements

This project builds on concepts from:

- Latent Diffusion Models (LDM)
- fastMRI dataset
- Diffusion-based MRI reconstruction literature

Special thanks to contributors and the research community exploring generative methods for medical imaging.

[^1]: https://fastmri.med.nyu.edu/