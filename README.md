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


## 🧠 First-Stage: k-Space Autoencoders

We have implemented three different architectures:
- U-net
- Discrete Autoencoder (KspaceAutoencoder)
- Variational Autoencoder (KspaceAutoencoderKL)

The U-net follows the basic structure with the possibilty to remove the resiual skip connections and input normalization.
The Discrete Autoencoder is based on the Encoder and Decoder classes of the Hugging Face diffusers library. 
The Variational Autoencoder follows the imlementation of the first-stage model in the LDM paper with diffusers backbone architecture AutoencoderKL.


Train a K-AE model:

```bash
python resume_train_first_stage.py --config cfg/<own>.yaml
```

Supported architectures:

- **U-Net bottleneck encoder**
- **Discrete Autoencoder (AE)**
- **Variational Autoencoder (VAE)**
- Variants with **SSIM-based losses**

The trained autoencoder produces **low-dimensional latent embeddings** used for diffusion modeling.

---

## 🌫️ Second-Stage: Latent Diffusion (LDenoiser)

Train a latent diffusion model on k-AE embeddings:

```bash
python scripts/train_diffusion.py --config configs/ldm.yaml
```

This model learns to denoise latent samples and approximate the distribution of fully sampled k-space.

---

## 🔄 Consistency-Guidance Sampler (CGS)

The CGS is used **during inference** to enforce partial data consistency.

Run inference on undersampled k-space:

```bash
python scripts/run_inference.py --config configs/inference.yaml
```

The pipeline:

1. Encode undersampled k-space  
2. Perform latent diffusion with CGS  
3. Decode to full k-space  
4. Apply inverse FFT for image reconstruction

---

## 📊 Evaluation

Evaluation metrics include:

- **PSNR**
- **SSIM**
- **NMSE**
- **LPIPS**

Example:

```bash
python scripts/evaluate.py --pred <results_path> --target <gt_path>
```

---

## 📘 Citation

If you use this repository, please cite the corresponding master’s thesis:

```
[Add citation once published]
```

---

## 📄 License

This project is released under the MIT License.

---

## 🤝 Contributing

Pull requests, issues, and discussions are welcome.  
The repository is designed to support further research on **k-space diffusion**, **latent generative MRI models**, and **data-consistent sampling strategies**.

---

## 🧩 Acknowledgements

This project builds on concepts from:

- Latent Diffusion Models (LDM)
- fastMRI dataset
- Diffusion-based MRI reconstruction literature

Special thanks to contributors and the research community exploring generative methods for medical imaging.

[^1]: https://fastmri.med.nyu.edu/