# kLD-MRI: Latent Diffusion in k-Space for Accelerated MRI Reconstruction

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

## 📂 Repository Structure

```
kLD-MRI/
├── data/                    # Preprocessed k-space datasets or links to raw data
├── kAE/                     # First-stage autoencoder architectures (U-Net, AE, VAE)
├── diffusion/               # LDenoiser latent diffusion model
├── cgs/                     # Consistency-Guidance Sampler implementation
├── utils/                   # Preprocessing, masking, and helper functions
├── configs/                 # Training and evaluation configs
├── scripts/                 # Training & evaluation entry points
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/kLD-MRI.git
cd kLD-MRI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

This framework uses **fully sampled and undersampled k-space data**.  
You may use:

- **fastMRI** singlecoil knee dataset  
- Custom datasets (must be stored as complex-valued tensors with shape `(2, H, W)`)

Run preprocessing:

```bash
python scripts/preprocess_kspace.py --input <raw_path> --output <processed_path>
```

---

## 🧠 First-Stage: k-Space Autoencoders

Train a k-AE model:

```bash
python scripts/train_kAE.py --config configs/ae_4x.yaml
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