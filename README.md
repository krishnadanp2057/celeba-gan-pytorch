# CelebA Face Generation with a Vanilla GAN (PyTorch)

A from-scratch implementation of a Generative Adversarial Network (GAN) in PyTorch that learns to generate 64×64 human face images from the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. The project covers the full pipeline: custom dataset loading, preprocessing, adversarial training, and visualization of generated samples.

## Overview

A GAN consists of two networks trained against each other:

- **Generator (G):** maps a random noise vector `z ~ N(0, I)` to a synthetic image.
- **Discriminator (D):** classifies an image as real (from the dataset) or fake (from the generator).

The generator tries to fool the discriminator, while the discriminator tries to tell the two apart. Over training, the generator learns to produce increasingly realistic faces.

## Features

- Custom PyTorch `Dataset` for loading CelebA images from a folder
- Preprocessing pipeline: resize, center crop to 64×64, normalize to `[-1, 1]`
- Fully connected Generator and Discriminator
- Standard GAN objective using Binary Cross-Entropy loss
- Adam optimizer (`lr = 2e-4`, `betas = (0.5, 0.999)`)
- Automatic GPU (CUDA) usage when available
- Generated image grids visualized after each epoch

## Model Architecture

**Generator**

| Layer | Output size |
|---|---|
| Input noise `z` | 100 |
| Linear + ReLU | 256 |
| Linear + ReLU | 512 |
| Linear + ReLU | 1024 |
| Linear + Tanh | 3 × 64 × 64 (12,288) |

**Discriminator**

| Layer | Output size |
|---|---|
| Flatten | 12,288 |
| Linear + LeakyReLU(0.2) | 1024 |
| Linear + LeakyReLU(0.2) | 512 |
| Linear + LeakyReLU(0.2) | 256 |
| Linear + Sigmoid | 1 |

## Dataset

This project uses the **CelebA (CelebFaces Attributes)** dataset, specifically the aligned and cropped images (`img_align_celeba`).

1. Download the dataset from the [official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
2. Extract the images into a folder, for example `./data/img_align_celeba/`.

> The dataset is not included in this repository because of its size and license terms.

## Getting Started

### Requirements

- Python 3.9+
- PyTorch and torchvision
- Pillow, NumPy, Matplotlib
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install torch torchvision pillow numpy matplotlib jupyter
```

### Running the project

1. Place the dataset at `./data/img_align_celeba/`.
2. Open `gan.ipynb` and set the dataset path:
   ```python
   dataset_path = "./data/img_align_celeba"
   ```
3. Run all cells. Training progress and generated samples are shown in the notebook.

A GPU is strongly recommended. The notebook also runs on Google Colab with a free GPU runtime.

## Training Configuration

| Parameter | Value |
|---|---|
| Image size | 64 × 64 |
| Batch size | 128 |
| Latent dimension | 100 |
| Optimizer | Adam (lr = 0.0002, β1 = 0.5, β2 = 0.999) |
| Loss | Binary Cross-Entropy |
| Epochs | Configurable (`train(..., epochs=N)`) |

## Results

Add generated samples here after training, for example:

| Epoch 1 | Epoch 10 | Epoch 25 |
|---|---|---|
| ![epoch1](samples/epoch_1.png) | ![epoch10](samples/epoch_10.png) | ![epoch25](samples/epoch_25.png) |

## Limitations and Future Work

- The current model uses fully connected layers, which do not exploit spatial structure in images. Sample quality is limited compared with convolutional approaches.
- Planned improvements:
  - Convolutional **DCGAN** architecture (`ConvTranspose2d` / `Conv2d` with BatchNorm)
  - Saving sample grids and model checkpoints to disk during training
  - Loss curve plotting for the generator and discriminator
  - Improved stability techniques (label smoothing, spectral normalization, WGAN-GP)
  - Quantitative evaluation with FID (Fréchet Inception Distance)

## References

- I. Goodfellow et al., [*Generative Adversarial Networks*](https://arxiv.org/abs/1406.2661), 2014
- A. Radford et al., [*Unsupervised Representation Learning with Deep Convolutional GANs*](https://arxiv.org/abs/1511.06434), 2015
- Z. Liu et al., [*Deep Learning Face Attributes in the Wild*](https://arxiv.org/abs/1411.7766) (CelebA), 2015

## Author

**Krishnandan Pandit**
GitHub: [@your-username](https://github.com/your-username)

## License

This project is released under the MIT License. The CelebA dataset is subject to its own terms of use for non-commercial research purposes.