# Flickr8k Text-to-Image Generation with LoRA

A Generative AI project that fine-tunes **Stable Diffusion v1.5** with **LoRA** on the Flickr8k image-caption dataset and provides a Gradio interface for text-to-image generation.

The project covers the complete workflow from dataset preparation and LoRA fine-tuning to inference and interactive deployment.

## Overview

The project uses image-caption pairs from Flickr8k to adapt Stable Diffusion v1.5 toward the visual concepts represented in the dataset.

The workflow is:

```text
Flickr8k Images + Captions
          |
          v
     Data Cleaning
          |
          v
    Metadata Creation
          |
          v
Stable Diffusion v1.5
          |
          v
   LoRA Fine-Tuning
          |
          v
   LoRA Adapter Weights
          |
          v
      Inference
          |
          v
      Gradio UI
          |
          v
     Generated Image
```

## Objectives

The main objectives are to:

- Prepare the Flickr8k image-caption dataset for training.
- Fine-tune Stable Diffusion using Low-Rank Adaptation (LoRA).
- Reduce the number of trainable parameters by adapting the UNet rather than training the complete diffusion model.
- Generate images from natural-language prompts.
- Build an interactive Gradio application for inference.
- Support reproducible image generation through a configurable seed.

## Dataset

The training notebook uses the **Flickr8k** image-caption dataset.

The notebook expects the dataset on Google Drive in the following structure:

```text
ImageCaptioning/
├── Images/
└── captions.txt
```

The notebook converts the caption file into a `metadata.csv` file and changes the image captions into descriptive prompts beginning with:

```text
a photograph of a scene, ...
```

The dataset itself is **not included in this repository** because of its size and licensing/distribution considerations.

## Model

### Base Model

The project uses:

```text
runwayml/stable-diffusion-v1-5
```

as the base text-to-image diffusion model.

### LoRA

The model is fine-tuned using **Low-Rank Adaptation (LoRA)**.

Instead of updating all Stable Diffusion parameters, LoRA adds trainable low-rank matrices to selected UNet layers.

The supplied adapter configuration uses:

- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.0`
- PEFT type: `LORA`

The configured target modules include attention projections and feed-forward layers such as:

```text
to_q
to_k
to_v
to_out.0
ff.net.0.proj
ff.net.2
```

This makes the fine-tuning process substantially more parameter-efficient than full-model training.

## Training Configuration

The notebook uses the following training settings:

| Parameter | Value |
|---|---:|
| Base model | Stable Diffusion v1.5 |
| Learning rate | `1e-5` |
| Epochs | `3` |
| Batch size | `1` |
| Gradient accumulation | `4` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| Weight decay | `1e-2` |
| LR scheduler | Cosine |
| Warmup steps | `100` |

The training pipeline also uses gradient checkpointing and optional xFormers memory-efficient attention to reduce GPU memory usage.

## Training Workflow

The notebook performs the following steps:

1. Installs compatible PyTorch and xFormers versions.
2. Loads the Flickr8k dataset from Google Drive.
3. Creates a cleaned metadata file.
4. Validates available image files.
5. Applies image preprocessing.
6. Loads Stable Diffusion v1.5.
7. Freezes the VAE and text encoder.
8. Applies LoRA to selected UNet modules.
9. Trains only the LoRA parameters.
10. Generates sample images using the fine-tuned UNet.
11. Saves the LoRA adapter and tokenizer configuration.

## Image Preprocessing

The training notebook validates image files and applies preprocessing before training.

The dataset class:

- Checks whether image files exist.
- Verifies that images can be opened.
- Converts images to RGB.
- Applies the configured image transformations.

Invalid or missing images are filtered from the training metadata.

## Inference

The project includes an inference application built using **Gradio**.

The application loads:

```text
Stable Diffusion v1.5
        +
Flickr8k LoRA Adapter
```

and provides a simple interface where the user can enter an image prompt and generate an image.

The application supports a seed parameter:

```text
-1 → Random generation
42 → Reproducible generation
```

Using the same prompt and fixed seed can reproduce the same generation under the same environment and model configuration.

## Gradio Application

The application provides:

- Text prompt input
- Seed input
- Image generation button
- Generated image output
- Generation status information

The default inference settings in the supplied application are:

- Inference steps: `8`
- Guidance scale: `5`

The application also uses the Euler Discrete Scheduler with Karras sigmas and enables available memory optimizations. 

## Repository Structure

```text
flickr8k-text-to-image-lora/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── flickr8k_lora_training.ipynb
│
├── app/
│   └── app.py
│
├── model/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── README.md
│
└── docs/
```

### `notebooks/flickr8k_lora_training.ipynb`

Contains the complete training workflow, including dataset preparation, LoRA configuration, training, testing, and model saving.

### `app/app.py`

Contains the Gradio inference application.

### `model/`

Contains the supplied LoRA adapter weights and configuration.

The adapter is intended to work with the Stable Diffusion v1.5 base model.

## Installation

Clone the repository:

```bash
git clone https://github.com/MansiiSapariya/flickr8k-text-to-image-lora.git
cd flickr8k-text-to-image-lora
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The supplied requirements include:

- PyTorch 2.6.0
- Diffusers
- Transformers
- Accelerate
- PEFT
- Gradio
- xFormers

## Training

Training is designed for a **GPU-enabled environment**, such as Google Colab.

Open:

```text
notebooks/flickr8k_lora_training.ipynb
```

Before running the notebook, place the Flickr8k dataset on Google Drive:

```text
/content/drive/MyDrive/ImageCaptioning/
├── Images/
└── captions.txt
```

The notebook creates the required metadata and trains the LoRA adapter.

Because Stable Diffusion fine-tuning is computationally intensive, a CUDA-enabled GPU is strongly recommended.

## Running the Gradio App

From the repository root:

```bash
python app/app.py
```

The application loads the Stable Diffusion v1.5 base model and the LoRA adapter configured in the application.

The supplied application currently points to the Hugging Face Hub repository:

```text
Khatijaliya/flicker8k-lora
```

If the adapter is hosted under a different Hugging Face repository, update the `LORA_FLICKER_HUB_REPO` value in `app/app.py`.

## Model Artifact

The repository includes the supplied LoRA adapter:

```text
model/adapter_model.safetensors
```

and its configuration:

```text
model/adapter_config.json
```

The adapter configuration identifies the model as a PEFT LoRA adapter and specifies the target UNet modules and rank configuration.

## Important Notes

### GPU Requirement

Training Stable Diffusion with LoRA requires substantial GPU memory and computation.

The notebook was designed around a CUDA-enabled Google Colab environment.

### Flickr8k Dataset

The dataset is not included in the repository.

You need to obtain the dataset separately and place it in the expected Google Drive location before running the training notebook.

### Hugging Face Model Access

The application downloads the Stable Diffusion base model and LoRA adapter from their configured model repositories.

You may need to authenticate with Hugging Face depending on the model/repository access requirements.

### Model Size

The LoRA adapter is included as a binary `.safetensors` file. The base Stable Diffusion model is not included because it is substantially larger and is downloaded through the Diffusers/Hugging Face model-loading process.

## Key Learnings

This project provides practical experience with:

- Generative AI
- Text-to-image generation
- Diffusion models
- Stable Diffusion
- LoRA fine-tuning
- Parameter-efficient fine-tuning
- Hugging Face Diffusers
- Hugging Face PEFT
- PyTorch
- GPU-based model training
- Image-caption datasets
- Prompt-based image generation
- Gradio deployment

## Limitations

- Training requires a capable GPU.
- The Flickr8k dataset is not included.
- Image quality depends on the base model, training configuration, dataset, and prompt.
- The supplied training configuration uses only three epochs and may not represent an optimal training schedule.
- The inference application depends on the availability of the configured Hugging Face adapter repository.
- Generated images may not perfectly match every prompt.

## Future Scope

Potential improvements include:

- Training for additional epochs with systematic validation.
- Hyperparameter tuning for LoRA rank, learning rate, and guidance scale.
- Comparing different LoRA configurations.
- Adding negative prompts to the inference interface.
- Allowing users to control inference steps and guidance scale.
- Adding image generation history.
- Evaluating generated images using CLIP or other image-text metrics.
- Deploying the application as a persistent Hugging Face Space.
- Experimenting with larger or newer diffusion models.

## Author

**Mansi Sapariya**

MSc Data Science  
Christ (Deemed to be University), Bangalore

## Academic Context

This project was developed as part of Generative AI coursework and demonstrates the practical implementation of parameter-efficient fine-tuning for text-to-image generation.
