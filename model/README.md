# LoRA Adapter

This directory contains the supplied LoRA adapter weights and configuration.

The adapter is trained on the Flickr8k image-caption dataset and is intended to be applied to the
`runwayml/stable-diffusion-v1-5` base model.

The application in `app/app.py` loads the adapter from the Hugging Face Hub by default. The local
weights are included here as the supplied fine-tuned model artifact.
