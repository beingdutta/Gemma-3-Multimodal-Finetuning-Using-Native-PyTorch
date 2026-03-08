# Fine-Tuning Gemma-3-4B-IT with Native PyTorch

This repository contains a script for fine-tuning the `google/gemma-3-4b-it` model for a Visual Question Answering (VQA) task using a native PyTorch training loop.

The primary goal of this project is to provide a clear, straightforward example of fine-tuning a large multi-modal model without relying on high-level abstractions like the Hugging Face `Trainer` class or `PEFT` libraries.

## 🤔 Motivation

While searching for resources, I found it challenging to find a simple, clean training script for multi-modal models that uses a standard PyTorch training loop. Many available examples depend on the Hugging Face `Trainer`, `deepspeed`, or other complex frameworks.

This repository was created to fill that gap, offering a minimalist and easy-to-understand implementation for those who prefer working directly with PyTorch.

---

## 🌟 Key Features

*   **Native PyTorch Loop**: The entire training and validation process is written in pure PyTorch, giving you full control over the training loop.
*   **No High-Level Abstractions**: This code does not use the Hugging Face `Trainer` class, `accelerate`, or `deepspeed`.
*   **Full Parameter Fine-Tuning**: The script performs full fine-tuning of the model. It does **not** use Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA or QLoRA.
*   **Memory Efficiency**:
    *   Uses `bfloat16` for mixed-precision training.
    *   Enables `gradient_checkpointing` to save memory at the cost of a small computational overhead.
    *   Includes `gradient_accumulation` to simulate larger batch sizes.
*   **Performance**:
    *   Integrates `flash_attention_2` for faster and more memory-efficient attention calculations.
*   **Custom Data Handling**: Demonstrates how to use a custom collate function to process image and text data for a multi-modal model.

## 💿 Dataset

This script is configured to train on the `dutta18/multi-domain-VQA-12K` dataset, specifically focusing on examples with the `"physical"` reasoning type. It also integrates custom Chain-of-Thought (CoT) data to guide the model's reasoning process.

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.8+
*   A CUDA-enabled GPU with sufficient VRAM.
*   Hugging Face Hub token for downloading the model.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-link>
cd <your-repo-name>
pip install -r requirements.txt
```

A potential `requirements.txt` would be:
```
torch
transformers
datasets
numpy
tqdm
einops
accelerate
flash-attn --no-build-isolation
```

### 3. Data Setup

The script expects the Chain-of-Thought (CoT) data to be present at the paths specified in the code. Please update these paths to point to your local `COT-train-set-12K.pkl` and `COT-val-set-12K.pkl` files.

### 4. Running the Script

You can configure hyperparameters like `LR`, `epochs`, and `batchSize_` directly within the `Gemma-3-4B-FT-Script.py` file.

Once configured, run the script:

```bash
python Gemma-3-4B-FT-Script.py
```

## 💡 A Note on Budget Constraints

This repository demonstrates **full fine-tuning**, which is computationally expensive and requires significant GPU memory. If you are working with limited hardware or budget constraints, I highly recommend exploring Parameter-Efficient Fine-Tuning (PEFT) methods.

Approaches like **LoRA** or **QLoRA** can dramatically reduce the memory footprint by training only a small subset of the model's parameters, making it feasible to fine-tune large models on consumer-grade GPUs.

## ⭐ Show Your Support

If you found this repository useful, please consider giving it a star! 🌟

---