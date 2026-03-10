# Fine-Tuning Gemma-3-4B-IT with Native PyTorch

This script fine-tunes `google/gemma-3-4b-it` for Visual Question Answering (VQA) using a native PyTorch training loop, avoiding high-level abstractions like Hugging Face `Trainer`.

*   **Native PyTorch Loop**: Pure PyTorch training and validation.
*   **Full Fine-Tuning**: Updates all parameters (no PEFT).
*   **Memory**: Uses `bfloat16`, gradient checkpointing, and accumulation.
*   **Performance**: Uses `flash_attention_2`.
*   **Data**: Custom collate function for multi-modal inputs.

*   From Huggingface discussion forum and the Gemma authors

*   <img width="1274" height="492" alt="image" src="https://github.com/user-attachments/assets/032ec48b-3cb4-43ae-8f41-e3b023978aba" />

## Dataset

Uses `dutta18/multi-domain-VQA-12K` dataset.

## Usage

**Packages Needed:**
`torch`, `transformers`, `datasets`, `numpy`, `tqdm`, `einops`, `accelerate`, `flash-attn`

**Steps:**
1.  Update paths to your local CoT pickle files in the script.
2.  Adjust hyperparameters in `Gemma-3-4B-FT-Script.py` if needed.
3.  Run: `python Gemma-3-4B-FT-Script.py`

## Note

This performs full fine-tuning and requires significant GPU memory. I have done the full finetuning on below hardware:

NVIDIA 80 GB H_100 GPU with a batch size of 4, that took me around 30 mins for 2 epochs. 

NVIDIA A40 46 GB GPU with batch size of 2 with gradient accumulation of 2, that took me around 45 mins.

For consumer GPUs, consider using for LoRA or QLORA.

UPDATE: I have also added the batched inference code with some performance metrics.

Please ★ Star the Repo if you found this useful.
