# Fine-Tuning Gemma-3-4B-IT with Native PyTorch

This script fine-tunes `google/gemma-3-4b-it` for Visual Question Answering (VQA) using a native PyTorch training loop, avoiding high-level abstractions like Hugging Face `Trainer`.

*   **Native PyTorch Loop**: Pure PyTorch training and validation.
*   **Full Fine-Tuning**: Updates all parameters (no PEFT).
*   **Memory**: Uses `bfloat16`, gradient checkpointing, and accumulation.
*   **Performance**: Uses `flash_attention_2`.
*   **Data**: Custom collate function for multi-modal inputs.

## Dataset

Uses `dutta18/multi-domain-VQA-12K` (physical reasoning) and custom Chain-of-Thought (CoT) data.

## Usage

**Packages Needed:**
`torch`, `transformers`, `datasets`, `numpy`, `tqdm`, `einops`, `accelerate`, `flash-attn`

**Steps:**
1.  Update paths to your local CoT pickle files in the script.
2.  Adjust hyperparameters in `Gemma-3-4B-FT-Script.py` if needed.
3.  Run: `python Gemma-3-4B-FT-Script.py`

## Note

This performs full fine-tuning and requires significant GPU memory. I have done the full finetuning on a NVIDIA 80 GB H_100 GPU with a batch size of 4, that took me around 30 mins for 2 epochs. 

For consumer GPUs, consider adapting for LoRA.

UPDATE: I have also added the batched inference code with some performance metrics.

-> Star the Repo if you found this useful.