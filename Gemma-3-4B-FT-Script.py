# Model Page: https://huggingface.co/google/gemma-3-4b-it

import os 
print(os.getenv("CONDA_DEFAULT_ENV"))
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import pickle
import random
import datasets
import datetime
import logging
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset
from functools import partial
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ### Set up Logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.info("Logging is set up in the script!")

# ### Force Determinism

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
print("\nForcing determinism by setting random seeds.")

# ### Load the MultiDomain Dataset

prefix = "Generate Step by Step Thought and the final answer from the given image and question"

def prepend_prefix(example):
    example['question'] = prefix + ': ' + example['question']
    return example

def ensure_rgb_batch(batch):
    if "image" in batch:
        batch["image"] = [img.convert("RGB") for img in batch["image"]]
    return batch

print("Loading and preparing dataset...")
dataset = load_dataset("dutta18/multi-domain-VQA-12K")

train_set, val_set = dataset['train'], dataset['validation']

train_set = train_set.map(prepend_prefix)
val_set = val_set.map(prepend_prefix)

train_set = train_set.with_transform(ensure_rgb_batch)
val_set = val_set.with_transform(ensure_rgb_batch)

# ### Import COT Data

with open('/home/aritrad/MOE-Directory/COT-Data-Multidomain-12K/COT-train-set-12K.pkl', 'rb') as file:
    COT_train = pickle.load(file)

with open('/home/aritrad/MOE-Directory/COT-Data-Multidomain-12K/COT-val-set-12K.pkl', 'rb') as file:
    COT_val = pickle.load(file)


# Replace the markdowns * and #:

COT_train = [ data.replace('*', '') for data in COT_train ]
COT_train = [ data.replace('#', '') for data in COT_train ]

COT_val = [ data.replace('*', '') for data in COT_val ]
COT_val = [ data.replace('#', '') for data in COT_val ]


assert len(train_set) == len(COT_train), "Length mismatch between dataset and COT list."
assert len(val_set) == len(COT_val), "Length mismatch between dataset and COT list."

print("Integrating COT data and filtering...")
# Use map with an index to replace the "answer" field
train_set = train_set.map(lambda example, idx: {"cot_answer": COT_train[idx].strip()}, with_indices=True, num_proc=8)
val_set = val_set.map(lambda example, idx: {"cot_answer": COT_val[idx].strip()}, with_indices=True, num_proc=8)
train_set = train_set.filter(lambda example: example["reasoning_type"] == "physical", num_proc=8)
val_set   = val_set.filter(lambda example: example["reasoning_type"] == "physical", num_proc=8)

# ### Prepare Dataloaders

from torch.utils.data import Dataset

class MultiDomain(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "image": self.data[idx]["image"],
            "question": self.data[idx]["question"],
            "cot_answer": self.data[idx]["cot_answer"]
        }


print("Creating PyTorch Datasets...")
train_dataset = MultiDomain(train_set)
val_dataset = MultiDomain(val_set)

# ### Prepare Dataloaders

SYSTEM_PROMPT = "You are a intelligent reasoning assistant who can generate step by step thought."

def collate(batch):

    prompt_texts = []
    full_texts = []
    images = []

    for example in batch:

        messages = [
            {
                "role": "system",
                "content":[{"type":"text","text":SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content":[
                    {"type":"image","image":example["image"]},
                    {"type":"text","text":example["question"]}
                ]
            }
        ]

        answer = example["cot_answer"]

        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        full_text = processor.apply_chat_template(
            messages + [{
                "role":"assistant",
                "content":[{"type":"text","text":answer}]
            }],
            tokenize=False
        )

        prompt_texts.append(prompt_text)
        full_texts.append(full_text)
        images.append([example["image"]])

    # encode prompt batch
    prompt_inputs = processor(
        text=prompt_texts,
        images=images,
        padding='longest',
        return_tensors="pt"
    )

    # encode full batch
    full_inputs = processor(
        text=full_texts,
        images=images,
        padding='longest',
        return_tensors="pt"
    )

    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]
    token_type_ids = full_inputs["token_type_ids"]
    pixel_values = full_inputs["pixel_values"]

    labels = input_ids.clone()

    # mask prompt tokens
    for i in range(len(batch)):
        prompt_len = prompt_inputs["attention_mask"][i].sum()
        labels[i, :prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "labels": labels
    }


# ### Model Loading

device = "cuda"

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

pretrained_id = "google/gemma-3-4b-it"

print(f"\nLoading model and processor: {pretrained_id}...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    pretrained_id,
    dtype=torch.bfloat16,
    low_cpu_mem_usage = True,
    attn_implementation="flash_attention_2",
    device_map='auto'
)

# Load processor. 
processor = AutoProcessor.from_pretrained(
    pretrained_id, 
    use_fast=True
)


processor.tokenizer.padding_side = "right"

# ### Calculate number of Params

def report_trainable_params():
    
    # Simple param report
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {trainable/1e6:.1f} M")

report_trainable_params()

# #### Apply Grad Checkpointing using: gradient_checkpointing_enable()

print("Enabling gradient checkpointing...")
model.gradient_checkpointing_enable()

# ### Create & Test Dataloader

batchSize_ = 2

print("Creating DataLoaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=batchSize_,
    collate_fn=collate,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batchSize_,
    collate_fn=collate,
    shuffle=False
)


# ### Validation Function

def do_validation():
    
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    model.train()
    torch.cuda.empty_cache()
    return avg_val_loss


# ### Training Hyperparams

from transformers import get_cosine_schedule_with_warmup

LR = 5e-5
epochs = 2
weight_decay = 0.01
gradient_accumulation_steps = 2

global_step = 0
best_val_loss = float("inf")

steps_per_epoch = len(train_loader) // gradient_accumulation_steps
total_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.02 * total_train_steps)         

print("Initializing optimizer and scheduler...")
optimizer = AdamW(model.parameters(), lr=LR, eps=1e-6, weight_decay=weight_decay)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_train_steps,
)


model.use_cache = False

# ### Train Loop

print("\nStarting training...")
for epoch in tqdm(range(epochs)):

    accumulated_loss = 0
    
    for idx, batch in enumerate(train_loader):
        
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
    
        loss.backward()
        accumulated_loss += loss.item()
        
        if (idx+1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            logger.info(f"[ Epoch {epoch+1} | idx: {idx} | Optim Step {global_step} | Train Loss: {loss.item():.4f} ]")

            if global_step % 125 == 0:
                avg_val_loss = do_validation()
                logger.info(f"Val Loss @ Optim step: {global_step} -> {avg_val_loss:.4f}\n")
            
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model.save_pretrained('./chkpts/gemma-3-4B-Ph-FT')
                    logger.info(f"***** ✅ Checkpoint Saved *****\n")

    logger.info(f"Epoch {epoch+1} completed. Avg loss: {accumulated_loss / len(train_loader):.4f}\n\n")
