"""
Lab: Preference Fine-Tuning with DPO (Direct Preference Optimization) and QLoRA
Reference: https://github.com/handsOnLLM/Hands-On-Large-Language-Models

This script implements preference data formulation, quantization configuration (NF4),
SFT LoRA merging, PEFT/LoRA configuration for DPO, DPO training configuration,
and final weight merging.
"""

from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import DPOConfig, DPOTrainer
import torch


# ==========================================
# 1. Data Preprocessing (Preference Data Formulating)
# ==========================================
def format_prompt(example):
    """Format the prompt using the <|user|> template TinyLlama is using."""
    system = "<|system|>\n" + example['system'] + "</s>\n"
    prompt = "<|user|>\n" + example['input'] + "</s>\n<|assistant|>\n"
    chosen = example['chosen'] + "</s>\n"
    rejected = example['rejected'] + "</s>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def prepare_dpo_dataset():
    print(">>> Loading and filtering DPO dataset...")
    dpo_dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
    dpo_dataset = dpo_dataset.filter(
        lambda r:
            r["status"] != "tie" and
            r["chosen_score"] >= 8 and
            not r["in_gsm8k_train"]
    )
    # Apply formatting and keep only formatted columns (prompt, chosen, rejected)
    dpo_dataset = dpo_dataset.map(format_prompt, remove_columns=dpo_dataset.column_names)
    print(f">>> Dataset prepared with {len(dpo_dataset)} examples (downsampled from original ~13,000).")
    return dpo_dataset


# ==========================================
# 2. Model & Quantization Configuration
# ==========================================
def run_dpo_pipeline():
    # Load dataset
    dpo_dataset = prepare_dpo_dataset()

    # 4-bit quantization configuration - Q in QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                 # Use 4-bit precision model loading
        bnb_4bit_quant_type="nf4",         # Quantization type
        bnb_4bit_compute_dtype=torch.float16,  # Compute dtype (float16)
        bnb_4bit_use_double_quant=True,     # Apply nested quantization
    )

    print(">>> Loading base SFT model and tokenizer...")
    # NOTE: In practice, replace "TinyLlama-1.1B-qlora" with your actual local SFT directory
    model_name = "TinyLlama-1.1B-qlora"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model with QLoRA configuration
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    # Merge LoRA and base model before DPO training
    print(">>> Merging base model and SFT LoRA adapter...")
    merged_model = model.merge_and_unload()

    # ==========================================
    # 3. LoRA Configuration for DPO
    # ==========================================
    print(">>> Setting up LoRA configuration for DPO...")
    peft_config = LoraConfig(
        lora_alpha=32,      # LoRA Scaling
        lora_dropout=0.1,   # Dropout for LoRA Layers
        r=64,               # Rank
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            'k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'
        ]
    )

    # Prepare model for kbit training
    merged_model = prepare_model_for_kbit_training(merged_model)
    dpo_model = get_peft_model(merged_model, peft_config)

    # ==========================================
    # 4. Training Configuration
    # ==========================================
    print(">>> Configuring DPO training arguments...")
    output_dir = "./results"
    
    training_arguments = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        max_steps=200,                  # Short run for illustration
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1
    )

    # ==========================================
    # 5. DPO Training execution
    # ==========================================
    print(">>> Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        dpo_model,
        args=training_arguments,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,                       # DPO temperature parameter
        max_prompt_length=512,
        max_length=512,
    )

    print(">>> Starting DPO fine-tuning...")
    dpo_trainer.train()

    # Save final DPO adapter
    dpo_output_dir = "TinyLlama-1.1B-dpo-qlora"
    print(f">> Saving DPO adapter to {dpo_output_dir}...")
    dpo_trainer.model.save_pretrained(dpo_output_dir)


# ==========================================
# 6. Merge Weights (SFT + DPO)
# ==========================================
def merge_sft_and_dpo_weights():
    print(">>> Final step: Merging SFT and DPO adapters...")
    
    # 1. Load SFT model and merge
    print(">>> 1. Loading and merging base model + SFT LoRA...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        "TinyLlama-1.1B-qlora",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    sft_model = model.merge_and_unload()

    # 2. Load DPO model and merge on top of SFT model
    print(">>> 2. Loading and merging SFT model + DPO LoRA...")
    dpo_model = PeftModel.from_pretrained(
        sft_model,
        "TinyLlama-1.1B-dpo-qlora",
        device_map="auto",
    )
    final_merged_model = dpo_model.merge_and_unload()
    
    print(">>> Weights successfully merged!")
    return final_merged_model


if __name__ == "__main__":
    # Note: Running this script requires access to a GPU and the pre-trained 'TinyLlama-1.1B-qlora' model.
    # It serves as a complete executable blueprint for the Preference Fine-Tuning Lab.
    print("=== DPO Fine-Tuning Lab Script ===")
    print("This script is ready to run. In a real GPU sandbox, invoke it using:")
    print("  python dpo_lab.py")
