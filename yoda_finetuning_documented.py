"""
Yoda-Style Language Model Fine-tuning
=====================================
This script fine-tunes the Microsoft Phi-3-mini model to speak like Yoda using 
LoRA (Low-Rank Adaptation) for efficient parameter training.

Author: Nikhil Kumar
Project: Finetuning LLMs with custom speaking styles
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
# Import essential libraries for model training and fine-tuning
import os
import torch
from datasets import load_dataset  # For loading training datasets from HuggingFace
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training  # PEFT for LoRA fine-tuning
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # Core transformers components
from trl import SFTConfig, SFTTrainer  # Supervised Fine-Tuning trainer from TRL library
from contextlib import nullcontext
from huggingface_hub import login  # For pushing model to HuggingFace Hub


# ============================================================================
# SECTION 2: MODEL QUANTIZATION CONFIGURATION
# ============================================================================
# Configure 4-bit quantization using BitsAndBytes to reduce memory footprint
# This allows training large models on consumer GPUs by compressing weights
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,  # Enable 4-bit quantization for memory efficiency
   bnb_4bit_quant_type="nf4",  # Use NormalFloat4 quantization type for better performance
   bnb_4bit_use_double_quant=True,  # Apply double quantization for additional compression
   bnb_4bit_compute_dtype=torch.float32  # Use float32 for computation to maintain numerical stability
)


# ============================================================================
# SECTION 3: LOAD BASE MODEL
# ============================================================================
# Load the pre-trained Phi-3 model with quantization for efficient training
repo_id = 'microsoft/Phi-3-mini-4k-instruct'  # Base model identifier on HuggingFace
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    device_map="cuda:0",  # Map model to GPU for training
    quantization_config=bnb_config  # Apply the 4-bit quantization config
)

# Check model memory footprint after quantization (in MB)
print(f"Model memory footprint: {model.get_memory_footprint()/1e6} MB")


# ============================================================================
# SECTION 4: PREPARE MODEL FOR LoRA FINE-TUNING
# ============================================================================
# Prepare the quantized model for k-bit training (enables gradient computation on quantized weights)
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation) parameters
# LoRA adds small trainable matrices to specific layers instead of training all parameters
config = LoraConfig(
    r=8,  # Rank of the low-rank decomposition (lower = fewer trainable parameters)
    lora_alpha=16,  # Scaling factor for LoRA weights (typically 2*r)
    bias="none",  # Don't train bias terms (modifying biases can alter base model behavior)
    lora_dropout=0.05,  # Dropout rate for LoRA layers to prevent overfitting
    task_type="CAUSAL_LM",  # Task type: Causal Language Modeling
    # Target modules: specify which model layers to apply LoRA to
    # These are attention and MLP projection layers in Phi-3
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)

# Wrap the model with LoRA adapters
model = get_peft_model(model, config)

# Check updated memory footprint and trainable parameters
print(f"Model with LoRA adapters: {model.get_memory_footprint()/1e6} MB")
train_p, tot_p = model.get_nb_trainable_parameters()
print(f'Trainable parameters: {train_p/1e6:.2f}M')
print(f'Total parameters: {tot_p/1e6:.2f}M')
print(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')


# ============================================================================
# SECTION 5: LOAD AND PREPARE DATASET
# ============================================================================
# Load the Yoda sentences dataset from HuggingFace
# This dataset contains normal sentences and their Yoda-style translations
dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
print(f"Dataset: {dataset}")
print(f"Sample: {dataset[0]}")

# Dataset formatting function
# Converts prompt/completion format to conversational format required by chat models
def format_dataset(examples):
    """
    Convert dataset from prompt/completion pairs to conversational message format.
    Each example is transformed into a user message (prompt) and assistant message (completion).
    
    Args:
        examples: Dictionary containing 'prompt' and 'completion' keys
        
    Returns:
        Dictionary with 'messages' key containing formatted conversation
    """
    if isinstance(examples["prompt"], list):
        # Handle batch processing (multiple examples at once)
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},  # User's input sentence
                {"role": "assistant", "content": examples["completion"][i]},  # Yoda-style response
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        # Handle single example
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

# Transform dataset: rename columns and apply formatting
dataset = dataset.rename_column("sentence", "prompt")  # Normal sentence becomes the prompt
dataset = dataset.rename_column("translation_extra", "completion")  # Yoda translation becomes completion
dataset = dataset.map(format_dataset)  # Apply formatting function to all examples
dataset = dataset.remove_columns(['prompt', 'completion', 'translation'])  # Remove old columns

# Verify the formatted data structure
messages = dataset[0]['messages']
print(f"Formatted message sample: {messages}")


# ============================================================================
# SECTION 6: TOKENIZER SETUP
# ============================================================================
# Load the tokenizer for the Phi-3 model
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Configure padding tokens (required for batch training)
tokenizer.pad_token = tokenizer.unk_token  # Use unknown token as padding
tokenizer.pad_token_id = tokenizer.unk_token_id

# Display the chat template used by Phi-3
print(f"Chat template: {tokenizer.chat_template}")

# Test tokenization with the chat template
print("Tokenized example:")
print(tokenizer.apply_chat_template(messages, tokenize=False))


# ============================================================================
# SECTION 7: TRAINING CONFIGURATION
# ============================================================================
# Configure the Supervised Fine-Tuning (SFT) trainer with optimized settings
sft_config = SFTConfig(
    ## GROUP 1: Memory optimization settings
    # These parameters maximize GPU RAM utilization
    gradient_checkpointing=True,  # Trade computation for memory by recomputing activations
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Required for newer PyTorch versions
    gradient_accumulation_steps=1,  # Update weights every 1 micro-batch
    per_device_train_batch_size=16,  # Initial micro-batch size
    auto_find_batch_size=True,  # Automatically reduce batch size if OOM occurs

    ## GROUP 2: Dataset-related settings
    max_length=64,  # Maximum sequence length for training
    packing=True,  # Pack multiple short sequences into one to minimize padding
    packing_strategy='wrapped',  # Packing strategy that approximates original behavior

    ## GROUP 3: Training hyperparameters
    num_train_epochs=10,  # Number of complete passes through the dataset
    learning_rate=3e-4,  # Learning rate for optimizer
    optim='paged_adamw_8bit',  # 8-bit Adam optimizer for memory efficiency

    ## GROUP 4: Logging and output settings
    logging_steps=10,  # Log metrics every 10 steps
    logging_dir='./logs',  # Directory for TensorBoard logs
    output_dir='./phi3-mini-yoda-adapter',  # Directory to save model checkpoints
    report_to='none',  # Disable reporting to external services

    # Use bfloat16 precision only if GPU supports it (better numerical stability than fp16)
    bf16=torch.cuda.is_bf16_supported(including_emulation=False)
)


# ============================================================================
# SECTION 8: INITIALIZE TRAINER
# ============================================================================
# Create the SFT trainer instance with model, tokenizer, config, and dataset
trainer = SFTTrainer(
    model=model,  # LoRA-wrapped model
    processing_class=tokenizer,  # Tokenizer for processing text
    args=sft_config,  # Training configuration
    train_dataset=dataset,  # Formatted Yoda dataset
)

# Inspect a training batch to verify data pipeline
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print(f"Batch input_ids: {batch['input_ids'][0]}")


# ============================================================================
# SECTION 9: TRAIN THE MODEL
# ============================================================================
# Start the fine-tuning process
# This will train only the LoRA adapter weights (0.33% of total parameters)
print("Starting training...")
trainer.train()


# ============================================================================
# SECTION 10: INFERENCE HELPER FUNCTIONS
# ============================================================================
def gen_prompt(tokenizer, sentence):
    """
    Generate a properly formatted prompt for the chat model.
    
    Args:
        tokenizer: The tokenizer instance
        sentence: Input sentence to convert to Yoda-style
        
    Returns:
        Formatted prompt string ready for generation
    """
    converted_sample = [
        {"role": "user", "content": sentence},  # User's input
    ]
    # Apply chat template and add generation prompt (tells model to generate assistant response)
    prompt = tokenizer.apply_chat_template(
        converted_sample,
        tokenize=False,  # Return string, not token IDs
        add_generation_prompt=True  # Add the "<|assistant|>" tag
    )
    return prompt


def generate(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    """
    Generate text from the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        skip_special_tokens: Whether to remove special tokens from output
        
    Returns:
        Generated text string
    """
    # Tokenize the input prompt and move to GPU
    tokenized_input = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    model.eval()  # Set model to evaluation mode
    
    # Use mixed precision if model was trained with it
    ctx = torch.autocast(device_type=model.device.type, dtype=model.dtype) \
          if model.dtype in [torch.float16, torch.bfloat16] else nullcontext()
    
    with ctx:
        # Generate text using the model
        generation_output = model.generate(
            **tokenized_input,
            eos_token_id=tokenizer.eos_token_id,  # Stop at end-of-sequence token
            max_new_tokens=max_new_tokens  # Limit output length
        )

    # Decode token IDs back to text
    output = tokenizer.batch_decode(generation_output, skip_special_tokens=skip_special_tokens)
    return output[0]


# ============================================================================
# SECTION 11: TEST THE MODEL
# ============================================================================
# Test the fine-tuned model with a sample sentence
sentence = 'The Force is strong in you!'
prompt = gen_prompt(tokenizer, sentence)
print(f"Input prompt:\n{prompt}")

# Generate Yoda-style response
output = generate(model, tokenizer, prompt)
print(f"Generated output:\n{output}")


# ============================================================================
# SECTION 12: SAVE AND UPLOAD MODEL
# ============================================================================
# Save the LoRA adapter locally
trainer.save_model('local-phi3-mini-yoda-adapter')
print("Model saved locally to: local-phi3-mini-yoda-adapter")

# Optional: Upload to HuggingFace Hub for sharing
# login()  # Authenticate with HuggingFace
# trainer.push_to_hub()  # Push model to your HuggingFace account
# print("Model pushed to HuggingFace Hub!")
