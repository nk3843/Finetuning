# Finetuning - Yoda Speak LLM

This project demonstrates **fine-tuning a Large Language Model (LLM)** to speak like Yoda from Star Wars using **LoRA (Low-Rank Adaptation)** for efficient parameter training.

## 📋 Overview

The project fine-tunes Microsoft's **Phi-3-mini-4k-instruct** model to transform normal English sentences into Yoda-style speech patterns. Using modern techniques like **4-bit quantization** and **LoRA adapters**, we can train this model efficiently on consumer GPUs while only modifying **0.33% of the total parameters**.

### Key Features

- ✅ **Memory Efficient**: Uses 4-bit quantization via BitsAndBytes
- ✅ **Parameter Efficient**: Only trains 12.58M out of 3.8B parameters using LoRA
- ✅ **Production Ready**: Includes complete training pipeline and inference functions
- ✅ **Well Documented**: Comprehensive comments explaining each code section

## 📂 Project Structure

```
Finetuning/
├── README.md                          # This file
├── Yoda_speak.ipynb                   # Original Jupyter notebook
└── yoda_finetuning_documented.py      # Fully documented Python script
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -U datasets bitsandbytes trl torch transformers accelerate peft
```

### Running the Code

The documented Python script is organized into 12 clear sections:

1. **Imports** - Essential libraries for training
2. **Quantization Config** - 4-bit compression settings
3. **Load Base Model** - Load Phi-3 with quantization
4. **LoRA Setup** - Configure efficient fine-tuning
5. **Dataset Loading** - Load Yoda sentences dataset
6. **Tokenizer Setup** - Configure text processing
7. **Training Config** - Optimize memory and learning parameters
8. **Initialize Trainer** - Create SFT trainer instance
9. **Train Model** - Run the fine-tuning process
10. **Inference Functions** - Helper functions for generation
11. **Test Model** - Validate the trained model
12. **Save/Upload** - Save locally or push to HuggingFace

### Example Usage

```python
from yoda_finetuning_documented import gen_prompt, generate, model, tokenizer

# Generate Yoda-style text
sentence = "The Force is strong in you!"
prompt = gen_prompt(tokenizer, sentence)
output = generate(model, tokenizer, prompt)
print(output)
# Output: "Strong in you, the Force is! Yes, hrrrm."
```

## 🔧 Technical Details

### Model Architecture

- **Base Model**: microsoft/Phi-3-mini-4k-instruct
- **Total Parameters**: 3.8B
- **Trainable Parameters**: 12.58M (0.33%)
- **Quantization**: 4-bit NormalFloat4 (nf4)
- **LoRA Rank**: 8
- **LoRA Alpha**: 16

### Training Configuration

- **Epochs**: 10
- **Batch Size**: 16 (with auto-find)
- **Learning Rate**: 3e-4
- **Optimizer**: paged_adamw_8bit
- **Max Sequence Length**: 64 tokens
- **Dataset Packing**: Enabled
- **Gradient Checkpointing**: Enabled

### Dataset

- **Source**: `dvgodoy/yoda_sentences` from HuggingFace
- **Size**: 720 sentence pairs
- **Format**: Normal English → Yoda-style translation

## 📊 Results

After 10 epochs of training (~44 minutes on T4 GPU):
- Initial training loss: 2.85
- Final training loss: 0.24
- Memory footprint: ~2.6 GB

### Sample Outputs

| Input | Output |
|-------|--------|
| "The Force is strong in you!" | "Strong in you, the Force is! Yes, hrrrm." |
| "The birch canoe slid on the smooth planks." | "On the smooth planks, the birch canoe slid. Yes, hrrrm." |

## 🎓 Learning Resources

### Key Concepts Explained in Comments

1. **4-bit Quantization**: Reduces model size by 75% while maintaining performance
2. **LoRA (Low-Rank Adaptation)**: Adds small trainable matrices instead of fine-tuning all weights
3. **Gradient Checkpointing**: Trades computation for memory savings
4. **Dataset Packing**: Combines short sequences to minimize padding waste
5. **Chat Templates**: Formats conversations for instruction-tuned models

### Code Comments Structure

Each section includes:
- 🎯 **Purpose**: What this code block does
- ⚙️ **Parameters**: Explanation of key settings
- 💡 **Why**: Rationale behind design choices
- 🔍 **Example**: Sample inputs/outputs where relevant

## 🤝 Contributing

This is a learning project demonstrating LLM fine-tuning best practices. Feel free to:
- Experiment with different LoRA ranks
- Try different base models
- Create adapters for other speaking styles

## 📝 License

This project is for educational purposes. The base model follows Microsoft's Phi-3 license.

## 🙏 Acknowledgments

- **Dataset**: [dvgodoy/yoda_sentences](https://huggingface.co/datasets/dvgodoy/yoda_sentences)
- **Base Model**: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Libraries**: HuggingFace Transformers, PEFT, TRL, BitsAndBytes
