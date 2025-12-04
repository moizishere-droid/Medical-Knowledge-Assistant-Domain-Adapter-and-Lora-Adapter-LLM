# 🏥 Medical Knowledge Assistant (Colab Version)

A **LoRA fine-tuned GPT-2 model** for medical Q&A. This project demonstrates **domain adaptation** and **instruction tuning** for generating accurate medical responses, optimized for **Google Colab free GPU**.

---

## 🚀 Highlights

* **Domain Adaptation + Instruction Tuning**: GPT-2 trained first on raw medical text, then fine-tuned on medical Q&A.
* **LoRA Fine-Tuning**: Parameter-efficient with only **~0.25% trainable parameters**.
* **Instruction-Following Format**: Structured prompts for clear, professional answers.
* **8-bit Quantization**: Efficient inference on consumer GPUs.
* **Colab-Friendly**: Small model and dataset, suitable for free GPU training.
* **Reusable Dataset**: Includes `medical_corpus.json` (text) and `medical_qa.json` (Q&A).
* **Deployed Model**: [Access the domain-adapted model on Hugging Face](https://huggingface.co/Abdulmoiz123/Medical_domain_adaptation)

---

## 📊 Training Configuration

| Parameter             | Value                       |
| --------------------- | --------------------------- |
| Base Model            | GPT-2 (124M parameters)     |
| Dataset               | Custom medical corpus + Q&A |
| Training Samples      | 20+ medical Q&A examples    |
| Batch Size            | 2                           |
| Gradient Accumulation | 4                           |
| Learning Rate         | 2e-4                        |
| Epochs                | 1                           |
| Max Sequence Length   | 256 tokens                  |
| Optimizer             | AdamW                       |
| Precision             | FP16 (Mixed Precision)      |

### LoRA Configuration

| Parameter            | Value         |
| -------------------- | ------------- |
| LoRA Rank (r)        | 8             |
| LoRA Alpha           | 16            |
| Target Modules       | c_attn        |
| LoRA Dropout         | 0.05          |
| Bias                 | none          |
| Trainable Parameters | ~300K (0.25%) |

---

### Usage (Load Model Directly from Hugging Face)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "Abdulmoiz123/Medical_domain_adaptation"  # Hugging Face link
BASE_MODEL = "gpt2"

print("📥 Loading domain-adapted medical model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

def generate_response(question, max_new_tokens=150, temperature=0.7, top_p=0.9):
    prompt = f"""Below is an instruction that describes a medical question. Write a precise and professional response.

### Instruction:
Answer the following medical question.

### Input:
{question}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    return response

# Example
question = "What are the symptoms of hypertension?"
print("💬 Response:", generate_response(question))
```

---

## 🔄 Reproducibility

* LoRA adapters saved separately (~300K params)
* Fixed random seed for deterministic results
* Full config files saved for replication

---

## 💡 Use Cases

* Medical Q&A assistant
* Educational healthcare tool
* Instruction-following AI for clinical info
* Lightweight Colab-friendly model for prototyping

---

## 📈 Advantages of LoRA Fine-Tuning

* Memory-efficient: Only ~0.25% of parameters trained
* Fast training and inference
* Easy deployment & modular adapter switching
* Colab GPU compatible

Do you want me to do that?
