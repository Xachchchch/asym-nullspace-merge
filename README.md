# Asymmetric Null-Space Projection for Data-Free LoRA Merging

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A Data-Free, Two-Sided Asymmetric Projection Framework for Zero-Interference LoRA Integration in Large Language Models.**

## 📌 Overview
Combining multiple specialized Low-Rank Adaptations (LoRA) into a single Base LLM often induces **Catastrophic Forgetting** due to parameter interference. This repository implements **Two-Sided Asymmetric Null-Space Projection (NSP)**:
- **Asymmetric Master-Plugin Hierarchy:** Master adapter weights remain mathematically invariant (0% degradation).
- **Two-Sided Latent Decoupling:** Projections applied to both Output ($B$) and Input ($A$) manifolds.
- **Data-Free & CPU-Efficient:** Operates directly on low-rank weight tensors with $O(d \cdot r^2)$ computational complexity.

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/Xachchchch/asym-nullspace-merge.git
cd asym-nullspace-merge

# Install dependencies
pip install -r requirements.txt

# Run mathematical validation tests
python -m unittest discover -s tests -p "test_*.py" -v
```

## 🛠️ CLI Usage & Adapter Export

### 1. Export as a Standalone HuggingFace PEFT Adapter (~30–50MB)
Merge plugin into master's null space and export standard `adapter_model.safetensors` + `adapter_config.json`:
```bash
python scripts/02_run_merge.py \
  --master path/to/master_lora \
  --plugins path/to/plugin_lora \
  --output_dir ./merged_peft_adapter \
  --export_adapter
```

Load the merged adapter immediately with HuggingFace PEFT:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = PeftModel.from_pretrained(model, "./merged_peft_adapter")
```

### 2. Merge Directly into Base Model Weights
```bash
python scripts/02_run_merge.py \
  --config configs/merge_reasoning_code.yaml \
  --base_model meta-llama/Meta-Llama-3-8B \
  --output_dir ./merged_llama3_model
```

### 3. End-to-End Real HuggingFace LoRA Merge & Inference Demo
Run a full end-to-end test on public adapters (e.g. Qwen2.5-0.5B Math + Python):
```bash
# Run predefined preset (merges in ~2 seconds and runs dual-task test prompts)
python scripts/05_hf_merge_demo.py --preset qwen2.5-0.5b

# Or merge custom HuggingFace Hub adapters
python scripts/05_hf_merge_demo.py --preset custom \
  --base_model meta-llama/Meta-Llama-3-8B \
  --master user/llama3-math-lora \
  --plugin user/llama3-code-lora \
  --output_dir ./outputs/llama3_math_code_adapter
```

## 📐 Mathematical Formulation

Given a Master LoRA $(B_m, A_m)$ and a Plugin LoRA $(B_p, A_p)$:

$$B_p' = (I - U_k U_k^T) B_p, \quad A_p' = A_p (I - V_k V_k^T)$$

$$\Delta W_{\text{merged}} = B_m A_m + B_p' A_p'$$

## 📊 Visualizations & Empirical Verification

Run the visualization script to generate empirical verification plots:

```bash
python scripts/04_visualize_spectra.py
```

### 1. Subspace Parameter Interference Heatmap
Before projection, Master and Plugin adapters share significant subspace overlap ($|U_m^T B_p|$), leading to parameter corruption. After Two-Sided Asymmetric NSP, the overlap is strictly eliminated ($U_m^T B_p' \approx 0$):

![Subspace Overlap Heatmap](assets/subspace_overlap_heatmap.png)

### 2. Principal Angles Spectrum
Cosine of principal angles between the Master and Plugin column spaces drops from high interference ($\cos \theta_i \gg 0$) straight down to strict orthogonality ($\cos \theta_i \equiv 0$):

![Principal Angles Spectrum](assets/principal_angles_spectrum.png)

## 📄 Citation

```bibtex
@article{blbulyan2026asymmetric,
  title={Asymmetric Null-Space Projection for Data-Free LoRA Merging},
  author={Blbulyan, Khachik},
  year={2026}
}
```
