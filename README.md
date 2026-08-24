# Asymmetric Null-Space Projection for Data-Free LoRA Merging

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A Data-Free, Two-Sided Asymmetric Projection Framework for Zero-Interference LoRA Integration in Large Language Models.**

## 📌 Overview

Combining multiple specialized Low-Rank Adaptations (LoRA) into a single base LLM typically causes **catastrophic forgetting** due to parameter interference in shared weight subspaces. This repository implements **Two-Sided Asymmetric Null-Space Projection (Asym-NSP)**:

| Property | Value |
|---|---|
| Master adapter degradation | **0%** (mathematically guaranteed) |
| Projection target | Both output ($B$) and input ($A$) manifolds |
| Compute complexity | $O(d \cdot r^2)$ — no data, no fine-tuning |
| Hardware requirement | CPU is sufficient |
| Output format | Standard HuggingFace PEFT adapter |

## 🚀 Quick Start

```bash
git clone https://github.com/Xachchchch/asym-nullspace-merge.git
cd asym-nullspace-merge
pip install -r requirements.txt
pip install -e .

# Mathematical validation
python scripts/01_synthetic_verify.py

# End-to-end demo on public HuggingFace adapters (~2 seconds on CPU)
python scripts/05_hf_merge_demo.py --preset qwen2.5-0.5b
```

## 📐 Mathematical Formulation

Given a **Master** LoRA $(B_m, A_m)$ and a **Plugin** LoRA $(B_p, A_p)$ with rank-$k$ SVD truncation:

$$U_k, \Sigma_k, V_k^T = \text{SVD}_k(B_m), \quad \hat{U}_k, \hat{\Sigma}_k, \hat{V}_k^T = \text{SVD}_k(A_m^T)$$

**Two-sided projection onto orthogonal complements:**

$$B_p' = \underbrace{(I - U_k U_k^T)}_{\text{left null-space}} B_p, \qquad A_p' = A_p \underbrace{(I - V_k V_k^T)}_{\text{right null-space}}$$

**Merged weight update (exact, zero approximation error):**

$$\Delta W_{\text{merged}} = \underbrace{B_m A_m}_{\text{Master — 100% preserved}} + \underbrace{B_p' A_p'}_{\text{Plugin — zero interference}}$$

**Standalone PEFT export** uses split-scaling to represent the sum as a single low-rank adapter:

$$B_{\text{merged}} = \begin{bmatrix} \sqrt{s_m}\, B_m & \sqrt{s_p}\, B_p' \end{bmatrix}, \quad A_{\text{merged}} = \begin{bmatrix} \sqrt{s_m}\, A_m \\ \sqrt{s_p}\, A_p' \end{bmatrix}$$

This identity holds with **zero approximation error**: $B_{\text{merged}} A_{\text{merged}} = s_m B_m A_m + s_p B_p' A_p'$.

## 🛠️ CLI Usage

### 1. Export as a Standalone PEFT Adapter

Merge plugin into master's null space and export standard `adapter_model.safetensors` + `adapter_config.json`:

```bash
python scripts/02_run_merge.py \
  --master path/to/master_lora \
  --plugins path/to/plugin_lora \
  --output_dir ./merged_peft_adapter \
  --export_adapter
```

Load immediately with HuggingFace PEFT:

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

### 3. End-to-End HuggingFace Demo

Run a full merge + inference test on public community adapters:

```bash
# Predefined preset: Qwen2.5-0.5B Math + Python (merges in ~2s on CPU)
python scripts/05_hf_merge_demo.py --preset qwen2.5-0.5b

# Custom adapters from HuggingFace Hub
python scripts/05_hf_merge_demo.py --preset custom \
  --base_model meta-llama/Meta-Llama-3-8B \
  --master user/llama3-math-lora \
  --plugin user/llama3-code-lora \
  --output_dir ./outputs/llama3_math_code_adapter

# Merge only, skip inference
python scripts/05_hf_merge_demo.py --preset qwen2.5-0.5b --skip_inference
```

### 4. Inspect & Validate Adapters

```bash
# Visualise subspace spectra and interference heatmaps
python scripts/04_visualize_spectra.py

# Create synthetic adapters and inspect merge quality
python scripts/06_create_and_inspect_demo.py

# Run benchmark evaluation (requires lm-eval)
python scripts/03_eval_benchmark.py --config configs/base_llama3.yaml
```

## 🐍 Python API

```python
from asym_nsp import AsymmetricNSPMerger, MergeConfig

config = MergeConfig(
    master_adapter_path="path/or/hub-id/master",
    plugin_adapter_paths=["path/or/hub-id/plugin"],
    energy_threshold=0.99,   # retain 99% of singular-value energy for projector basis
    project_left=True,       # project B matrices (output manifold)
    project_right=True,      # project A matrices (input manifold)
    device="cpu",
    dtype="float32",
)

merger = AsymmetricNSPMerger(config)

# Option A: export as standalone PEFT adapter
merger.export_as_peft_adapter(output_dir="./merged_adapter")

# Option B: get merged ΔW tensors and apply to a base model
merged_deltas = merger.merge_adapters()
merger.apply_and_save(
    base_model_path="meta-llama/Meta-Llama-3-8B",
    output_dir="./merged_model",
    merged_deltas=merged_deltas,
)
```

### Low-level projection API

```python
from asym_nsp import (
    compute_subspace_basis,
    project_left_null_space,
    project_right_null_space,
    project_two_sided_nsp,
    TwoSidedNullSpaceProjector,
)

# SVD basis with energy threshold (robust against zero matrices & threshold ≥ 1.0)
U_k, sv = compute_subspace_basis(B_master, energy_threshold=0.99, basis_side="left")

# Manual projection
B_proj = project_left_null_space(B_plugin, U_k)

# Full two-sided projection in one call
B_proj, A_proj = project_two_sided_nsp(
    B_plugin, A_plugin, B_master, A_master,
    energy_threshold=0.99,
)
```

## 📁 Repository Structure

```
asymmetric-lora-nsp/
├── src/asym_nsp/
│   ├── core.py          # SVD basis, left/right null-space projectors
│   ├── merger.py        # AsymmetricNSPMerger, MergeConfig, PEFT export
│   └── lora_loader.py   # HuggingFace & safetensors LoRA loader
├── scripts/
│   ├── 01_synthetic_verify.py      # Mathematical correctness validation
│   ├── 02_run_merge.py             # Main CLI entry point
│   ├── 03_eval_benchmark.py        # lm-eval harness integration
│   ├── 04_visualize_spectra.py     # Subspace overlap & spectra plots
│   ├── 05_hf_merge_demo.py         # End-to-end HuggingFace demo
│   └── 06_create_and_inspect_demo.py
├── configs/
│   ├── base_llama3.yaml
│   └── merge_reasoning_code.yaml
├── tests/
│   ├── test_orthogonality.py       # Verifies U_k^T B_p' ≈ 0
│   └── test_merge_shapes.py        # Tensor shape contracts
└── pyproject.toml
```

## 📊 Verification

```bash
# Synthetic orthogonality test (no GPU / pretrained weights needed)
python scripts/01_synthetic_verify.py

# Visualise before/after projection spectra
python scripts/04_visualize_spectra.py
```

After Two-Sided Asymmetric NSP, the subspace overlap $\|U_m^T B_p'\|_F \approx 0$ and the cosine of principal angles between master and plugin column spaces drops to strictly zero — visible in the generated heatmap and principal-angle spectrum plots.

## ⚙️ Configuration

```yaml
# configs/merge_reasoning_code.yaml
master_adapter_path: "path/to/master"
plugin_adapter_paths:
  - "path/to/plugin"
output_dir: "./outputs/merged"
energy_threshold: 0.99   # SVD energy cutoff for projector rank
project_left: true
project_right: true
export_adapter: true
lora_alpha: 32.0
device: "cpu"
dtype: "float32"
```

## 📄 Citation

```bibtex
@article{blbulyan2026asymmetric,
  title   = {Asymmetric Null-Space Projection for Data-Free Zero-Interference LoRA Merging},
  author  = {Blbulyan, Khachik},
  year    = {2026},
  url     = {https://github.com/Xachchchch/asym-nullspace-merge}
}
```
