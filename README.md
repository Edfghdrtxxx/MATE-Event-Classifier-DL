# MATE-Event-Classifier-DL

> **A Physics-Informed Deep Learning Framework for Automated Particle Identification in Time Projection Chambers**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

MATE-Event-Classifier-DL is a deep learning framework designed for the **MATE (Multi-purpose Active-target Time projection chamber for nuclear astrophysical and Exotic beam experiments)** at the Institute of Modern Physics, CAS.

This project addresses the challenge of **automated particle identification** in medium/low-energy nuclear physics experiments. Rather than treating the detector as a black-box image source, we employ a **Physics-Informed approach** that explicitly integrates domain knowledge (Moment of Inertia Tensors) into the model architecture.

**Key Philosophy:** *Interpretability and reliability over raw accuracy.* The system is designed so domain experts can audit model decisions.

---

## Key Features

### ⚛️ Physics-Informed Architecture
- Explicit injection of **physical state vectors** (Moment of Inertia Tensor: I_xx, I_yy, I_xy = I_yx ; Total Charge Deposition) into the model
- Physics features guide attention via **Cross-Attention** and **Gated Fusion** mechanisms
- Ensures model predictions are grounded in physical constraints

### 🔍 Real Attention-Based XAI (Explainable AI)
- Transparent decision-making via **real attention weight extraction** (not synthetic visualizations)
- Spatial attention heatmaps enable physicists to verify model focus
- Systematic **error analysis pipelines** to characterize failure modes

### 🧠 Hybrid CNN-ViT Architectures
| Model Variant | Description |
|---------------|-------------|
| **V3 ResNet-18** | Baseline CNN classifier adapted for 80×48 TPC images |
| **V4 CrossAttention** | ResNet-18 + Physics-guided spatial attention |
| **V4 GatedFusion** | ResNet-18 + Learnable physics-image fusion gates |
| **V5 ViT** | Vision Transformer with `timm` pretrained backbone |
| **V5 ViT+CrossAttention** | ViT + Physics-informed cross-attention |
| **V5 ViT+GatedFusion** | ViT + Gated physics-image fusion |

### 💻 Simulation-Based Training (Sim-to-Real Pipeline)
- Trained on high-fidelity **Geant4-based Monte Carlo simulations** (MATESIM)
- Addresses labeled data scarcity in nuclear physics
- Domain adaptation for real experimental data is under active development

---

## Architecture

![Architecture Schematic](assets/images/workflow_schematic.png)

*Physics-informed hybrid architecture combining CNN/ViT backbones with cross-attention mechanism.*

---

## Data Format

| Property | Value |
|----------|-------|
| **Image Shape** | `(80, 48, 2)` → `(N, 2, 80, 48)` for PyTorch |
| **Projection Plane** | Y-Z plane (beam along Z-axis) |
| **Channel 0** | Charge Deposition (energy) |
| **Channel 1** | Drift Time Proxy (X-coordinate) |
| **Detector Dimensions** | Y: [-150, 150] mm, Z: [0, 300] mm, X: [-100, 100] mm |

---

## Performance & Reliability

The framework supports multiple classification tasks:

| Task | Classes | Best Accuracy |
|------|---------|---------------|
| Binary (³He vs ⁴He) | 2 | ~99%+ |
| Binary (¹²C vs ¹³C) | 2 | ~99%+ |
| Proton-Deuteron-Triton | 3 | ~97%+ |
| 5-Class (p, d, t, ³He, α) | 5 | ~95%+ |

![Performance Comparison](assets/images/performance_comparison.png)

*Comparison of model accuracy with and without Physics-Informed constraints.*

![Attention Visualization](assets/images/attention_visualization.png)

*XAI Visualization: The model correctly focuses on the particle track (high energy deposition area).*

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Edfghdrtxxx/MATE-Event-Classifier-DL.git
cd MATE-Event-Classifier-DL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare Data
Convert simulated ROOT files to HDF5 format:
```bash
python scripts/utils/root_to_hdf5_converter_v3.py --input <file>.root --output <file>.h5 --tree_name cbmsim
```

### 2. Train a Model
```bash
# V4 CrossAttention (Physics-Informed)
python scripts/AO_training/V4_CrossAttention_5Class.py --mode modular --epochs 100

# V5 ViT + Gated Fusion
python scripts/AO_training/V5_ViTGatedFusion_5Class.py --mode modular --epochs 50
```

### 3. Evaluate
```bash
python scripts/evaluation/BinaryResNet_V3_comprehensive_analysis.py
```

---

## Project Structure

```
MATE-Event-Classifier-DL/
├── scripts/
│   ├── AO_training/          # 28+ training scripts (all support dual-mode data loading)
│   │   ├── V3_*.py           # Baseline ResNet classifiers
│   │   ├── V4_*.py           # Physics-Informed CrossAttention & GatedFusion
│   │   ├── V5_*.py           # Vision Transformer variants
│   │   └── hyperparameter_sweep_*.py  # Automated hyperparameter search
│   ├── data/                 # Unified data loading module
│   │   ├── unified_tpc_dataset.py    # TPCDatasetConfig + create_tpc_dataloaders API
│   │   └── tpc_config.py     # Configuration dataclass
│   ├── models/               # Model architectures
│   │   ├── v4_cross_attention.py
│   │   ├── base_model.py
│   │   └── backbones/
│   ├── evaluation/           # Performance analysis & visualization
│   │   ├── error_analysis.py
│   │   ├── model_evaluator.py
│   │   └── report_generator.py
│   ├── visualization/        # XAI and attention heatmap generation
│   │   ├── generate_attention_heatmap_example.py
│   │   └── visualize_*.py
│   └── utils/                # Data conversion & physics features
│       ├── root_to_hdf5_converter_v3.py
│       └── physics_features.py
├── dataset/HDF5_Form/        # Standardized training data
├── outputs/                  # Timestamped experiment outputs
│   ├── V4_CrossAttention_*/  # Model weights, logs, attention files
│   └── *.md                  # Analysis reports
├── docs/                     # Documentation
├── GEMINI.md                 # Project context for AI assistants
└── requirements.txt
```

---

## Model Architecture Details

### Physics Features (Moment of Inertia Tensor)
```
I_xx : Second moment about X-axis
I_yy : Second moment about Y-axis  
I_xy : Cross-correlation term
Eigen_Ratio : λ_max / λ_min (elongation measure)
```

These features encode the **geometric shape** of particle tracks, providing physics-based priors that guide attention.

### Training Configuration (Default)
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR / ReduceLROnPlateau |
| Loss | CrossEntropyLoss (with optional class weights) |
| Data Split | 80% train, 20% validation |
| Batch Size | 64-128 |

---

## Roadmap & Future Work

- [x] **Phase 1:** Physics-Informed Architecture Design & Simulation Training
- [x] **Phase 2:** Multi-class classification (5-class, isotope pairs)
- [x] **Phase 3:** Dual-mode data loading (modular + legacy support)
- [x] **Phase 4:** Attention-based XAI visualization pipeline
- [ ] **Phase 5:** Sim-to-Real Domain Adaptation (UDA/DANN) — *In Progress*
- [ ] **Phase 6:** Deployment for real MATE experimental data
- [ ] **Phase 7:** Excitation spectrum reconstruction

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hu2024mate,
  author = {Hu, Zhiheng},
  title = {MATE-Event-Classifier-DL: Physics-Informed Deep Learning for TPC Particle Identification},
  year = {2024},
  url = {https://github.com/Edfghdrtxxx/MATE-Event-Classifier-DL}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Institute of Modern Physics, CAS** for MATE detector data and MATESIM simulation framework
- **RIKEN & Osaka University RCNP** for collaborative research opportunities
- The PyTorch and `timm` communities for excellent deep learning tools



