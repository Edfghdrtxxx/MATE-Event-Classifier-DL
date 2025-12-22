# MATE-Event-Classifier-DL

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## Overview

**MATE-Event-Classifier-DL** is a robust deep learning framework designed for the **MATE (Multi-purpose Active-target Time projection chamber for nuclear astrophysical and exotic beam Experiments)**. It addresses the critical challenge of **automated particle identification** in medium/low-energy nuclear physics experiments using physics-informed deep learning.

This project implements a **Physics-Informed Hybrid Architecture** that fuses MATE sensor data with physical state parameters (Moment of Inertia Tensor). The model is currently trained on **high-fidelity Monte Carlo simulations** (MATESIM based on Geant4) to establish a baseline for identifying rare isotopes (3He vs 4He/13C vs 14C) in complex experimental environments.

## Key Features

*   **⚛️ Physics-Informed Deep Learning**:
    *   Explicit injection of physical state vectors (Moment of Inertia Tensor: I_xx, I_yy, I_xy, Eigen_Ratio) into the embedding space.
    *   Ensures model predictions are consistent with fundamental physical constraints.

*   **💻 Simulation-Based Training**:
    *   Trained on high-fidelity Monte Carlo simulations (**MATESIM** based on Geant4).
    *   Addresses the scarcity of labeled experimental data in nuclear physics.

*   **🧠 Hybrid ResNet-ViT Architecture**:
    *   **ResNet-18 Backbone** (adapted for 80×48 input): Efficiently extracts local spatial features from MATE imagery.
    *   **Vision Transformer (ViT)**: Captures long-range global dependencies and trajectory patterns.
    *   **Physics Token Mechanism**: Dedicated token for physics-informed feature aggregation.

*   **🔍 Real Attention-Based XAI (Explainable AI)**:
    *   Provides transparent decision-making via **real attention weight extraction** (not dummy visualizations).
    *   Enables physicists to verify that the model focuses on relevant particle tracks rather than noise artifacts and learns features from real event topology.

## Architecture & Visuals

![Cross-Attention Workflow Schematic](assets/images/workflow_schematic.png)
*Physics-informed hybrid architecture combining ResNet-18 and Vision Transformer with cross-attention mechanism.*

## Performance & Reliability

The model is evaluated on synthetic test sets to validate the Physics-Informed hypothesis.

![Dynamic Comparison](assets/images/performance_comparison.png)
*Comparison of model accuracy with and without Physics-Informed constraints.*

![Attention Heatmap of Physics-Informed Model](assets/images/attention_visualization.png)
*XAI Visualization: The model correctly focuses on the particle track (high energy deposition area).*

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

## Quick Start

### 1. Prepare Data

Ensure your MATE simulation data (HDF5 format) is placed in `../dataset/HDF5_Form/`:
- `sim_inv_12C300MeV_4He_3He_25k.h5` (³He particles, label 0)
- `sim_inv_12C300MeV_4He_4He_25k.h5` (⁴He particles, label 1)

### 2. Train the Model

```bash
# Full training (uses all 25k samples per class)
python train.py

# Quick test with limited data
python train.py --max_samples_per_class 1000 --epochs 10

# Custom configuration
python train.py \
    --data_dir ../dataset/HDF5_Form \
    --epochs 80 \
    --batch_size 64 \
    --lr 0.001 \
    --save_dir checkpoints
```

### 3. Evaluate the Model

```bash
# Evaluate with attention visualization
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --visualize_attention \
    --num_vis_samples 5

# Evaluate without visualization (faster)
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 4. Results

After evaluation, check the `evaluation_results/` directory for:
- `confusion_matrix.png` - Classification performance
- `attention_sample_*.png` - XAI attention visualizations
- `evaluation_results.json` - Detailed metrics

## Roadmap & Future Work

We are actively working on bridging the gap between simulation and reality:

- [x] **Phase 1**: Physics-Informed Architecture Design & Simulation Training (✅ **Completed**).
- [ ] **Phase 2**: **Sim-to-Real Domain Adaptation** (🔄 **Planned**).
      - Implementing Unsupervised Domain Adaptation (UDA) techniques to align features between MATESIM data and real MATE experimental data.
      - Exploring methods: Domain-Adversarial Neural Networks (DANN), Maximum Mean Discrepancy (MMD).
- [ ] **Phase 3**: Realizing the reconstruction of Excitation Spectrum and Reaction Kinematics.

**Note**: The current codebase focuses on **simulation-based training**. Sim-to-Real adaptation is planned for future development and is not yet implemented.

## Project Structure

```text
├── models/             # Neural Network definitions (ResNet+ViT Hybrid)
│   ├── __init__.py
│   └── model.py        # Physics-Informed Hybrid Model
├── utils/              # XAI visualization and metric calculation tools
│   ├── __init__.py
│   └── utils.py        # Attention visualization, confusion matrix, metrics
├── data/               # Data loaders for MATESIM datasets
│   ├── __init__.py
│   └── dataloader.py   # HDF5 dataset loader with train/val/test split
├── configs/            # Hyperparameter configurations
│   └── config.yaml     # Model and training hyperparameters
├── train.py            # Main training entry point
├── evaluate.py         # Inference and reliability analysis
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

## Technical Details

### Model Architecture
- **Input**: 64×64×2 TPC images (Charge deposition + Drift time)
- **Physics Features**: 4D Moment of Inertia tensor (I_xx, I_yy, I_xy, Eigen_Ratio)
- **Backbone**: Modified ResNet-18 (adapted for small input size)
- **Transformer**: 4-layer encoder with 8 attention heads
- **Output**: Binary classification (³He vs ⁴He)

### Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss**: Cross-Entropy Loss
- **Data Split**: 80% train, 10% validation, 10% test

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mate_event_classifier_2025,
  title={MATE-Event-Classifier-DL: Physics-Informed Deep Learning for TPC Data Analysis},
  author={Zhiheng Hu},
  year={2025},
  url={https://github.com/Edfghdrtxxx/MATE-Event-Classifier-DL}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MATESIM Toolkit**: Monte Carlo simulation framework based on Geant4
- **MATE-TPC Collaboration**: For providing the experimental context and physics insights



