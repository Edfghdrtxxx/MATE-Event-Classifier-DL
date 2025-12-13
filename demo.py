"""
Quick Demo Script for MATE Event Classifier

This script demonstrates the basic functionality of the Physics-Informed Hybrid Model
without requiring the full dataset. It's useful for:
1. Verifying the installation
2. Testing the model architecture
3. Understanding the data flow

Usage:
    python demo.py
"""

import torch
import numpy as np
from models.model import PhysicsInformedHybridModel
from utils.utils import visualize_attention_map
import os

def main():
    print("=" * 60)
    print("MATE Event Classifier - Quick Demo")
    print("=" * 60)
    
    # 1. Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # 2. Initialize model
    print("\n✓ Initializing Physics-Informed Hybrid Model...")
    model = PhysicsInformedHybridModel(
        num_classes=2,           # 3He vs 4He
        num_physics_params=4,    # Moment of Inertia features
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        in_channels=2
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {num_params:,} parameters")
    
    # 3. Create dummy data (simulating TPC data)
    print("\n✓ Creating dummy TPC data...")
    batch_size = 4
    
    # TPC images: (batch, 2, 64, 64)
    # Channel 0: Charge deposition, Channel 1: Drift time
    dummy_images = torch.randn(batch_size, 2, 64, 64).to(device)
    
    # Physics features: (batch, 4)
    # Moment of Inertia: I_xx, I_yy, I_xy, Eigen_Ratio
    dummy_physics = torch.randn(batch_size, 4).to(device)
    
    print(f"  Image shape: {dummy_images.shape}")
    print(f"  Physics features shape: {dummy_physics.shape}")
    
    # 4. Forward pass without attention
    print("\n✓ Running forward pass (without attention extraction)...")
    model.eval()
    with torch.no_grad():
        logits, _ = model(dummy_images, dummy_physics, return_attention=False)
    
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {preds.cpu().numpy()}")
    print(f"  Probabilities (3He, 4He):")
    for i in range(batch_size):
        print(f"    Sample {i+1}: [{probs[i,0]:.3f}, {probs[i,1]:.3f}]")
    
    # 5. Forward pass with attention extraction
    print("\n✓ Running forward pass (with attention extraction)...")
    with torch.no_grad():
        logits, attn_weights = model(dummy_images, dummy_physics, return_attention=True)
    
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Expected: (batch={batch_size}, num_heads=8, seq_len=65, seq_len=65)")
    
    # 6. Visualize attention for first sample
    print("\n✓ Generating attention visualization...")
    os.makedirs("demo_outputs", exist_ok=True)
    
    visualize_attention_map(
        dummy_images[0].cpu(),
        attn_weights[0].cpu(),
        num_heads=8,
        save_path="demo_outputs/demo_attention_map.png",
        show_individual_heads=False
    )
    
    print(f"  Saved to: demo_outputs/demo_attention_map.png")
    
    # 7. Test with different batch sizes
    print("\n✓ Testing with different batch sizes...")
    for bs in [1, 8, 16]:
        test_img = torch.randn(bs, 2, 64, 64).to(device)
        test_phys = torch.randn(bs, 4).to(device)
        
        with torch.no_grad():
            test_logits, _ = model(test_img, test_phys, return_attention=False)
        
        assert test_logits.shape == (bs, 2), f"Expected shape ({bs}, 2), got {test_logits.shape}"
        print(f"  Batch size {bs:2d}: ✓ Passed")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Model architecture is working correctly")
    print("  2. Attention extraction is functional")
    print("  3. Visualization pipeline is operational")
    print("\nNext Steps:")
    print("  - Prepare your HDF5 data in ../dataset/HDF5_Form/")
    print("  - Run: python train.py --max_samples_per_class 1000 --epochs 10")
    print("  - Evaluate: python evaluate.py --checkpoint checkpoints/best_model.pth")
    print("=" * 60)

if __name__ == "__main__":
    main()
