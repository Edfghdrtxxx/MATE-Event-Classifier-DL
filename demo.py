"""
Quick Demo Script for MATE Event Classifier

This script demonstrates the basic functionality of the Physics-Informed Hybrid Model
without requiring the full dataset. It's useful for:
1. Verifying the installation
2. Testing the model architecture
3. Understanding the data flow

Usage:
    python demo.py
    python demo.py --img_size 64  # For legacy 64x64 format
"""

import torch
import numpy as np
import yaml
import os
import argparse
from models.model import PhysicsInformedHybridModel
from utils.utils import visualize_attention_map

def main(args):
    print("=" * 60)
    print("MATE Event Classifier - Quick Demo")
    print("=" * 60)
    print(f"\nImage format: {args.img_height}x{args.img_width} (Y-Z projection)")
    
    # 1. Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # 2. Initialize model
    print("\nInitializing Physics-Informed Hybrid Model...")
    model = PhysicsInformedHybridModel(
        num_classes=args.num_classes,
        num_physics_params=4,    # Moment of Inertia features
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=4,
        dropout=0.1,
        in_channels=2,
        img_height=args.img_height,
        img_width=args.img_width
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {num_params:,} parameters")
    print(f"  Input: ({args.img_height}, {args.img_width}, 2) images + 4 physics features")
    print(f"  Output: {args.num_classes} classes")
    
    # 3. Create dummy data (simulating TPC data)
    print("\nCreating dummy TPC data...")
    batch_size = 4
    
    # TPC images: (batch, 2, H, W)
    # Channel 0: Charge deposition, Channel 1: Drift time
    dummy_images = torch.randn(batch_size, 2, args.img_height, args.img_width).to(device)
    
    # Physics features: (batch, 4)
    # Moment of Inertia: I_xx, I_yy, I_xy, Eigen_Ratio
    dummy_physics = torch.randn(batch_size, 4).to(device)
    
    print(f"  Image shape: {dummy_images.shape}")
    print(f"  Physics features shape: {dummy_physics.shape}")
    
    # 4. Forward pass without attention
    print("\nRunning forward pass (without attention extraction)...")
    model.eval()
    with torch.no_grad():
        logits, _ = model(dummy_images, dummy_physics, return_attention=False)
    
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {preds.cpu().numpy()}")
    print(f"  Probabilities:")
    for i in range(batch_size):
        prob_str = ", ".join([f"{p:.3f}" for p in probs[i].cpu().numpy()])
        print(f"    Sample {i+1}: [{prob_str}]")
    
    # 5. Forward pass with attention extraction
    print("\nRunning forward pass (with attention extraction)...")
    with torch.no_grad():
        logits, attn_weights = model(dummy_images, dummy_physics, return_attention=True)
    
    # Calculate expected sequence length
    num_patches = (args.img_height // 8) * (args.img_width // 8)
    seq_len = num_patches + 1  # +1 for physics token
    
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Expected: (batch={batch_size}, num_heads={args.num_heads}, seq_len={seq_len}, seq_len={seq_len})")
    
    # 6. Visualize attention for first sample
    print("\nGenerating attention visualization...")
    os.makedirs("demo_outputs", exist_ok=True)
    
    visualize_attention_map(
        dummy_images[0].cpu(),
        attn_weights[0].cpu(),
        num_heads=args.num_heads,
        save_path="demo_outputs/demo_attention_map.png",
        show_individual_heads=False,
        img_height=args.img_height,
        img_width=args.img_width
    )
    
    print(f"  Saved to: demo_outputs/demo_attention_map.png")
    
    # 7. Test with different batch sizes
    print("\nTesting with different batch sizes...")
    for bs in [1, 8, 16]:
        test_img = torch.randn(bs, 2, args.img_height, args.img_width).to(device)
        test_phys = torch.randn(bs, 4).to(device)
        
        with torch.no_grad():
            test_logits, _ = model(test_img, test_phys, return_attention=False)
        
        expected_shape = (bs, args.num_classes)
        assert test_logits.shape == expected_shape, f"Expected shape {expected_shape}, got {test_logits.shape}"
        print(f"  Batch size {bs:2d}: Passed")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Model architecture is working correctly")
    print("  2. Attention extraction is functional")
    print("  3. Visualization pipeline is operational")
    print(f"\nData Format:")
    print(f"  - Image: {args.img_height}x{args.img_width} Y-Z projection")
    print(f"  - Channel 0: Charge deposition")
    print(f"  - Channel 1: Drift time (X-coordinate)")
    print("\nNext Steps:")
    print("  - Prepare your HDF5 data in dataset/HDF5_Form/")
    print("  - Run: python train.py --max_samples_per_class 1000 --epochs 10")
    print("  - Evaluate: python evaluate.py --checkpoint outputs/best_model.pth")
    print("=" * 60)

if __name__ == "__main__":
    # Load defaults from config
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        default_height = config['data']['img_height']
        default_width = config['data']['img_width']
        default_classes = config['model']['num_classes']
        default_embed_dim = config['model']['embed_dim']
        default_num_heads = config['model']['num_heads']
    else:
        default_height = 80
        default_width = 48
        default_classes = 2
        default_embed_dim = 256
        default_num_heads = 8
    
    parser = argparse.ArgumentParser(description="MATE Event Classifier Demo")
    parser.add_argument('--img_height', type=int, default=default_height,
                       help='Image height (default: 80 for Y-Z projection)')
    parser.add_argument('--img_width', type=int, default=default_width,
                       help='Image width (default: 48 for Y-Z projection)')
    parser.add_argument('--img_size', type=int, default=None,
                       help='Square image size (overrides img_height and img_width)')
    parser.add_argument('--num_classes', type=int, default=default_classes,
                       help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=default_embed_dim,
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=default_num_heads,
                       help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Handle --img_size for legacy 64x64 format
    if args.img_size is not None:
        args.img_height = args.img_size
        args.img_width = args.img_size
    
    main(args)
