"""
Quick Verification Script - Tests core components without full demo
Validates the Physics-Informed Hybrid Model for MATE TPC data analysis
"""

import sys
import traceback

def test_imports():
    """Test 1: Check if all modules can be imported"""
    print("=" * 60)
    print("Test 1: Checking imports...")
    print("=" * 60)
    
    try:
        import torch
        print("  PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
    except ImportError as e:
        print(f"  Failed to import PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print("  NumPy imported successfully")
    except ImportError as e:
        print(f"  Failed to import NumPy: {e}")
        return False
    
    try:
        from models.model import PhysicsInformedHybridModel
        print("  PhysicsInformedHybridModel imported successfully")
    except ImportError as e:
        print(f"  Failed to import model: {e}")
        print("  Make sure you're running from the mate_experiment_pdl directory")
        return False
    
    try:
        from utils.utils import visualize_attention_map
        print("  Utils imported successfully")
    except ImportError as e:
        print(f"  Failed to import utils: {e}")
        return False
    
    return True

def test_model_creation():
    """Test 2: Check if model can be created"""
    print("\n" + "=" * 60)
    print("Test 2: Creating model...")
    print("=" * 60)
    
    try:
        import torch
        from models.model import PhysicsInformedHybridModel
        
        # Test with new 80x48 format (Y-Z projection)
        model = PhysicsInformedHybridModel(
            num_classes=2,
            num_physics_params=4,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            in_channels=2,
            img_height=80,
            img_width=48
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model created successfully (80x48 Y-Z projection)")
        print(f"  Total parameters: {num_params:,}")
        
        return True
    except Exception as e:
        print(f"  Failed to create model: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test 3: Check if forward pass works"""
    print("\n" + "=" * 60)
    print("Test 3: Testing forward pass...")
    print("=" * 60)
    
    try:
        import torch
        from models.model import PhysicsInformedHybridModel
        
        # Test with 80x48 format
        model = PhysicsInformedHybridModel(
            num_classes=2,
            num_physics_params=4,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            img_height=80,
            img_width=48
        )
        model.eval()
        
        # Create dummy data (80x48 Y-Z projection)
        dummy_img = torch.randn(2, 2, 80, 48)
        dummy_phys = torch.randn(2, 4)
        
        # Forward pass without attention
        with torch.no_grad():
            logits, _ = model(dummy_img, dummy_phys, return_attention=False)
        
        print(f"  Forward pass successful")
        print(f"  Input image shape: {dummy_img.shape}")
        print(f"  Input physics shape: {dummy_phys.shape}")
        print(f"  Output logits shape: {logits.shape}")
        
        if logits.shape != (2, 2):
            print(f"  Unexpected output shape: {logits.shape}, expected (2, 2)")
            return False
        
        return True
    except Exception as e:
        print(f"  Forward pass failed: {e}")
        traceback.print_exc()
        return False

def test_attention_extraction():
    """Test 4: Check if attention extraction works"""
    print("\n" + "=" * 60)
    print("Test 4: Testing attention extraction...")
    print("=" * 60)
    
    try:
        import torch
        from models.model import PhysicsInformedHybridModel
        
        model = PhysicsInformedHybridModel(
            num_classes=2,
            num_physics_params=4,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            img_height=80,
            img_width=48
        )
        model.eval()
        
        # Create dummy data
        dummy_img = torch.randn(2, 2, 80, 48)
        dummy_phys = torch.randn(2, 4)
        
        # Forward pass with attention
        with torch.no_grad():
            logits, attn_weights = model(dummy_img, dummy_phys, return_attention=True)
        
        # For 80x48: feature map = 10x6 = 60 patches + 1 physics token = 61
        expected_seq_len = 61
        expected_shape = (2, 8, expected_seq_len, expected_seq_len)
        
        print(f"  Attention extraction successful")
        print(f"  Attention shape: {attn_weights.shape}")
        print(f"  Expected: {expected_shape}")
        
        if attn_weights.shape != expected_shape:
            print(f"  Unexpected attention shape: {attn_weights.shape}")
            return False
        
        return True
    except Exception as e:
        print(f"  Attention extraction failed: {e}")
        traceback.print_exc()
        return False

def test_legacy_format():
    """Test 5: Check backward compatibility with 64x64 format"""
    print("\n" + "=" * 60)
    print("Test 5: Testing legacy 64x64 format compatibility...")
    print("=" * 60)
    
    try:
        import torch
        from models.model import PhysicsInformedHybridModel
        
        model = PhysicsInformedHybridModel(
            num_classes=2,
            num_physics_params=4,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            img_height=64,
            img_width=64
        )
        model.eval()
        
        # Create dummy data (legacy 64x64 format)
        dummy_img = torch.randn(2, 2, 64, 64)
        dummy_phys = torch.randn(2, 4)
        
        # Forward pass with attention
        with torch.no_grad():
            logits, attn_weights = model(dummy_img, dummy_phys, return_attention=True)
        
        # For 64x64: feature map = 8x8 = 64 patches + 1 physics token = 65
        expected_seq_len = 65
        expected_shape = (2, 8, expected_seq_len, expected_seq_len)
        
        print(f"  Legacy format test successful")
        print(f"  Input: 64x64 images")
        print(f"  Output logits: {logits.shape}")
        print(f"  Attention: {attn_weights.shape} (expected: {expected_shape})")
        
        if attn_weights.shape != expected_shape:
            print(f"  Unexpected attention shape: {attn_weights.shape}")
            return False
        
        return True
    except Exception as e:
        print(f"  Legacy format test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("MATE Event Classifier - Quick Verification")
    print("=" * 60)
    print("\nThis script tests core components without running the full demo.\n")
    print("Data Format: Y-Z projection (80x48), Channel 0: Charge, Channel 1: Drift Time\n")
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports succeeded
        results.append(("Model Creation", test_model_creation()))
        results.append(("Forward Pass", test_forward_pass()))
        results.append(("Attention Extraction", test_attention_extraction()))
        results.append(("Legacy 64x64 Compatibility", test_legacy_format()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Your setup is working correctly.")
        print("\nYou can now run the full demo:")
        print("  python demo.py")
    else:
        print("Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Make sure you're in the mate_experiment_pdl directory")
        print("  3. Check that all files are present (models/, utils/, etc.)")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
