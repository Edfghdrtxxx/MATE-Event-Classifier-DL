import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.model import PhysicsInformedHybridModel
from data.dataloader import get_dataloader
from utils.utils import compute_classification_metrics
import os
import json
from tqdm import tqdm

def train(args):
    print(f"=" * 60)
    print(f"MATE Event Classifier Training")
    print(f"=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Number of classes: {args.num_classes}")
    print(f"  Data directory: {args.data_dir}")
    print(f"=" * 60)
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # 2. Load Data
    print("\nLoading datasets...")
    train_loader = get_dataloader(
        batch_size=args.batch_size, 
        mode='train',
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples_per_class
    )
    
    val_loader = get_dataloader(
        batch_size=args.batch_size, 
        mode='val',
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples_per_class
    )
    
    # 3. Initialize Model
    print("\nInitializing Physics-Informed Hybrid Model...")
    model = PhysicsInformedHybridModel(
        num_classes=args.num_classes,
        num_physics_params=4,  # Moment of Inertia features (I_xx, I_yy, I_xy, Eigen_Ratio)
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        in_channels=2  # TPC data: Charge + Time
    ).to(device)
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 4. Optimizer & Loss & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 5. Training Loop
    print("\nStarting training...")
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")
        for batch_idx, (img, phys, target) in enumerate(pbar):
            img, phys, target = img.to(device), phys.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(img, phys, return_attention=False)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * train_correct / train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img, phys, target in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Val  "):
                img, phys, target = img.to(device), phys.to(device), target.to(device)
                
                logits, _ = model(img, phys, return_attention=False)
                loss = criterion(logits, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print("-" * 60)

    # 6. Save Training History
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to: {args.save_dir}/best_model.pth")
    print(f"  Training history saved to: {args.save_dir}/training_history.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    
    # Load configuration from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="MATE Event Classifier Training")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=config['data']['data_dir'],
                       help='Directory containing HDF5 data files')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Maximum samples per class (None = use all data)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'], 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'], 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config['training']['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=config['data']['num_workers'],
                       help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=config['model']['num_classes'], 
                       help='Number of classes (2 for 3He vs 4He)')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default=config['training']['save_dir'],
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    train(args)
