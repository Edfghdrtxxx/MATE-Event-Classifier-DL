import yaml
import argparse
import torch
import numpy as np
from models.model import PhysicsInformedHybridModel
from data.dataloader import get_dataloader
from utils.utils import plot_confusion_matrix, visualize_attention_map, compute_classification_metrics
import os
from tqdm import tqdm
import json

# Class name mappings for different classification tasks
CLASS_NAMES = {
    2: ['3He', '4He'],
    3: ['Proton', 'Deuteron', 'Triton'],
    5: ['Proton', 'Deuteron', 'Triton', '3He', 'Alpha']
}

def evaluate(args, config):
    print(f"=" * 60)
    print(f"MATE Event Classifier Evaluation")
    print(f"=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print("\nInitializing model...")
    
    # Try to load config from checkpoint first
    model_config = {
        'num_classes': args.num_classes,
        'img_height': args.img_height,
        'img_width': args.img_width,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers
    }
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Load config from checkpoint if available
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            model_config.update(saved_config)
            print(f"  Using config from checkpoint:")
            for k, v in saved_config.items():
                print(f"    {k}: {v}")
    
    model = PhysicsInformedHybridModel(
        num_classes=model_config['num_classes'],
        num_physics_params=config['model']['num_physics_params'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        dropout=config['model']['dropout'],
        in_channels=config['model']['in_channels'],
        img_height=model_config['img_height'],
        img_width=model_config['img_width']
    ).to(device)
    
    # Load checkpoint weights
    if os.path.exists(args.checkpoint):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"    Epoch: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                print(f"    Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print(f"  Using randomly initialized weights for demonstration.")

    model.eval()
    
    # 2. Load Test Data
    print("\nLoading test dataset...")
    test_loader = get_dataloader(
        batch_size=args.batch_size, 
        mode='test',
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples_per_class,
        img_height=model_config['img_height'],
        img_width=model_config['img_width']
    )

    # 3. Inference Loop
    print("\nRunning inference...")
    all_preds = []
    all_targets = []
    all_probs = []
    
    # For attention visualization
    sample_images = []
    sample_attentions = []
    sample_labels = []
    sample_preds = []
    
    with torch.no_grad():
        for i, (img, phys, target) in enumerate(tqdm(test_loader, desc="Evaluating")):
            img, phys = img.to(device), phys.to(device)
            
            # Get predictions with attention for first few samples
            extract_attention = (args.visualize_attention and i < args.num_vis_samples)
            logits, attn_weights = model(img, phys, return_attention=extract_attention)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Store samples for visualization
            if extract_attention:
                for j in range(min(args.num_vis_samples - len(sample_images), img.size(0))):
                    sample_images.append(img[j].cpu())
                    sample_attentions.append(attn_weights[j].cpu())
                    sample_labels.append(target[j].item())
                    sample_preds.append(preds[j].item())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # 4. Compute Metrics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # Get class names based on number of classes
    num_classes = model_config['num_classes']
    class_names = CLASS_NAMES.get(num_classes, [f'Class_{i}' for i in range(num_classes)])
    
    metrics = compute_classification_metrics(all_targets, all_preds, class_names)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"  Total Samples: {metrics['num_samples']}")
    
    print(f"\nPer-Class Performance:")
    for class_name in class_names:
        print(f"  {class_name}:")
        print(f"    Precision: {metrics[f'{class_name}_precision']*100:.2f}%")
        print(f"    Recall:    {metrics[f'{class_name}_recall']*100:.2f}%")
        print(f"    F1-Score:  {metrics[f'{class_name}_f1']*100:.2f}%")
        print(f"    Support:   {int(metrics[f'{class_name}_support'])}")
    
    # 5. Generate Visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Confusion Matrix
    print(f"\nGenerating confusion matrix...")
    plot_confusion_matrix(
        all_targets, 
        all_preds, 
        classes=class_names,
        normalize=True,
        title='Confusion Matrix (Normalized)',
        save_path=os.path.join(args.output_dir, "confusion_matrix.png")
    )
    
    # Attention Maps
    if args.visualize_attention and len(sample_images) > 0:
        print(f"\nGenerating attention visualizations...")
        for idx in range(min(args.num_vis_samples, len(sample_images))):
            true_label = class_names[sample_labels[idx]]
            pred_label = class_names[sample_preds[idx]]
            correct = "correct" if sample_labels[idx] == sample_preds[idx] else "wrong"
            
            save_path = os.path.join(
                args.output_dir, 
                f"attention_sample_{idx+1}_{true_label}_pred_{pred_label}_{correct}.png"
            )
            
            visualize_attention_map(
                sample_images[idx],
                sample_attentions[idx],
                num_heads=model_config['num_heads'],
                save_path=save_path,
                show_individual_heads=args.show_individual_heads,
                img_height=model_config['img_height'],
                img_width=model_config['img_width']
            )
            
            status = "Correct" if correct == "correct" else "Wrong"
            print(f"  Sample {idx+1}: True={true_label}, Pred={pred_label} ({status})")
    
    # 6. Save Results
    results = {
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                   for k, v in metrics.items()},
        'confusion_matrix': {
            'true_labels': all_targets.tolist(),
            'predictions': all_preds.tolist()
        },
        'class_names': class_names,
        'model_config': model_config
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"  Results saved to: {args.output_dir}/")
    print(f"  - evaluation_results.json")
    print(f"  - confusion_matrix.png")
    if args.visualize_attention:
        print(f"  - attention_sample_*.png ({len(sample_images)} samples)")
    print(f"{'='*60}")

if __name__ == "__main__":
    
    # Load configuration from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="MATE Event Classifier Evaluation")
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=config['model']['num_classes'],
                       help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=config['model']['embed_dim'],
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=config['model']['num_heads'],
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--img_height', type=int, default=config['data']['img_height'],
                       help='Image height')
    parser.add_argument('--img_width', type=int, default=config['data']['img_width'],
                       help='Image width')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=config['data']['data_dir'],
                       help='Directory containing HDF5 data files')
    parser.add_argument('--batch_size', type=int, default=config['evaluation']['batch_size'],
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=config['data']['num_workers'],
                       help='Number of data loading workers')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Maximum samples per class (None = use all)')
    
    # Visualization parameters
    parser.add_argument('--visualize_attention', action='store_true', default=True,
                       help='Generate attention map visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=config['evaluation']['num_vis_samples'],
                       help='Number of samples to visualize')
    parser.add_argument('--show_individual_heads', action='store_true',
                       help='Show individual attention heads')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=config['evaluation']['output_dir'],
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    evaluate(args, config)
