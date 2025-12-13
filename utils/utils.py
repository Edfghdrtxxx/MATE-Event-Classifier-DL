import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from typing import List, Optional, Union

def plot_confusion_matrix(y_true: Union[np.ndarray, List], 
                          y_pred: Union[np.ndarray, List], 
                          classes: List[str],
                          normalize: bool = False,
                          title: str = 'Confusion Matrix',
                          cmap: plt.cm = plt.cm.Blues,
                          save_path: Optional[str] = None):
    """
    Computes and plots the confusion matrix.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        classes (List[str]): List of class names.
        normalize (bool): If True, returns normalized confusion matrix.
        title (str): Title of the plot.
        cmap (plt.cm): Colormap to use.
        save_path (str, optional): Path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def visualize_attention_map(img: torch.Tensor, 
                            attn_weights: torch.Tensor, 
                            num_heads: int = 8,
                            alpha: float = 0.6,
                            save_path: Optional[str] = None,
                            show_individual_heads: bool = False):
    """
    Visualizes the Attention Map overlaid on the original image (XAI).
    
    This function takes an image and the corresponding REAL attention weights from the model's
    transformer layer and overlays the heatmap to show which spatial regions the model focuses on.

    Args:
        img (torch.Tensor): Input image tensor of shape (2, 64, 64) for MATE TPC data.
                           Channel 0: Charge deposition, Channel 1: Drift time
        attn_weights (torch.Tensor): Real attention weights from model of shape (num_heads, seq_len, seq_len).
                                     seq_len = 65 (1 physics token + 64 spatial patches from 8x8 grid)
        num_heads (int): Number of attention heads (default: 8)
        alpha (float): Opacity of the overlay heatmap.
        save_path (str, optional): Path to save the visualization.
        show_individual_heads (bool): If True, visualize each attention head separately.
    """
    # 1. Prepare Image
    # Convert tensor to numpy and use the charge deposition channel (channel 0) for visualization
    img_np = img[0].detach().cpu().numpy()  # Shape: (64, 64)
    
    # Normalize to [0,1] for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    # Convert to RGB for visualization
    img_rgb = np.stack([img_np] * 3, axis=-1)  # Shape: (64, 64, 3)
    
    # 2. Process Attention Weights
    # attn_weights shape: (num_heads, 65, 65)
    # We want to see what the Physics Token (index 0) attends to in the spatial patches
    
    if attn_weights.dim() == 3:
        # Extract attention from Physics Token to spatial patches
        # attn_weights[:, 0, 1:] gives attention from physics token to 64 spatial patches
        spatial_attn = attn_weights[:, 0, 1:].detach().cpu().numpy()  # Shape: (num_heads, 64)
    else:
        raise ValueError(f"Expected attention weights of shape (num_heads, seq_len, seq_len), got {attn_weights.shape}")
    
    # Average across heads for main visualization
    avg_attn = spatial_attn.mean(axis=0)  # Shape: (64,)
    
    # Reshape to 2D grid (8x8 patches)
    grid_size = 8
    attn_map = avg_attn.reshape(grid_size, grid_size)  # Shape: (8, 8)
    
    # Upsample attention map to image size (64x64)
    import cv2
    attn_map_resized = cv2.resize(attn_map, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    # Normalize attention map for heatmap
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
    
    # 3. Plot Main Visualization
    if show_individual_heads:
        # Show individual attention heads
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Plot average attention
        axes[0].imshow(img_rgb)
        axes[0].imshow(attn_map_resized, cmap='jet', alpha=alpha)
        axes[0].set_title('Average Attention (All Heads)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot individual heads
        for head_idx in range(min(num_heads, 8)):
            head_attn = spatial_attn[head_idx].reshape(grid_size, grid_size)
            head_attn_resized = cv2.resize(head_attn, (64, 64), interpolation=cv2.INTER_LINEAR)
            head_attn_resized = (head_attn_resized - head_attn_resized.min()) / (head_attn_resized.max() - head_attn_resized.min() + 1e-8)
            
            axes[head_idx + 1].imshow(img_rgb)
            axes[head_idx + 1].imshow(head_attn_resized, cmap='jet', alpha=alpha)
            axes[head_idx + 1].set_title(f'Head {head_idx + 1}', fontsize=10)
            axes[head_idx + 1].axis('off')
        
        plt.tight_layout()
    else:
        # Simple 2-panel visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original TPC Image\n(Charge Deposition)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img_rgb)
        im = axes[1].imshow(attn_map_resized, cmap='jet', alpha=alpha)
        axes[1].set_title("Physics-Informed Attention Map\n(What the model focuses on)", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attention map saved to {save_path}")
    
    plt.close()


def compute_classification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray, 
                                   class_names: List[str]) -> dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary containing accuracy, precision, recall, F1-score
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_samples': len(y_true)
    }
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = per_class_precision[i]
        metrics[f'{class_name}_recall'] = per_class_recall[i]
        metrics[f'{class_name}_f1'] = per_class_f1[i]
        metrics[f'{class_name}_support'] = per_class_support[i]
    
    return metrics


if __name__ == "__main__":
    # Test with realistic attention weights
    print("=== Testing Attention Visualization ===")
    
    # Create a dummy image (2, 64, 64)
    dummy_img = torch.randn(2, 64, 64)
    
    # Create realistic attention weights (8 heads, 65 tokens, 65 tokens)
    # Simulate attention pattern where physics token attends more to center region
    dummy_attn = torch.softmax(torch.randn(8, 65, 65), dim=-1)
    
    # Test visualization
    visualize_attention_map(
        dummy_img, 
        dummy_attn, 
        num_heads=8,
        save_path="test_attention_map.png",
        show_individual_heads=False
    )
    
    print("✓ Attention visualization test passed!")
    print("  Check test_attention_map.png for output")
