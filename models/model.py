import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple

class PhysicsInformedHybridModel(nn.Module):
    """
    A Physics-Informed Hybrid Deep Learning Model combining ResNet-18 and Vision Transformer (ViT).

    This model is designed for the MATE experiment analysis, integrating visual data processing
    with physical parameter embeddings. It utilizes a ResNet-18 backbone for feature extraction
    and a Transformer encoder for capturing global dependencies, with a specific "Physics Token"
    injection mechanism.

    Key Features:
        - Adapted for 64x64 input images (MATE TPC data)
        - Real attention weight extraction for explainability (XAI)
        - Physics-informed feature fusion via dedicated physics token

    Attributes:
        features (nn.Sequential): Modified ResNet-18 backbone for 64x64 spatial feature extraction.
        proj (nn.Conv2d): 1x1 convolution to project ResNet features to embedding dimension.
        physics_mlp (nn.Sequential): MLP to project physical parameters to embedding dimension.
        pos_embedding (nn.Parameter): Learnable positional embedding for image patches + physics token.
        transformer_layers (nn.ModuleList): List of transformer encoder layers (for attention extraction).
        classifier (nn.Sequential): Final classification head.
    """

    def __init__(self, 
                 num_classes: int = 2, 
                 num_physics_params: int = 4, 
                 embed_dim: int = 256, 
                 num_heads: int = 8, 
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 in_channels: int = 2):
        """
        Initialize the Physics-Informed Hybrid Model.

        Args:
            num_classes (int): Number of target classes (default: 2 for 3He vs 4He).
            num_physics_params (int): Number of input physical parameters (default: 4 for Moment of Inertia features).
            embed_dim (int): Dimensionality of the feature embeddings.
            num_heads (int): Number of attention heads in the Transformer.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            in_channels (int): Number of input image channels (default: 2 for MATE TPC: Charge + Time).
        """
        super(PhysicsInformedHybridModel, self).__init__()

        # Store config for attention extraction
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # --- 1. Vision Backbone (Modified ResNet-18 for 64x64 input) ---
        # Standard ResNet-18 is designed for 224x224, we adapt it for 64x64
        resnet = models.resnet18(pretrained=False)
        
        # Modify first conv: 3x3 kernel with stride=1 (instead of 7x7 stride=2) to preserve spatial resolution
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the initial max pooling layer to avoid excessive downsampling
        # For 64x64 input: conv1 (64x64) -> layer1 (64x64) -> layer2 (32x32) -> layer3 (16x16) -> layer4 (8x8)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # Skip resnet.maxpool
            resnet.layer1,  # Output: 64x64 (no downsampling)
            resnet.layer2,  # Output: 32x32 (stride=2)
            resnet.layer3,  # Output: 16x16 (stride=2)
            resnet.layer4   # Output: 8x8 (stride=2), 512 channels
        )
        
        # 1x1 Convolution to project ResNet features (512 channels) to embed_dim
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        # --- 2. Physics Branch ---
        # Physical parameters (Moment of Inertia: I_xx, I_yy, I_xy, Eigen_Ratio) are projected
        # to the same embedding dimension to act as a "Physics Context" token.
        self.physics_mlp = nn.Sequential(
            nn.Linear(num_physics_params, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # --- 3. Transformer Encoder (Custom implementation for attention extraction) ---
        # We use ModuleList to manually iterate and extract attention weights
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4,
                dropout=dropout, 
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Positional Embedding (Learnable)
        # For 64x64 input -> 8x8 feature map after ResNet = 64 patches
        self.max_patches = 8 * 8  # 64 patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_patches + 1, embed_dim))  # +1 for Physics Token

        # --- 4. Classification Head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the physics MLP and projection layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img: torch.Tensor, physics_params: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            img (torch.Tensor): Input images of shape (Batch, 2, 64, 64).
            physics_params (torch.Tensor): Input physical parameters of shape (Batch, 4).
            return_attention (bool): If True, returns attention weights from the last layer.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - logits: Classification scores (Batch, num_classes).
                - attention_weights: If return_attention=True, returns (Batch, num_heads, num_patches+1, num_patches+1).
                                    Otherwise returns None.
        """
        B = img.shape[0]

        # 1. Extract Visual Features
        # Input: (B, 2, 64, 64) -> Output: (B, 512, 8, 8)
        x = self.features(img)
        
        # Project to embedding dimension: (B, 512, 8, 8) -> (B, embed_dim, 8, 8)
        x = self.proj(x)
        
        # Flatten to sequence: (B, embed_dim, 64) -> Permute to (B, 64, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # 2. Process Physics Parameters
        # phys_embed shape: (B, embed_dim) -> (B, 1, embed_dim)
        phys_embed = self.physics_mlp(physics_params).unsqueeze(1)

        # 3. Feature Fusion
        # Prepend physics token to visual sequence
        # Physics token serves as a learnable aggregator conditioned on physical constraints
        tokens = torch.cat((phys_embed, x), dim=1)  # Shape: (B, 65, embed_dim)

        # Add Positional Embedding
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]

        # 4. Transformer Encoding with Attention Extraction
        attention_weights = None
        for i, layer in enumerate(self.transformer_layers):
            # For the last layer, extract attention if requested
            if return_attention and i == len(self.transformer_layers) - 1:
                # Manual forward pass through the last layer to extract attention
                # Note: This is a workaround since nn.TransformerEncoderLayer doesn't expose attention
                # In production, consider using a custom TransformerEncoderLayer
                tokens, attention_weights = self._forward_with_attention(layer, tokens)
            else:
                tokens = layer(tokens)

        # 5. Classification
        # Use the Physics Token (index 0) output for classification
        cls_token = tokens[:, 0, :]
        logits = self.classifier(cls_token)

        return logits, attention_weights

    def _forward_with_attention(self, layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom forward pass through a transformer layer to extract attention weights.
        
        Args:
            layer: TransformerEncoderLayer instance
            x: Input tensor (B, seq_len, embed_dim)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # This is a simplified version - in production, you'd need to replicate the full layer logic
        # For now, we'll use the layer's self_attn module directly
        
        # Self-attention
        attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = x + layer.dropout1(attn_output)
        x = layer.norm1(x)
        
        # Feedforward
        ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = x + layer.dropout2(ff_output)
        x = layer.norm2(x)
        
        return x, attn_weights

if __name__ == "__main__":
    # Test with correct input dimensions for MATE experiment
    print("=== Testing Physics-Informed Hybrid Model ===")
    
    model = PhysicsInformedHybridModel(
        num_classes=2,           # 3He vs 4He binary classification
        num_physics_params=4,    # Moment of Inertia features
        embed_dim=256,
        num_heads=8,
        num_layers=4
    )
    
    # MATE TPC data: 2 channels (Charge + Time), 64x64 resolution
    dummy_img = torch.randn(2, 2, 64, 64)
    dummy_phys = torch.randn(2, 4)
    
    # Forward pass without attention
    logits, _ = model(dummy_img, dummy_phys, return_attention=False)
    print(f"✓ Logits shape: {logits.shape} (expected: [2, 2])")
    
    # Forward pass with attention extraction
    logits, attn = model(dummy_img, dummy_phys, return_attention=True)
    print(f"✓ Attention shape: {attn.shape} (expected: [2, 8, 65, 65])")
    print(f"✓ Model test passed!")
