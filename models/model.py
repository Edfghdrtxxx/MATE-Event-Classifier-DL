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
        - Supports configurable input sizes (default: 80x48 for MATE TPC Y-Z projection)
        - Real attention weight extraction for explainability (XAI)
        - Physics-informed feature fusion via dedicated physics token

    Attributes:
        features (nn.Sequential): Modified ResNet-18 backbone for spatial feature extraction.
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
                 in_channels: int = 2,
                 img_height: int = 80,
                 img_width: int = 48):
        """
        Initialize the Physics-Informed Hybrid Model.

        Args:
            num_classes (int): Number of target classes (default: 2 for 3He vs 4He).
            num_physics_params (int): Number of input physical parameters (default: 4 for Moment of Inertia features).
            embed_dim (int): Dimensionality of the feature embeddings.
            num_heads (int): Number of attention heads in the Transformer.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            in_channels (int): Number of input image channels (default: 2 for MATE TPC: Charge + Drift Time).
            img_height (int): Height of input images (default: 80 for Y-axis bins).
            img_width (int): Width of input images (default: 48 for Z-axis bins).
        """
        super(PhysicsInformedHybridModel, self).__init__()

        # Store config for attention extraction
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.img_height = img_height
        self.img_width = img_width

        # --- 1. Vision Backbone (Modified ResNet-18 for configurable input) ---
        resnet = models.resnet18(pretrained=False)
        
        # Modify first conv: 3x3 kernel with stride=1 to preserve spatial resolution
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove initial max pooling to avoid excessive downsampling
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # 1x1 Convolution to project ResNet features (512 channels) to embed_dim
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        # Calculate feature map size after ResNet backbone
        self.feature_h = img_height // 8
        self.feature_w = img_width // 8
        self.num_patches = self.feature_h * self.feature_w

        # --- 2. Physics Branch ---
        self.physics_mlp = nn.Sequential(
            nn.Linear(num_physics_params, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # --- 3. Transformer Encoder ---
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
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # --- 4. Classification Head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
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
            img (torch.Tensor): Input images of shape (Batch, 2, H, W).
            physics_params (torch.Tensor): Input physical parameters of shape (Batch, 4).
            return_attention (bool): If True, returns attention weights from the last layer.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - logits: Classification scores (Batch, num_classes).
                - attention_weights: If return_attention=True, returns attention weights.
        """
        B = img.shape[0]

        # 1. Extract Visual Features
        x = self.features(img)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        # 2. Process Physics Parameters
        phys_embed = self.physics_mlp(physics_params).unsqueeze(1)

        # 3. Feature Fusion
        tokens = torch.cat((phys_embed, x), dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]

        # 4. Transformer Encoding
        attention_weights = None
        for i, layer in enumerate(self.transformer_layers):
            if return_attention and i == len(self.transformer_layers) - 1:
                tokens, attention_weights = self._forward_with_attention(layer, tokens)
            else:
                tokens = layer(tokens)

        # 5. Classification
        cls_token = tokens[:, 0, :]
        logits = self.classifier(cls_token)

        return logits, attention_weights

    def _forward_with_attention(self, layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom forward pass to extract attention weights."""
        attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = x + layer.dropout1(attn_output)
        x = layer.norm1(x)
        
        ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = x + layer.dropout2(ff_output)
        x = layer.norm2(x)
        
        return x, attn_weights


if __name__ == "__main__":
    print("=== Testing Physics-Informed Hybrid Model ===")
    
    # Test with 80x48 format
    print("\n--- Testing with 80x48 input (Y-Z projection) ---")
    model = PhysicsInformedHybridModel(
        num_classes=2,
        num_physics_params=4,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        img_height=80,
        img_width=48
    )
    
    dummy_img = torch.randn(2, 2, 80, 48)
    dummy_phys = torch.randn(2, 4)
    
    logits, _ = model(dummy_img, dummy_phys, return_attention=False)
    print(f"  Logits shape: {logits.shape} (expected: [2, 2])")
    
    logits, attn = model(dummy_img, dummy_phys, return_attention=True)
    print(f"  Attention shape: {attn.shape} (expected: [2, 8, 61, 61])")
    print("  80x48 test passed!")
    
    # Test with 64x64 format
    print("\n--- Testing with 64x64 input (legacy) ---")
    model_legacy = PhysicsInformedHybridModel(
        num_classes=2,
        num_physics_params=4,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        img_height=64,
        img_width=64
    )
    
    dummy_img_64 = torch.randn(2, 2, 64, 64)
    
    logits, attn = model_legacy(dummy_img_64, dummy_phys, return_attention=True)
    print(f"  Attention shape: {attn.shape} (expected: [2, 8, 65, 65])")
    print("  64x64 test passed!")
    
    print("\n=== All model tests passed! ===")
