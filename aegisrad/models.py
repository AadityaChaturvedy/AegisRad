import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    """ResNet-50 based vision encoder with custom naming"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)

        # Map to match checkpoint naming: stem, layer1-4
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Free the temporary full ResNet-50 immediately (~90 MB)
        del resnet
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)  # [B, 2048, 7, 7]
        
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)  # [B, 2048]
        
        return features, pooled

class QFormer(nn.Module):
    """Q-Former with 32 query tokens"""
    def __init__(self):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, 32, 2048))
        self.visual_proj = nn.Linear(2048, 2048)
        
    def forward(self, visual_features):
        B = visual_features.size(0)
        queries = self.query_tokens.expand(B, -1, -1)        # [B, 32, 2048]

        # Project visual features into query space
        vis_flat = visual_features.view(B, 2048, -1).transpose(1, 2)  # [B, 49, 2048]
        vis_proj = self.visual_proj(vis_flat)                          # [B, 49, 2048]

        # Scaled dot-product cross-attention: queries attend to visual patches
        scale   = queries.size(-1) ** 0.5
        attn    = torch.matmul(queries, vis_proj.transpose(1, 2)) / scale  # [B, 32, 49]
        weights = torch.softmax(attn, dim=-1)
        attended = torch.matmul(weights, vis_proj)  # [B, 32, 2048]

        # Residual: learned queries + image-conditioned context
        return queries + attended  # [B, 32, 2048]


class ReflexiveProjector(nn.Module):
    def __init__(self, hidden_dim=2048, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256 * num_queries)  # Outputs 256 * 32 = 8192
        )
        self.norm = nn.LayerNorm(256)  # Normalizes the 256-dim space
        self.out_proj = nn.Linear(256, hidden_dim)  # Projects back to 2048

    def forward(self, text_repr):
        B = text_repr.size(0)
        projected = self.projector(text_repr)
        recon_queries = projected.view(B, self.num_queries, 256)
        return self.out_proj(self.norm(recon_queries))

class AttentionPooling(nn.Module):
    """Learns to weigh the 32 query tokens based on their diagnostic importance."""
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: [B, 32, 2048]
        weights = self.attn(x)  # [B, 32, 1]
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # [B, 2048]
        return pooled


class ClinicalHead(nn.Module):
    """Multi-label classification head with integrated Attention Pooling"""
    def __init__(self, num_classes=14, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.pooler = AttentionPooling()
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # If input is [B, 32, 2048], pool it first
        if self.use_attention and len(x.shape) == 3:
            x = self.pooler(x)
        elif len(x.shape) == 3:
            x = x.mean(dim=1)
            
        return self.classifier(x)
