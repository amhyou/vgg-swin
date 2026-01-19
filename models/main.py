import torch
import torch.nn as nn
import torchvision.models as models
import timm


class VGGSwinHybridNet(nn.Module):
    def __init__(self, num_classes=5):
        super(VGGSwinHybridNet, self).__init__()

        # 1. Truncated VGG16 Backbone
        # We only take the first 17 layers (up to the 3rd MaxPool)
        # This gives us a 28x28 spatial resolution
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(vgg.features.children())[:16])

        # 2. Swin Transformer Head
        # Using Swin-Tiny. It normally expects 56x56, but we will adapt 28x28.
        swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.embed_dim = swin.num_features  # 768

        # 3. The Bridge: Match VGG Channels (256) to Swin Stage 1 (96)
        # alignment with Swin-Tiny's pre-trained positional embeddings.
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 96, kernel_size=1),
        )

        # 4. Swin hierarchical stages
        self.swin_blocks = swin.layers
        self.norm = swin.norm
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 5. Classifier
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # Local Features at 28x28 resolution: [Batch, 256, 28, 28]
        x = self.backbone(x)

        # Map to Swin-friendly tokens: [Batch, 96, 56, 56]
        x = self.bridge(x)

        # Prepare for Swin window partitioning: [Batch, 56, 56, 96]
        x = x.permute(0, 2, 3, 1)

        # Global Context modeling
        for layer in self.swin_blocks:
            x = layer(x)

        # Final prediction
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)