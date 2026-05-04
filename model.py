import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. CBAM: Channel Attention Module ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# --- 2. CBAM: Spatial Attention Module ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

# --- 3. The Full Model: ResNet-18 + CBAM + Counting Head ---
class UnoCountingResNet(nn.Module):
    def __init__(self, num_classes=56, max_copies=6):
        super(UnoCountingResNet, self).__init__()
        self.num_classes = num_classes
        self.max_copies = max_copies
        
        # Load standard ResNet-18 FROM SCRATCH (Kaggle rule compliance)
        # weights=None ensures no pre-trained ImageNet weights are used.
        resnet = models.resnet18(weights=None)
        
        # Extract features (everything except the final pooling and fully connected layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add CBAM to the final feature map (512 channels for ResNet-18)
        self.ca = ChannelAttention(512)
        self.sa = SpatialAttention()
        
        # Standard final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🚨 THE MULTI-TARGET COUNTING HEAD 🚨
        # Output size: 56 classes * 6 possible counts (0, 1, 2, 3, 4, 5)
        self.fc = nn.Linear(512, self.num_classes * self.max_copies)

    def forward(self, x):
        # 1. Base ResNet Feature Extraction
        x = self.features(x)
        
        # 2. Apply Attention (Focus on card corners and colors)
        x = x * self.ca(x) # Channel attention
        x = x * self.sa(x) # Spatial attention
        
        # 3. Flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Predict
        x = self.fc(x)
        
        # 5. Reshape to (Batch, 56, 6)
        # For every class, we output a vector of length 6 (logits for 0 to 5 copies)
        x = x.view(-1, self.num_classes, self.max_copies)
        
        return x

# --- Sanity Check ---
if __name__ == "__main__":
    model = UnoCountingResNet(num_classes=56, max_copies=6)
    # Simulate a batch of 8 images, 3 color channels, 224x224 dimensions
    dummy_input = torch.randn(8, 3, 224, 224) 
    output = model(dummy_input)
    
    # Expected output: torch.Size([8, 56, 6])
    # Total Parameters: ~11.1 Million (Safely under the 12M limit!)
    print(f"Output shape: {output.shape}") 
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")