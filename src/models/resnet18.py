import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaselineResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(BaselineResNet, self).__init__()
        
        # Load trọng số pretrained giúp mô hình trích xuất đặc trưng cạnh/góc tốt hơn
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        
        # 1. Sửa lớp Conv1: Từ 3 channels (RGB) xuống 1 channel (Spectrogram)
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy trung bình trọng số từ 3 kênh cũ sang kênh mới (giúp hội tụ nhanh hơn khởi tạo ngẫu nhiên)
        if pretrained:
            with torch.no_grad():
                self.model.conv1.weight[:] = original_conv1.weight.mean(dim=1, keepdim=True)
                
        # 2. Sửa lớp Fully Connected cuối cùng: Trả về 1 giá trị (raw logit)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        # x: (Batch, Channel=1, Mels, TimeFrames)
        return self.model(x)

if __name__ == "__main__":
    model = BaselineResNet()
    dummy_input = torch.randn(2, 1, 80, 301) # Shape khớp với file logmel.py
    output = model(dummy_input)
    print(f"[INFO] Kiến trúc ResNet18 (1-channel) khởi tạo thành công!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} -> (Batch, 1 logit)")