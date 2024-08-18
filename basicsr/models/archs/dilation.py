import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

class CustomConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(CustomConvBlock, self).__init__()
        self.depthwise_conv31 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=4, stride=1, dilation=2, groups=in_channels, bias=True)
        self.pointwise_conv31 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, bias=True)

        self.depthwise_conv32 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=2, stride=1, dilation=2, groups=in_channels, bias=True)
        self.pointwise_conv32 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.pointwise_conv31(self.depthwise_conv31(x))
        x2 = self.pointwise_conv32(self.depthwise_conv32(x))
        x=torch.cat((x1, x2), dim=1)
        return x

# Example usage with input shape (3, 224, 224)
in_channels = 64
input_res = (in_channels, 256, 256)

model = CustomConvBlock(in_channels=in_channels)

macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                         print_per_layer_stat=True, verbose=True)

print(f'MACs: {macs}')
print(f'Params: {params}')
