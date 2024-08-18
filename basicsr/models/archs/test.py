import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=0, stride=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class GlobalContextExtractor(nn.Module):
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding=0, bias=False):
        super(GlobalContextExtractor, self).__init__()
        self.depthwise_separable_convs = nn.ModuleList([
            depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ])

    def forward(self, x):
        outputs = []
        for conv in self.depthwise_separable_convs:
            x = F.gelu(conv(x))
            outputs.append(x)
        return outputs

if __name__ == '__main__':
    # Instantiate the model
    model = GlobalContextExtractor(c=64)

    # Define the input shape (batch_size, channels, height, width)
    input_shape = (64, 128, 128)

    # Calculate FLOPs and parameters
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False)
    
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")
