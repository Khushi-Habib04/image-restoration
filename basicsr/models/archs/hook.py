import torch
import torch.nn as nn
from kb import KBBlock_s #  # Make sure the KBNet_s class is correctly imported
from torch import Tensor


class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(self.hook)

    def hook(self, layer, _, output):
        if isinstance(output, list):
            for i, out in enumerate(output):
                print(f"{layer.__name__}[{i}]: {out.shape}")
        else:
            print(f"{layer.__name__}: {output.shape}")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


img_channel = 3
width = 64
enc_blks = [2, 2, 4, 6]
middle_blk_num = 10
dec_blks = [2, 2, 2, 2]
GCE_CONVS_nums = [3, 3, 2, 2]

# Instantiate the model
verbose_nafnet = VerboseExecution(KBBlock_s(64))

# Create a dummy input with 3 channels
dummy_input = torch.ones(1, 64, 224, 224)

# Forward pass to print the shapes
_ = verbose_nafnet(dummy_input)
