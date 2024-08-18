import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from basicsr.models.archs.kb_utils import KBAFunction
from basicsr.models.archs.kb_utils import LayerNorm2d, SimpleGate

        
class KBBlock_s(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False, GCE_Conv=2):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)
        self.GCE_Conv = GCE_Conv
        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )


        #self.conv1 = nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #self.conv21 = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True)
        self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.Conv2d(c, c, kernel_size=5, padding=2, stride=1, groups=c // 4, bias=True),
            )
        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, interc, kernel_size=3, padding=1, stride=1, groups=interc, bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        

        self.conv211 = nn.Conv2d(c, self.nset, kernel_size=1)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.depthwise_conv31 = nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=3, padding=2, stride=1, dilation=2, groups=c//2, bias=True)
        self.pointwise_conv31 = nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=1, bias=True)
        self.dropout1 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        sca = self.sca(x)
        lg = self.conv11(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.pointwise_conv31(self.depthwise_conv31(x1))
        x2 = self.pointwise_conv31(self.depthwise_conv31(x2))
        x3 = torch.cat((x1, x2), dim=1)
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        x = self.KBA(x3, att, self.k, self.g, self.b, self.w) * self.ga1 + x3
        x = x * lg *sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm1(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout1(x)
        return y + x * self.gamma

class KBNet_s(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1,1,1,28],
                 dec_blk_nums=[1, 1,1,1], basicblock='KBBlock_s',lightweight=False, ffn_scale=2):
        super().__init__()
        basicblock = eval(basicblock)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from arch_util import measure_inference_speed

    
    img_channel = 3

    width = 64
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    # width = 8
    # enc_blks = [1, 1, 1, 1]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = KBNet_s()

    net=KBNet_s()
    inp_shape = (3, 256, 256)
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    print(macs, params)
    
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    data = torch.randn((1, *inp_shape))
    print(device)
    measure_inference_speed(net.to(device), (data.to(device),), max_iter=500, log_interval=50)