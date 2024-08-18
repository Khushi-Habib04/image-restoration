# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init

# from basicsr.models.archs.kb_utils import KBAFunction
# from basicsr.models.archs.kb_utils import LayerNorm2d, SimpleGate
# class depthwise_separable_conv(nn.Module):
#     def __init__(self, nin, nout, kernel_size = 3, padding = 0, stide = 1, bias=False):
#         super(depthwise_separable_conv, self).__init__()
#         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
#         self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stide, padding=padding, groups=nin, bias=bias)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# class UpsampleWithFlops(nn.Upsample):
#     def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
#         super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
#         self.__flops__ = 0

#     def forward(self, input):
#         self.__flops__ += input.numel()
#         return super(UpsampleWithFlops, self).forward(input)

# class GlobalContextExtractor(nn.Module):
#     def __init__(self, c, kernel_sizes=[3,3,5], strides=[3,3,5], padding=0, bias=False):
#         super(GlobalContextExtractor, self).__init__()

#         self.depthwise_separable_convs = nn.ModuleList([
#             depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
#             for kernel_size, stride in zip(kernel_sizes, strides)
#         ])

#     def forward(self, x):
#         outputs = []
#         for conv in self.depthwise_separable_convs:
#             x = F.gelu(conv(x))
#             outputs.append(x)
#         return outputs
        
# class KBBlock_s(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False, GCE_Conv=2):
#         super(KBBlock_s, self).__init__()
#         self.k, self.c = k, c
#         self.nset = nset
#         dw_ch = int(c * DW_Expand)
#         ffn_ch = int(FFN_Expand * c)
#         self.GCE_Conv = GCE_Conv
#         self.g = c // gc
#         self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
#         self.b = nn.Parameter(torch.zeros(1, nset, c))
#         self.init_p(self.w, self.b)

#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)

#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
#         )

#         if not lightweight:
#             self.conv11 = nn.Sequential(
#                 nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
#                 nn.Conv2d(c, c, kernel_size=5, padding=2, stride=1, groups=c // 4, bias=True),
#             )
#         else:
#             self.conv11 = nn.Sequential(
#                 nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
#                 nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
#             )

#         self.conv1 = nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv21 = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True)

#         interc = min(c, 32)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, interc, kernel_size=3, padding=1, stride=1, groups=interc, bias=True),
#             SimpleGate(),
#             nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
#         )
        
#         self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[1, 3], strides=[1,3])
#         self.project_out = nn.Conv2d(dw_ch * 2, c, kernel_size=1)
#         self.project_out2 = nn.Conv2d(c, self.nset, kernel_size=1,padding=0, stride=1)
#         self.sca2 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(in_channels=int(dw_ch*1.5), out_channels=dw_ch*2, kernel_size=1, padding=0, stride=1,
#                         groups=1, bias=True))

#         self.conv211 = nn.Conv2d(c, self.nset, kernel_size=1)
#         self.conv3 = nn.Conv2d(dw_ch // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv01 = nn.Conv2d(in_channels=c, out_channels=dw_ch, kernel_size=1,
#                                 padding=0, stride=1, groups=1, bias=True)
#         self.conv02 = nn.Conv2d(in_channels=dw_ch, out_channels=dw_ch,
#                                 kernel_size=3, padding=1, stride=1, groups=dw_ch,
#                                bias=True)

#         self.dropout1 = nn.Identity()
#         self.dropout2 = nn.Identity()

#         self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
#         self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
#         self.sg = SimpleGate()

#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

#     def init_p(self, weight, bias=None):
#         init.kaiming_uniform_(weight, a=math.sqrt(5))
#         if bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(bias, -bound, bound)

#     def KBA(self, x, att, selfk, selfg, selfb, selfw):
#         return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

#     def forward(self, inp):
#         x = inp
#         b,c,h,w = x.shape
#         self.upsample = UpsampleWithFlops(size=(h,w), mode='nearest')
#         x = self.norm1(x)
#         sca = self.sca(x)
#         x1=self.conv11(x)
#          # Global Context Extractor + Range fusion
#         a=self.conv01(x)
#         a=self.conv02(a)
#         a=F.gelu(a)
#         x_1, x_2 = a.chunk(2, dim=1)
#         x1, x2 = self.GCE(x_1 + x_2)
#         a = torch.cat([x, self.upsample(x1), self.upsample(x2)], dim = 1)
#         a = self.sca2(a)
#         a = self.project_out(a)
        
#         #x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
#         x = x * a * x1 * sca

#         x = self.conv3(x)
#         x = self.dropout1(x)
#         y = inp + x * self.beta

#         # FFN
#         x = self.norm2(y)
#         x = self.conv4(x)
#         x = self.sg(x)
#         x = self.conv5(x)

#         x = self.dropout2(x)
#         return y + x * self.gamma

# class KBBlock_ss(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,lightweight=False):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
#                                bias=True)
#         self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )

#         # SimpleGate
#         self.sg = SimpleGate()

#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)

#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

#     def forward(self, inp):
#         x = inp

#         x = self.norm1(x)

#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.sg(x)
#         x = x * self.sca(x)
#         x = self.conv3(x)

#         x = self.dropout1(x)

#         y = inp + x * self.beta

#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)

#         x = self.dropout2(x)

#         return y + x * self.gamma

# class KBBlock_sss(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
#         super(KBBlock_sss, self).__init__()
#         self.k, self.c = k, c
#         self.nset = nset
#         dw_ch = int(c * DW_Expand)
#         ffn_ch = int(FFN_Expand * c)

#         self.g = c // gc
#         self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
#         self.b = nn.Parameter(torch.zeros(1, nset, c))
#         self.init_p(self.w, self.b)

#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)

#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )

#         if not lightweight:
#             self.conv11 = nn.Sequential(
#                 nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
#                           bias=True),
#                 nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
#                           bias=True),
#             )
#         else:
#             self.conv11 = nn.Sequential(
#                 nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
#                           bias=True),
#                 nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
#                           bias=True),
#             )

#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=True)
#         self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
#                                 bias=True)

#         interc = min(c, 32)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
#                       bias=True),
#             SimpleGate(),
#             nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
#         )

#         self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

#         self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=True)

#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=True)
#         self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=True)

#         self.dropout1 = nn.Identity()
#         self.dropout2 = nn.Identity()

#         self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
#         self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
#         self.sg = SimpleGate()

#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

#     def init_p(self, weight, bias=None):
#         init.kaiming_uniform_(weight, a=math.sqrt(5))
#         if bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(bias, -bound, bound)

#     def KBA(self, x, att, selfk, selfg, selfb, selfw):
#         return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

#     def forward(self, inp):
#         x = inp

#         x = self.norm1(x)
#         sca = self.sca(x)

#         # KBA module
#         att = self.conv2(x) * self.attgamma + self.conv211(x)
#         uf = self.conv21(self.conv1(x))
#         x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
#         x = self.KBA(x, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
#         x = x * sca

#         x = self.conv3(x)
#         x = self.dropout1(x)
#         y = inp + x * self.beta

#         # FFN
#         x = self.norm2(y)
#         x = self.conv4(x)
#         x = self.sg(x)
#         x = self.conv5(x)

#         x = self.dropout2(x)
#         return y + x * self.gamma
# class KBNet_s(nn.Module):
#     def __init__(self, img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
#                  dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s',basicblock2='KBBlock_sss',basicblock3='KBBlock_ss', lightweight=False, ffn_scale=2):
#         super().__init__()
#         basicblock = eval(basicblock)
#         #basicblock2 = eval(basicblock2)
#         basicblock3 = eval(basicblock3)
#         self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
#                                groups=1, bias=True)

#         self.encoders = nn.ModuleList()
#         self.middle_blks = nn.ModuleList()
#         self.decoders = nn.ModuleList()

#         self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
#                                 groups=1, bias=True)

#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()

#         chan = width
#         for num in enc_blk_nums:
#             self.encoders.append(
#                 nn.Sequential(
#                     *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
#                 )
#             )
#             self.downs.append(
#                 nn.Conv2d(chan, 2 * chan, 2, 2)
#             )
#             chan = chan * 2

#         self.middle_blks = \
#             nn.Sequential(
#                 *[basicblock3(chan, FFN_Expand=ffn_scale) for _ in range(middle_blk_num)]
#             )

#         for num in dec_blk_nums:
#             self.ups.append(
#                 nn.Sequential(
#                     nn.Conv2d(chan, chan * 2, 1, bias=False),
#                     nn.PixelShuffle(2)
#                 )
#             )
#             chan = chan // 2
#             self.decoders.append(
#                 nn.Sequential(
#                     *[basicblock3(chan, FFN_Expand=ffn_scale) for _ in range(num)]
#                 )
#             )

#         self.padder_size = 2 ** len(self.encoders)

#     def forward(self, inp):
#         B, C, H, W = inp.shape
#         inp = self.check_image_size(inp)
#         x = self.intro(inp)

#         encs = []

#         for encoder, down in zip(self.encoders, self.downs):
#             x = encoder(x)
#             encs.append(x)
#             x = down(x)

#         x = self.middle_blks(x)

#         for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
#             x = up(x)
#             x = x + enc_skip
#             x = decoder(x)

#         x = self.ending(x)
#         x = x + inp

#         return x[:, :, :H, :W]

#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
#         mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
#         return x

# if __name__ == '__main__':
#     from ptflops import get_model_complexity_info
#     from arch_util import measure_inference_speed

    
#     img_channel = 3

#     width = 64
#     enc_blks = [2, 2, 4, 8]
#     middle_blk_num = 12
#     dec_blks = [2, 2, 2, 2]

#     # width = 8
#     # enc_blks = [1, 1, 1, 1]
#     # middle_blk_num = 1
#     # dec_blks = [1, 1, 1, 1]
    
#     GCE_CONVS_nums = [3,3,2,2]


#     net = KBNet_s(img_channel=img_channel,width=width, middle_blk_num=middle_blk_num,
#                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


#     inp_shape = (3, 256, 256)
#     macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
#     print(macs, params)
    
    
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"
#     data = torch.randn((1, *inp_shape))
#     print(device)
#     #measure_inference_speed(net.to(device), (data.to(device),), max_iter=500, log_interval=50)
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from basicsr.models.archs.kb_utils import KBAFunction
from basicsr.models.archs.kb_utils import LayerNorm2d, SimpleGate


class KBBlock_s(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

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
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma


class KBNet_s(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=False, ffn_scale=2):
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
                *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(middle_blk_num)]
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
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
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
        '''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from basicsr.models.archs.kb_utils import KBAFunction
from basicsr.models.archs.kb_utils import LayerNorm2d, SimpleGate
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 0, stide = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stide, padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class UpsampleWithFlops(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
        self.__flops__ = 0

    def forward(self, input):
        self.__flops__ += input.numel()
        return super(UpsampleWithFlops, self).forward(input)

class GlobalContextExtractor(nn.Module):
    def __init__(self, c, kernel_sizes=[3,3,5], strides=[3,3,5], padding=0, bias=False):
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
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.Conv2d(c, c, kernel_size=5, padding=2, stride=1, groups=c // 4, bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
            )

        self.conv1 = nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv21 = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, interc, kernel_size=3, padding=1, stride=1, groups=interc, bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        
        self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[1, 3], strides=[1,3])
        self.project_out = nn.Conv2d(dw_ch * 2, c, kernel_size=1)
        self.project_out2 = nn.Conv2d(c, self.nset, kernel_size=1,padding=0, stride=1)
        self.sca2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=int(dw_ch*1.5), out_channels=dw_ch*2, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))

        self.conv211 = nn.Conv2d(c, self.nset, kernel_size=1)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv01 = nn.Conv2d(in_channels=c, out_channels=dw_ch, kernel_size=1,
                                padding=0, stride=1, groups=1, bias=True)
        self.conv02 = nn.Conv2d(in_channels=dw_ch, out_channels=dw_ch,
                                kernel_size=3, padding=1, stride=1, groups=dw_ch,
                               bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

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
        b,c,h,w = x.shape
        self.upsample = UpsampleWithFlops(size=(h,w), mode='nearest')
        x = self.norm1(x)
        sca = self.sca(x)
         # Global Context Extractor + Range fusion
        a=self.conv01(x)
        a=self.conv02(a)
        a=F.gelu(a)
        x_1, x_2 = a.chunk(2, dim=1)
        x1, x2 = self.GCE(x_1 + x_2)
        a = torch.cat([x, self.upsample(x1), self.upsample(x2)], dim = 1)
        a = self.sca2(a)
        a = self.project_out(a)
        x = a * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma


class KBBlock_sss(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
        super(KBBlock_sss, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv33 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=2, stride=1,
                               dilation=2, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

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
        # KBA module
        att = self.conv2(x) * self.attgamma
        uf= self.conv11(x)
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        x = x * sca 

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma

class KBNet_s(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s',basicblock3='KBBlock_sss',lightweight=False, ffn_scale=2):
        super().__init__()
        basicblock = eval(basicblock)
        basicblock3 = eval(basicblock3)
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
                    *[basicblock3(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basicblock3(chan, FFN_Expand=ffn_scale) for _ in range(middle_blk_num)]
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
                    *[basicblock3(chan, FFN_Expand=ffn_scale) for _ in range(num)]
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
    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 10
    dec_blks = [2, 2, 2, 2]

    # width = 8
    # enc_blks = [1, 1, 1, 1]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = KBNet_s(img_channel=img_channel,width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    print(macs, params)
    
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    data = torch.randn((1, *inp_shape))
    print(device)
    #measure_inference_speed(net.to(device), (data.to(device),), max_iter=500, log_interval=50)