# import torch
# import torch.nn as nn
# import torch.linalg
# import torch.nn.init as init
# from torch.nn import functional as F
# import numpy as np
# from torch.nn import Transformer
# import math
# # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# # from mamba_ssm.modules.mamba_simple import Mamba
# # from einops import repeat
# from functools import partial
# from typing import Optional, Callable
#
# criterion = nn.CrossEntropyLoss()
# torch.set_printoptions(profile='full')
# from itertools import combinations
#
# l1_loss = nn.L1Loss()
#
#
# def custom_cdl_loss(p, m):
#     """
#     自定义CDL损失函数实现。
#
#     参数:
#         p (torch.Tensor): 预测概率矩阵，大小为 (h, w)
#         m (torch.Tensor): 真值标签矩阵，大小为 (h, w)
#
#     返回:
#         loss (torch.Tensor): 标量损失值
#     """
#     N = p.numel()  # N = h*w
#     loss = F.binary_cross_entropy(p, m, reduction="sum") / N
#     return loss
#
#
# def is_tensor_valid(tensor):
#     return not torch.isnan(tensor).any().item()
#
#
# class Conv1(nn.Module):
#     def __init__(self, C_in, C_out):
#         super(Conv1, self).__init__()
#         self.layer = nn.Sequential(
#
#             nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(C_out, affine=True),
#             # 防止过拟合
#             nn.ReLU(inplace=True),
#             nn.Conv2d(C_out, C_out, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(C_out, affine=True),
#             # 防止过拟合
#             nn.LeakyReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16, min_mid=4, use_max=True):
#         super().__init__()
#         mid = max(min_mid, in_channels // reduction)
#         self.use_max = use_max
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, mid, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid, in_channels, 1, bias=False)
#         )
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg = F.adaptive_avg_pool2d(x, 1)
#         out = self.mlp(avg)
#         if self.use_max:
#             mx = F.adaptive_max_pool2d(x, 1)
#             out = out + self.mlp(mx)
#         w = self.act(out)
#         # 残差式缩放避免过抑制
#         return x * (1 + w)
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7, add_std=False):
#         super().__init__()
#         self.add_std = add_std
#         in_c = 2 + (1 if add_std else 0)
#         padding = kernel_size // 2
#         self.conv = nn.Conv2d(in_c, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg = x.mean(1, keepdim=True)
#         mx = x.amax(1, keepdim=True)
#         cat = [avg, mx]
#         if self.add_std:
#             std = x.var(1, keepdim=True, unbiased=False).sqrt()
#             cat.append(std)
#         cat = torch.cat(cat, dim=1)
#         w = self.act(self.conv(cat))
#         return x * (1 + w)
#
#
# # 核心的AdaIN函数
# def adaptive_instance_normalization_per_sample(content, style):
#     # per-sample statistics
#     c_mu = content.mean(dim=(2,3), keepdim=True)
#     c_std = content.var(dim=(2,3), keepdim=True, unbiased=False).sqrt() + 1e-5
#     s_mu = style.mean(dim=(2,3), keepdim=True)
#     s_std = style.var(dim=(2,3), keepdim=True, unbiased=False).sqrt() + 1e-5
#     normalized = (content - c_mu) / c_std
#     return normalized * s_std + s_mu
#
#
# class convarg(nn.Module):
#     def __init__(self, C_in, C_out, k=0):
#         super(convarg, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(C_out, C_out, 1, 1, 0),
#             nn.InstanceNorm2d(C_out),
#             # 防止过拟合
#             nn.ReLU(inplace=True),
#         )
#         self.ca = ChannelAttention(C_out)
#         self.sa = SpatialAttention(k)
#         self.decoder = nn.Sequential(
#             nn.Conv2d(C_out, C_out, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(C_out, C_out, 3, 1, 1)
#         )
#
#     def forward(self, x, y):
#         out = self.layer(x)
#         # out = out * self.ca(out)
#         # out = out * self.sa(out)
#
#         stylized_feature = adaptive_instance_normalization_per_sample(out, y)
#
#         ans = self.ca(stylized_feature)
#         ans = self.sa(ans)
#         ans = self.decoder(ans)
#         return ans + out
#
#
# # 下采样模块
# class DownSampling(nn.Module):
#     def __init__(self, C):
#         super(DownSampling, self).__init__()
#         self.Down = nn.Sequential(
#             # 使用卷积进行2倍的下采样，通道数不变
#             nn.Conv2d(C, C, 3, 2, 1),
#             nn.LeakyReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.Down(x)
#
#
# # 上采样模块
# class UpSampling(nn.Module):
#
#     def __init__(self, C):
#         super(UpSampling, self).__init__()
#         # 特征图大小扩大2倍，通道数减半
#         self.Up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(C, C // 2, 3, 1, 1),
#             nn.InstanceNorm2d(C // 2, affine=True),
#             nn.LeakyReLU(inplace=True),
#         )
#
#     def forward(self, x, r):
#         # 使用邻近插值进行下采样
#         x = self.Up(x)
#         # 拼接，当前上采样的，和之前下采样过程中的
#         return torch.cat((x, r), 1)
#
#
# # encoder模块
# class es_i(nn.Module):
#     def __init__(self):
#         super(es_i, self).__init__()
#         self.inc = Conv1(2, 64)
#         self.down1 = DownSampling(64)
#         self.c2 = Conv1(64, 128)
#         self.down2 = DownSampling(128)
#         self.c3 = Conv1(128, 256)
#         self.down3 = DownSampling(256)
#         self.c4 = Conv1(256, 512)
#         self.down4 = DownSampling(512)
#         self.c5 = Conv1(512, 1024)
#
#     def forward(self, x):
#         features = []
#         x1 = self.inc(x)
#         features.append(x1)
#         x2 = self.c2(self.down1(x1))
#         features.append(x2)
#         x3 = self.c3(self.down2(x2))
#         features.append(x3)
#         x4 = self.c4(self.down3(x3))
#         features.append(x4)
#         x5 = self.c5(self.down4(x4))
#         features.append(x5)
#         return features
#
#
# class up2(nn.Module):
#     def __init__(self, C):
#         super(up2, self).__init__()
#         self.Up = nn.Upsample(scale_factor=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(C, C // 2, 3, 1, 1),
#             nn.InstanceNorm2d(C // 2, affine=True),
#             nn.LeakyReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(self.Up(x))
#
#
# class dec_seg(nn.Module):
#     def __init__(self):
#         super(dec_seg, self).__init__()
#         self.u1 = UpSampling(1024)
#         self.c6 = Conv1(1024, 512)
#         self.u2 = UpSampling(512)
#         self.c7 = Conv1(512, 256)
#         self.u3 = UpSampling(256)
#         self.c8 = Conv1(256, 128)
#         self.u4 = UpSampling(128)
#         self.c9 = Conv1(128, 64)
#
#         self.outc = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0, bias=False)
#
#         self.conv12 = nn.Sequential(
#             nn.Conv2d(1024, 256, 3, 1, 1),  # 先保持通道数不变
#
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, 1, 1),  # 先保持通道数不变
#
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 128, 3, 1, 1),
#
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 4, 1, 1, 0),  # 再映射到类别数
#             nn.Sigmoid()
#         )
#
#         self.seta = nn.Conv2d(256, 1024, 1, 1, 0)
#         self.seta2 = nn.Conv2d(5120, 1024, 1, 1, 0)
#
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(256, 256 // 4, 1),
#             nn.ReLU(),
#             nn.Conv2d(256 // 4, 1, 1),
#         )
#
#     def attenavg(self, x):
#         weights = self.attention(x)
#         weights = torch.softmax(weights.view(weights.size(0)), dim=0)
#         return torch.sum(x * weights.view(-1, 1, 1, 1), dim=0, keepdim=True)
#
#     def forward(self, fuses, s_label, bank_f, bank_label, epoch):
#         x1, x2, x3, x4, x5 = fuses
#
#         loss, enhance_fs = self.CDL(x5, s_label, bank_f, bank_label)
#
#         x5 = self.seta2(torch.cat([x5, enhance_fs], dim=1))
#
#         o1 = self.c6(self.u1(x5, x4))
#         # o1 = self.con2(torch.cat([o1, d4], dim=1))
#         o2 = self.c7(self.u2(o1, x3))
#
#         # o2 = self.con3(torch.cat([o2, d3], dim=1))
#         o3 = self.c8(self.u3(o2, x2))
#
#         # o3 = self.con4(torch.cat([o3, d2], dim=1))
#         o4 = self.c9(self.u4(o3, x1))
#
#         # o4 = self.con5(torch.cat([o4, d1], dim=1))
#         out = self.outc(o4)
#
#         return out, loss
#
#     def CDL(self, cur_f, s_label, bank_fs, bank_label):
#         bank_size = bank_label.memory.shape[0]
#         batch_size = cur_f.shape[0]
#         # Flatten s_label
#
#         s_label = s_label.flatten(2)
#
#         # cur_f = self.conv1(cur_f)
#         cur_f_flatten = torch.softmax(self.conv12(cur_f), dim=1).flatten(2)
#
#         # Pre-compute all bank_f and bank_label projections
#
#         bank_label_flatten = bank_label.memory.flatten(2).permute(0, 2, 1)
#
#         bank_f_tensors = [bank_f.memory for bank_f in bank_fs]  # 预计算 conv1
#         bank_f_pre_tensors = [torch.softmax(self.conv12(bf), dim=1).flatten(2).permute(0, 2, 1) for bf in
#                               bank_f_tensors]
#
#         # 拆分batchsize进行处理
#         CDL_loss = 0.0
#         SF_list = []
#         for j in range(batch_size):
#             Mj = torch.bmm(bank_label_flatten.detach(), s_label[j].unsqueeze(0).repeat(bank_size, 1, 1))  # (100, N, N)
#
#             P_list = [torch.bmm(bank_f.detach(), cur_f_flatten[j].unsqueeze(0).repeat(bank_size, 1, 1)) for bank_f in
#                       bank_f_pre_tensors]
#             #
#             cdl_loss_j = sum([F.binary_cross_entropy(P, Mj, reduction="mean") for P in P_list]) / 4
#             # cdl_loss = F.mse_loss(P,M,reduction="mean")
#             sf_list_j = [torch.bmm(bf.flatten(2), P).reshape(-1, 256, 12, 12) for bf, P in zip(bank_f_tensors, P_list)]
#             sf_list_j = [self.seta(self.attenavg(sf)) for sf in sf_list_j]
#             SF_list.append(sf_list_j)
#             CDL_loss += cdl_loss_j
#
#         seg_f_list = []
#         for i in range(len(SF_list[0])):
#             tmp_f = []
#             for k in range(len(SF_list)):
#                 tmp_f.append(SF_list[k][i])
#             seg_f_list.append(torch.cat(tmp_f, dim=0))
#         return CDL_loss / 4, torch.cat(seg_f_list, dim=1)
#
#
# class dec_i(nn.Module):
#     def __init__(self, UpSampling, Conv1, convarg, up2):
#         super(dec_i, self).__init__()
#
#         # Decoder路径构建
#         channels = [1024, 512, 256, 128]
#         out_channels = [512, 256, 128, 64]
#         self.upsample_blocks = nn.ModuleList([UpSampling(ch) for ch in channels])
#         self.conv_blocks = nn.ModuleList([Conv1(in_c, out_c) for in_c, out_c in zip(channels, out_channels)])
#         self.out_conv = nn.Conv2d(64, 1, 3, 1, 1, bias=False)
#
#         # 模态融合结构
#         fusion_in = [128, 256, 512, 1024, 2048]
#         fusion_out = [64, 128, 256, 512, 1024]
#         # self.modal_fusion = nn.ModuleList([
#         #     nn.ModuleList([convarg(f_in, f_out) for _ in range(4)])
#         #     for f_in, f_out in zip(fusion_in, fusion_out)
#         # ])
#         self.modal_fusion = nn.ModuleList([
#             nn.ModuleList([convarg(128, 64, 7) for _ in range(4)]),
#             nn.ModuleList([convarg(256, 128, 7) for _ in range(4)]),
#             nn.ModuleList([convarg(512, 256, 7) for _ in range(4)]),
#             nn.ModuleList([convarg(1024, 512, 7) for _ in range(4)]),
#             nn.ModuleList([convarg(2048, 1024, 7) for _ in range(4)]),
#         ])
#
#         self.convs = nn.ModuleList([
#             nn.Conv2d(f_in, f_out, 1) for f_in, f_out in zip([256, 512, 1024, 2048, 4096], fusion_out)
#         ])
#
#         self.bank_upsample = nn.ModuleList([up2(c) for c in channels])  # 4层upsample用于生成bank
#
#     def forward(self, fuses, idx, miss_index, bank=None):
#         fea_loss = 0.0
#         if bank is not None:
#             fusion_features, fea_loss = self.fusion(fuses, miss_index, bank, idx)
#             x1, x2, x3, x4, x5 = fusion_features
#         else:
#             x1, x2, x3, x4, x5 = fuses[idx]
#
#         # Decoder阶段
#         x = x5
#         for up, conv, skip in zip(self.upsample_blocks, self.conv_blocks, [x4, x3, x2, x1]):
#             x = conv(up(x, skip))
#         out = self.out_conv(x)
#         return torch.tanh(out), fea_loss
#
#     def fusion(self, features, miss, bank, idx):
#         real_features = [f.detach() for f in features[idx]]
#         loss = 0.0
#         batch_size = real_features[0].size(0)
#         fea_layer_banks = [[None] * 5 for _ in range(4)]
#
#         for j, m in enumerate(miss):
#             if m == 1:
#                 # 拆分batchsize进行处理
#                 bank_j = bank[j].memory.view(bank[j].memory.size(0), -1)
#                 tmp_f = []
#                 for k in range(batch_size):
#                     f_j_flat = features[j][4][k].unsqueeze(0).view(1, -1)
#
#                     # # 找欧氏距离最小的索引（更快的替代方式）
#                     # dists = torch.norm(f_j_flat - bank_j, dim=1)
#                     # topk_index = torch.argmin(dists)
#
#                     # f_j_flat = F.normalize(features[j][4].view(1, -1), dim=-1)  # (1, d)
#
#                     # 2) 把 memory bank 展平成 (N, d) 并做归一化
#                     bank_j = F.normalize(bank[j].memory.view(bank[j].memory.size(0), -1), dim=-1)  # (N, d)
#
#                     # 3) 计算相似度 (scaled dot-product，可选 /sqrt(d))
#                     sims = torch.matmul(f_j_flat, bank_j.t()).squeeze(0)  # (N,)
#                     # 若想模仿 Transformers，可做 sims /= math.sqrt(f_j_flat.shape[-1])
#
#                     topk_index = torch.argmax(sims)  # 相似度最大的索引
#
#                     base_feat = bank[idx].memory[topk_index].unsqueeze(0).detach()
#                     tmp_f.append(base_feat)
#                 # 将所有batch的base_feat堆叠起来
#                 base_feats = torch.cat(tmp_f, dim=0)
#                 # 取出距离最小的bank特征
#                 layer_banks = [base_feats]
#                 for up in self.bank_upsample:
#                     layer_banks.append(up(layer_banks[-1]))
#
#                 fea_layer_banks[j] = layer_banks
#
#         # 构造五层bank特征
#         # layer_banks = [bank.memory[-1].unsqueeze(0).detach()]
#         # for up in self.bank_upsample:
#         #     layer_banks.append(up(layer_banks[-1]))
#
#         counts = sum(miss)
#         fusion_features = []
#         cached_zeros = [torch.zeros_like(f) for f in real_features]  # 缓存用于填0l
#         loss = 0.0
#
#         for i in range(5):
#             inputs = []
#             loss_i = 0.0
#             if i < 2:
#                 for j, m in enumerate(miss):
#                     if m == 1:
#                         inputs.append(features[j][i])
#             else:
#                 for j, m in enumerate(miss):
#                     if m == 1:
#                         fused = self.modal_fusion[i][j](features[j][i], fea_layer_banks[j][4 - i])
#                         loss_i += F.l1_loss(fused, real_features[i])
#                         inputs.append(fused)
#                 # else:
#                 #     # 直接使用bank特征
#                 #     inputs.append(cached_zeros[i])
#
#             concat = sum(inputs) / len(inputs)  # self.convs[i](torch.cat(inputs, dim=1))
#             fusion_features.append(concat)
#
#             loss += loss_i / counts  # F.mse_loss(concat, real_features[i])
#         return fusion_features, loss
#
#
# class MemoryBank(nn.Module):
#     def __init__(self, memory_bank_sizes=100, feat_dims=1024, h=12, w=12):
#         super(MemoryBank, self).__init__()
#         self.memory_bank_size = memory_bank_sizes
#         self.feat_dims = feat_dims
#         self.h = h
#         self.w = w
#
#         # Initialize the memory bank
#         self.memory = nn.Parameter(torch.zeros(memory_bank_sizes, feat_dims, h, w))
#         self.count = 0
#
#     def initialize_memory_bank(self, input_feats):
#         with torch.no_grad():
#             # 拆分batch处理
#             for k in range(input_feats.size(0)):
#                 self.memory[self.count % self.memory_bank_size] = input_feats[k].unsqueeze(0)
#                 self.count += 1
#
#
# class Generator_teacher(nn.Module):
#     def __init__(self):
#         super(Generator_teacher, self).__init__()
#         # shared_layers = SharedLayers()
#         # self.encoders_c = es_c()
#         self.es = nn.ModuleList([es_i() for _ in range(4)])
#         # self.es_c = es_i()
#         # self.fum2 = feature_unification_module()
#         # self.fum1 = feature_unification_module()
#
#         # self.dec = nn.ModuleList([dec_i(UpSampling, Conv1, convarg, up2) for _ in range(4)])
#         self.seg = dec_seg()
#
#     def forward(self, x1, x2, x3, x4, label=None, pre_label=None, small_label=None, bank=None, l_bank=None,
#                 miss_index=None, epoch=None, seg=0,
#                 up=0):  # label=None,pre_label=None,small_label=None,f_bank=None,l_bank=None, miss_index=None,epoch=None,seg=0,up=0
#
#         x_input = [x1, x2, x3, x4]
#         count = sum(miss_index)
#         features = []
#         if seg == 1:
#             mod_input = []
#             pre_label = torch.zeros_like(x1)
#             # # commonfs = []
#             for i in range(4):
#                 x_temp = torch.cat([x_input[i], pre_label], dim=1)
#
#                 # comf = self.encoders_c(x_temp)
#                 spef = self.es[i](x_temp)
#
#                 # if(miss_index[i]==1):
#                 #     commonfs.append(comf)
#
#                 mod_input.append(spef)
#             seg_input = []
#
#             for j in range(5):
#                 layer_input = 0
#                 for idx, k in enumerate(miss_index):
#                     if k == 1:
#                         layer_input += mod_input[idx][j]
#                     # else:
#                     #     layer_input += torch.zeros_like(mod_input[idx][j])
#                 seg_input.append(layer_input / count)
#             seg_out, loss = self.seg(seg_input, small_label, bank, l_bank, epoch)
#
#             # cf_loss = pairwise_feature_alignment_loss(commonfs)
#             return seg_out,mod_input, loss
#
#         if epoch > 10:
#             tmp = pre_label
#         else:
#             tmp = label
#
#         for i in range(4):
#             x_temp = torch.cat([x_input[i], tmp], dim=1)
#             spef = self.es[i](x_temp)
#
#             features.append(spef)
#
#         outputs = [None, None, None, None]
#         # feature_loss = 0.0
#
#         real_loss = 0.0
#         for idx, k in enumerate(miss_index):
#             if k == 0:
#                 outputs[idx], fea_loss = self.dec[idx](features, idx, miss_index, bank)
#                 real_loss += fea_loss
#             else:
#                 outputs[idx], _ = self.dec[idx](features, idx, miss_index, None)
#
#         #outputs = torch.cat(outputs, dim=1)
#
#         # comf_loss2 = pairwise_feature_alignment_loss(commonfs2)
#
#         return outputs, features, real_loss
#
#
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, dataset='BRATS2020'):
#         super(Discriminator, self).__init__()
#
#         # inp, stride, pad, dil, kernel = (256, 2, 1, 1, 8)
#         # np.floor(((inp + 2*pad - dil*(kernel - 1) - 1)/stride) + 1)
#
#         if 'BRATS' in dataset:
#             def discriminator_block(in_filters, out_filters, normalization=True):
#                 """Returns downsampling layers of each discriminator block"""
#                 layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#                 if normalization:
#                     layers.append(nn.InstanceNorm2d(out_filters))
#                 layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 return layers
#
#             self.model = nn.Sequential(
#                 *discriminator_block(in_channels, 64, normalization=False),
#                 *discriminator_block(64, 128),
#                 *discriminator_block(128, 256),
#                 *discriminator_block(256, 512),
#                 nn.ZeroPad2d((1, 0, 1, 0)),
#                 nn.Conv2d(512, out_channels, 4, padding=1, bias=False)
#             )
#
#     def forward(self, img_A):
#         # Concatenate image and condition image by channels to produce input
#         return self.model(img_A)
#
#
# class Discriminators(nn.Module):
#     def __init__(self):
#         super(Discriminators, self).__init__()
#         self.dis = nn.ModuleList([Discriminator() for _ in range(4)])
#
#     def forward(self, imgs):
#         return [self.dis[i](imgs[i]) for i in range(4)]
#
#

import torch
import torch.nn as nn
import torch.linalg
import torch.nn.init as init
from torch.nn import functional as F
import numpy as np
from torch.nn import Transformer
import math
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# from mamba_ssm.modules.mamba_simple import Mamba
# from einops import repeat
from functools import partial
from typing import Optional, Callable

criterion = nn.CrossEntropyLoss()
torch.set_printoptions(profile='full')
from itertools import combinations

l1_loss = nn.L1Loss()


def custom_cdl_loss(p, m):
    """
    自定义CDL损失函数实现。

    参数:
        p (torch.Tensor): 预测概率矩阵，大小为 (h, w)
        m (torch.Tensor): 真值标签矩阵，大小为 (h, w)

    返回:
        loss (torch.Tensor): 标量损失值
    """
    N = p.numel()  # N = h*w
    loss = F.binary_cross_entropy(p, m, reduction="sum") / N
    return loss


def is_tensor_valid(tensor):
    return not torch.isnan(tensor).any().item()


class Conv1(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv1, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(C_out, affine=True),
            # 防止过拟合
            nn.ReLU(inplace=True),
            nn.Conv2d(C_out, C_out, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(C_out, affine=True),
            # 防止过拟合
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channels = max(1, in_channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg_out + max_out)


# Spatial Attention with efficient kernel
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):  # kernel 5 instead of 7 for efficiency
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # reduce across channel dimension using built-in ops
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


# 核心的AdaIN函数
def adaptive_instance_normalization(content_feat, style_mean, style_std):
    # 归一化内容特征
    content_mean = content_feat.mean(dim=(2, 3), keepdim=True)
    content_std = content_feat.std(dim=(2, 3), keepdim=True) + 1e-5
    normalized_content = (content_feat - content_mean) / content_std

    # 应用新的风格
    stylized_content = normalized_content * style_std + style_mean
    return stylized_content


class convarg(nn.Module):
    def __init__(self, C_in, C_out, k=0):
        super(convarg, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_out, C_out, 1, 1, 0),
            nn.InstanceNorm2d(C_out),
            # 防止过拟合
            nn.ReLU(inplace=True),
        )
        self.ca = ChannelAttention(C_out)
        self.sa = SpatialAttention(k)
        self.decoder = nn.Sequential(
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_out, C_out, 3, 1, 1)
        )

    def forward(self, x, y):
        out = self.layer(x)
        # out = out * self.ca(out)
        # out = out * self.sa(out)

        style_mean = y.mean(dim=(0, 2, 3), keepdim=True)
        style_std = y.std(dim=(0, 2, 3), keepdim=True) + 1e-5

        stylized_feature = adaptive_instance_normalization(out, style_mean, style_std)

        ans = self.ca(stylized_feature) * stylized_feature
        ans = self.sa(ans) * ans

        return ans


class TransformerDecoderBlock(nn.Module):
    """
    一个标准的 Transformer Decoder Block (Pre-LN 结构)
    包含: 1. Masked Self-Attention, 2. Cross-Attention, 3. Feed-Forward Network
    """

    def __init__(self, embed_dim, num_heads, ffn_dim_multiplier=4, dropout=0.1):
        """
        参数:
        - embed_dim (int): 输入特征的维度 (即 C)
        - num_heads (int): 多头注意力的头数
        - ffn_dim_multiplier (int): FFN 中间隐藏层的维度相对于 embed_dim 的倍数
        - dropout (float): Dropout 的比例
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. 带掩码的自注意力层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # 2. 交叉注意力层
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        # 3. 前馈神经网络
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_dim_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_dim_multiplier, embed_dim),
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        参数:
        - tgt (Tensor): Decoder 的输入, 形状为 (B, Seq_len_tgt, C)
        - memory (Tensor): Encoder 的输出, 形状为 (B, Seq_len_mem, C)
        - tgt_mask (Tensor, optional): 自注意力掩码, 防止信息泄露.
        - memory_mask (Tensor, optional): 交叉注意力掩码.
        """
        # --- 1. 带掩码的自注意力 ---
        # Pre-LN: 先 Norm
        tgt_norm = self.norm1(tgt)
        # Self-Attention
        sa_output, _ = self.self_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask)
        # 残差连接
        tgt = tgt + self.dropout1(sa_output)

        # --- 2. 交叉注意力 ---
        # Pre-LN: 先 Norm
        tgt_norm = self.norm2(tgt)
        # Cross-Attention
        ca_output, _ = self.cross_attn(tgt_norm, memory, memory, key_padding_mask=memory_mask)
        # 残差连接
        tgt = tgt + self.dropout2(ca_output)

        # --- 3. 前馈神经网络 ---
        # Pre-LN: 先 Norm
        tgt_norm = self.norm3(tgt)
        # FFN
        ffn_output = self.ffn(tgt_norm)
        # 残差连接
        tgt = tgt + self.dropout3(ffn_output)

        return tgt


class VisionTransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ffn_dim_multiplier=4, dropout=0.1):
        """
        参数:
        - num_layers (int): Decoder Block 的数量 (这里设置为6)
        - embed_dim (int): 特征维度 (C)
        - num_heads (int): 多头注意力的头数
        - ffn_dim_multiplier (int): FFN 中间层维度倍数
        - dropout (float): Dropout 比例
        """
        super().__init__()

        # 使用 nn.ModuleList 来存储所有的 Decoder Block
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ffn_dim_multiplier, dropout)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers
        self.norm = nn.LayerNorm(embed_dim)  # 在所有 Block 计算后进行最后一次 Norm

        # 保留您原有的下采样逻辑
        # self.downsample_thresh = 96
        # self.downsample_size = 48

    def forward(self, A, B):
        """
        这里的 A 对应 Transformer 的 target (tgt)
        这里的 B 对应 Transformer 的 memory (encoder_output)

        参数:
        - A (Tensor): Query 特征图, 形状为 (B, C, H, W)
        - B (Tensor): Key/Value 特征图, 形状为 (B, C, H_mem, W_mem)
        """
        B_batch, C, H, W = A.shape

        # 1. 准备输入：将 4D 特征图转换为 3D 序列
        # (B, C, H, W) -> (B, C, HW) -> (B, HW, C)
        tgt = A.flatten(2).transpose(1, 2)

        # 对 B (memory) 进行可能的下采样，并转换为 3D 序列
        # if max(B.shape[2], B.shape[3]) >= self.downsample_thresh:
        #     B_kv = F.adaptive_avg_pool2d(B, (self.downsample_size, self.downsample_size))
        # else:
        B_kv = B
        memory = B_kv.flatten(2).transpose(1, 2)

        # 2. 依次通过6个 Decoder Block
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)  # 在这个视觉任务场景下，通常不需要掩码

        # 3. 最后的 LayerNorm
        output = self.norm(output)

        # 4. 将输出转换回 4D 特征图的形状
        # (B, HW, C) -> (B, C, HW) -> (B, C, H, W)
        output = output.transpose(1, 2).view(B_batch, C, H, W)

        return output


class convarg2(nn.Module):
    def __init__(self, C_in, C_out):
        super(convarg2, self).__init__()
        # self.layer = nn.Sequential(
        #     nn.Conv2d(C_in, C_out, 1, 1, 0),
        #     nn.InstanceNorm2d(C_out),
        #     # 防止过拟合
        #     nn.ReLU(inplace=True),
        # )
        # # self.ca = ChannelAttention(C_out)
        # # self.sa = SpatialAttention()
        # self.config = MambaConfig(d_model=C_out,d_state=8, n_layers=2)
        # self.model = Mamba(self.config)
        self.vit = VisionTransformerDecoder(2, C_out, 2)

    def forward(self, x, y):
        # b,c,h,w = x.size()
        # x = self.layer(x)  # 卷积+归一化+激活

        # x_seq = x.flatten(2).transpose(1, 2)  # (b, h*w, c)
        # out = self.model(x_seq)              # Mamba 输入 (b, L, D)

        # out = out.transpose(1, 2).view(b, -1, h, w)
        # out = self.layer(x)
        # out = out * self.ca(out)
        # out = out * self.sa(out)

        return self.vit(x, y)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(C, C // 2, 3, 1, 1),
            nn.InstanceNorm2d(C // 2, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        x = self.Up(x)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# encoder模块
class es_i(nn.Module):
    def __init__(self):
        super(es_i, self).__init__()
        self.inc = Conv1(2, 64)
        self.down1 = DownSampling(64)
        self.c2 = Conv1(64, 128)
        self.down2 = DownSampling(128)
        self.c3 = Conv1(128, 256)
        self.down3 = DownSampling(256)
        self.c4 = Conv1(256, 512)
        self.down4 = DownSampling(512)
        self.c5 = Conv1(512, 1024)

    def forward(self, x):
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.c2(self.down1(x1))
        features.append(x2)
        x3 = self.c3(self.down2(x2))
        features.append(x3)
        x4 = self.c4(self.down3(x3))
        features.append(x4)
        x5 = self.c5(self.down4(x4))
        features.append(x5)
        return features


class up2(nn.Module):
    def __init__(self, C):
        super(up2, self).__init__()
        self.Up = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, 1, 1),
            nn.InstanceNorm2d(C // 2, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.Up(x))


class dec_seg(nn.Module):
    def __init__(self):
        super(dec_seg, self).__init__()
        self.u1 = UpSampling(1024)
        self.c6 = Conv1(1024, 512)
        self.u2 = UpSampling(512)
        self.c7 = Conv1(512, 256)
        self.u3 = UpSampling(256)
        self.c8 = Conv1(256, 128)
        self.u4 = UpSampling(128)
        self.c9 = Conv1(128, 64)

        self.outc = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv12 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, 1, 1),  # 先保持通道数不变

            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # 先保持通道数不变

            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),

            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, 1, 1, 0),  # 再映射到类别数
            nn.Sigmoid()
        )

        self.seta = nn.Conv2d(256, 1024, 1, 1, 0)
        self.seta2 = nn.Conv2d(5120, 1024, 1, 1, 0)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256 // 4, 1),
            nn.ReLU(),
            nn.Conv2d(256 // 4, 1, 1),
        )

    def attenavg(self, x):
        weights = self.attention(x)
        weights = torch.softmax(weights.view(weights.size(0)), dim=0)
        return torch.sum(x * weights.view(-1, 1, 1, 1), dim=0, keepdim=True)

    def forward(self, fuses, s_label, bank_f, bank_label, epoch):
        x1, x2, x3, x4, x5 = fuses

        loss, enhance_fs = self.CDL(x5, s_label, bank_f, bank_label)

        x5 = self.seta2(torch.cat([x5, enhance_fs], dim=1))

        o1 = self.c6(self.u1(x5, x4))
        # o1 = self.con2(torch.cat([o1, d4], dim=1))
        o2 = self.c7(self.u2(o1, x3))

        # o2 = self.con3(torch.cat([o2, d3], dim=1))
        o3 = self.c8(self.u3(o2, x2))

        # o3 = self.con4(torch.cat([o3, d2], dim=1))
        o4 = self.c9(self.u4(o3, x1))

        # o4 = self.con5(torch.cat([o4, d1], dim=1))
        out = self.outc(o4)

        return out, loss

    def CDL(self, cur_f, s_label, bank_fs, bank_label):
        bank_size = bank_label.memory.shape[0]
        batch_size = cur_f.shape[0]
        # Flatten s_label

        s_label = s_label.flatten(2)

        # cur_f = self.conv1(cur_f)
        cur_f_flatten = torch.softmax(self.conv12(cur_f), dim=1).flatten(2)

        # Pre-compute all bank_f and bank_label projections

        bank_label_flatten = bank_label.memory.flatten(2).permute(0, 2, 1)

        bank_f_tensors = [bank_f.memory for bank_f in bank_fs]  # 预计算 conv1
        bank_f_pre_tensors = [torch.softmax(self.conv12(bf), dim=1).flatten(2).permute(0, 2, 1) for bf in
                              bank_f_tensors]

        # 拆分batchsize进行处理
        CDL_loss = 0.0
        SF_list = []
        for j in range(batch_size):
            Mj = torch.bmm(bank_label_flatten.detach(), s_label[j].unsqueeze(0).repeat(bank_size, 1, 1))  # (100, N, N)

            P_list = [torch.bmm(bank_f.detach(), cur_f_flatten[j].unsqueeze(0).repeat(bank_size, 1, 1)) for bank_f in
                      bank_f_pre_tensors]
            #
            cdl_loss_j = sum([F.binary_cross_entropy(P, Mj, reduction="mean") for P in P_list]) / 4
            # cdl_loss = F.mse_loss(P,M,reduction="mean")
            sf_list_j = [torch.bmm(bf.flatten(2), P).reshape(-1, 256, 12, 12) for bf, P in zip(bank_f_tensors, P_list)]
            sf_list_j = [self.seta(self.attenavg(sf)) for sf in sf_list_j]
            SF_list.append(sf_list_j)
            CDL_loss += cdl_loss_j

        seg_f_list = []
        for i in range(len(SF_list[0])):
            tmp_f = []
            for k in range(len(SF_list)):
                tmp_f.append(SF_list[k][i])
            seg_f_list.append(torch.cat(tmp_f, dim=0))
        return CDL_loss / 4, torch.cat(seg_f_list, dim=1)


class dec_i(nn.Module):
    def __init__(self, UpSampling, Conv1, convarg, up2):
        super(dec_i, self).__init__()

        # Decoder路径构建
        channels = [1024, 512, 256, 128]
        out_channels = [512, 256, 128, 64]
        self.upsample_blocks = nn.ModuleList([UpSampling(ch) for ch in channels])
        self.conv_blocks = nn.ModuleList([Conv1(in_c, out_c) for in_c, out_c in zip(channels, out_channels)])
        self.out_conv = nn.Conv2d(64, 1, 3, 1, 1, bias=False)

        # 模态融合结构
        fusion_in = [128, 256, 512, 1024, 2048]
        fusion_out = [64, 128, 256, 512, 1024]
        # self.modal_fusion = nn.ModuleList([
        #     nn.ModuleList([convarg(f_in, f_out) for _ in range(4)])
        #     for f_in, f_out in zip(fusion_in, fusion_out)
        # ])
        self.modal_fusion = nn.ModuleList([
            nn.ModuleList([convarg(128, 64, 7) for _ in range(4)]),
            nn.ModuleList([convarg(256, 128, 7) for _ in range(4)]),
            nn.ModuleList([convarg(512, 256, 7) for _ in range(4)]),
            nn.ModuleList([convarg(1024, 512, 7) for _ in range(4)]),
            nn.ModuleList([convarg(2048, 1024, 7) for _ in range(4)]),
        ])

        self.convs = nn.ModuleList([
            nn.Conv2d(f_in, f_out, 1) for f_in, f_out in zip([256, 512, 1024, 2048, 4096], fusion_out)
        ])

        self.bank_upsample = nn.ModuleList([up2(c) for c in channels])  # 4层upsample用于生成bank

    def forward(self, fuses, idx, miss_index, bank=None):
        fea_loss = 0.0
        if bank is not None:
            fusion_features, fea_loss = self.fusion(fuses, miss_index, bank, idx)
            x1, x2, x3, x4, x5 = fusion_features
        else:
            x1, x2, x3, x4, x5 = fuses[idx]

        # Decoder阶段
        x = x5
        for up, conv, skip in zip(self.upsample_blocks, self.conv_blocks, [x4, x3, x2, x1]):
            x = conv(up(x, skip))
        out = self.out_conv(x)
        return torch.tanh(out), fea_loss

    def fusion(self, features, miss, bank, idx):
        real_features = [f.detach() for f in features[idx]]
        loss = 0.0
        batch_size = real_features[0].size(0)
        fea_layer_banks = [[None] * 5 for _ in range(4)]

        for j, m in enumerate(miss):
            if m == 1:
                # 拆分batchsize进行处理
                bank_j = bank[j].memory.view(bank[j].memory.size(0), -1)
                tmp_f = []
                for k in range(batch_size):
                    f_j_flat = features[j][4][k].unsqueeze(0).view(1, -1)

                    # # 找欧氏距离最小的索引（更快的替代方式）
                    # dists = torch.norm(f_j_flat - bank_j, dim=1)
                    # topk_index = torch.argmin(dists)

                    # f_j_flat = F.normalize(features[j][4].view(1, -1), dim=-1)  # (1, d)

                    # 2) 把 memory bank 展平成 (N, d) 并做归一化
                    bank_j = F.normalize(bank[j].memory.view(bank[j].memory.size(0), -1), dim=-1)  # (N, d)

                    # 3) 计算相似度 (scaled dot-product，可选 /sqrt(d))
                    sims = torch.matmul(f_j_flat, bank_j.t()).squeeze(0)  # (N,)
                    # 若想模仿 Transformers，可做 sims /= math.sqrt(f_j_flat.shape[-1])

                    topk_index = torch.argmax(sims)  # 相似度最大的索引

                    base_feat = bank[idx].memory[topk_index].unsqueeze(0).detach()
                    tmp_f.append(base_feat)
                # 将所有batch的base_feat堆叠起来
                base_feats = torch.cat(tmp_f, dim=0)
                # 取出距离最小的bank特征
                layer_banks = [base_feats]
                for up in self.bank_upsample:
                    layer_banks.append(up(layer_banks[-1]))

                fea_layer_banks[j] = layer_banks

        # 构造五层bank特征
        # layer_banks = [bank.memory[-1].unsqueeze(0).detach()]
        # for up in self.bank_upsample:
        #     layer_banks.append(up(layer_banks[-1]))

        counts = sum(miss)
        fusion_features = []
        cached_zeros = [torch.zeros_like(f) for f in real_features]  # 缓存用于填0l
        loss = 0.0

        for i in range(5):
            inputs = []
            loss_i = 0.0
            if i < 2:
                for j, m in enumerate(miss):
                    if m == 1:
                        inputs.append(features[j][i])
            else:
                for j, m in enumerate(miss):
                    if m == 1:
                        fused = self.modal_fusion[i][j](features[j][i], fea_layer_banks[j][4 - i])
                        loss_i += F.l1_loss(fused, real_features[i])
                        inputs.append(fused)
                # else:
                #     # 直接使用bank特征
                #     inputs.append(cached_zeros[i])

            concat = sum(inputs) / len(inputs)  # self.convs[i](torch.cat(inputs, dim=1))
            fusion_features.append(concat)

            loss += loss_i / counts  # F.mse_loss(concat, real_features[i])
        return fusion_features, loss


class MemoryBank(nn.Module):
    def __init__(self, memory_bank_sizes=100, feat_dims=1024, h=12, w=12):
        super(MemoryBank, self).__init__()
        self.memory_bank_size = memory_bank_sizes
        self.feat_dims = feat_dims
        self.h = h
        self.w = w

        # Initialize the memory bank
        self.memory = nn.Parameter(torch.zeros(memory_bank_sizes, feat_dims, h, w))
        self.count = 0

    def initialize_memory_bank(self, input_feats):
        with torch.no_grad():
            # 拆分batch处理
            for k in range(input_feats.size(0)):
                self.memory[self.count % self.memory_bank_size] = input_feats[k].unsqueeze(0)
                self.count += 1


class Generator_teacher(nn.Module):
    def __init__(self):
        super(Generator_teacher, self).__init__()
        # shared_layers = SharedLayers()
        # self.encoders_c = es_c()
        self.es = nn.ModuleList([es_i() for _ in range(4)])
        # self.es_c = es_i()
        # self.fum2 = feature_unification_module()
        # self.fum1 = feature_unification_module()

        self.dec = nn.ModuleList([dec_i(UpSampling, Conv1, convarg, up2) for _ in range(4)])
        self.seg = dec_seg()

    def forward(self, x1, x2, x3, x4, label=None, pre_label=None, small_label=None, bank=None, l_bank=None,
                miss_index=None, epoch=None, seg=0,
                up=0):  # label=None,pre_label=None,small_label=None,f_bank=None,l_bank=None, miss_index=None,epoch=None,seg=0,up=0

        x_input = [x1, x2, x3, x4]
        count = sum(miss_index)
        features = []
        if seg == 1:
            mod_input = []
            pre_label = torch.zeros_like(x1)
            # # commonfs = []
            for i in range(4):
                x_temp = torch.cat([x_input[i], pre_label], dim=1)

                # comf = self.encoders_c(x_temp)
                spef = self.es[i](x_temp)

                # if(miss_index[i]==1):
                #     commonfs.append(comf)

                mod_input.append(spef)
            seg_input = []

            for j in range(5):
                layer_input = 0
                for idx, k in enumerate(miss_index):
                    if k == 1:
                        layer_input += mod_input[idx][j]
                    # else:
                    #     layer_input += torch.zeros_like(mod_input[idx][j])
                seg_input.append(layer_input / count)
            seg_out, loss = self.seg(seg_input, small_label, bank, l_bank, epoch)

            # cf_loss = pairwise_feature_alignment_loss(commonfs)
            return seg_out, loss

        if epoch > 10:
            tmp = pre_label
        else:
            tmp = label

        for i in range(4):
            x_temp = torch.cat([x_input[i], tmp], dim=1)
            spef = self.es[i](x_temp)

            features.append(spef)

        outputs = [None, None, None, None]
        # feature_loss = 0.0

        real_loss = 0.0
        for idx, k in enumerate(miss_index):
            if k == 0:
                outputs[idx], fea_loss = self.dec[idx](features, idx, miss_index, bank)
                real_loss += fea_loss
            else:
                outputs[idx], _ = self.dec[idx](features, idx, miss_index, None)

        outputs = torch.cat(outputs, dim=1)

        # comf_loss2 = pairwise_feature_alignment_loss(commonfs2)

        return outputs, features, real_loss


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dataset='BRATS2020'):
        super(Discriminator, self).__init__()

        # inp, stride, pad, dil, kernel = (256, 2, 1, 1, 8)
        # np.floor(((inp + 2*pad - dil*(kernel - 1) - 1)/stride) + 1)

        if 'BRATS' in dataset:
            def discriminator_block(in_filters, out_filters, normalization=True):
                """Returns downsampling layers of each discriminator block"""
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                if normalization:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *discriminator_block(in_channels, 64, normalization=False),
                *discriminator_block(64, 128),
                *discriminator_block(128, 256),
                *discriminator_block(256, 512),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(512, out_channels, 4, padding=1, bias=False)
            )

    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        return self.model(img_A)


class Discriminators(nn.Module):
    def __init__(self):
        super(Discriminators, self).__init__()
        self.dis = nn.ModuleList([Discriminator() for _ in range(4)])

    def forward(self, imgs):
        return [self.dis[i](imgs[i]) for i in range(4)]


