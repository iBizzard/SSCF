# -*- encoding:utf-8 -*-
import csv


def denorm(x):
    res = (x + 1.) / 2.
    res.clamp_(0, 1)
    return res


# 逐步调整学习率的函数
# def adjust_learning_rate(optimizer, epoch, initial_lr, total_epochs):
#     """在前50个epoch中保持初始学习率，之后线性下降到0"""
#     if epoch >= 50:
#         lr = initial_lr * (1 - (epoch - 50) / (total_epochs - 50))
#     else:
#         lr = initial_lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import datetime
import itertools
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch

from dataloader3 import *
from model0815 import *

import random
# from torch.utils.data import DataLoader
# from utils import *
import argparse
import time
import torch.nn as nn
# from torch.cuda.amp import GradScaler, autocast
import cv2
import os
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一�?


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


set_seed(1234)

def get_value(val):
    return val if isinstance(val, float) else val.item()


def diceloss(outputs, targets):
    nums = len(targets)
    loss = 0.0
    smooth = 1e-6
    for i in range(nums):
        output = outputs[:, i, :, :]
        target = targets[i]
        intersection = torch.sum(output * target)
        dice = (2. * intersection + smooth) / (torch.sum(output) + torch.sum(target) + smooth)
        loss += 1 - dice
    return loss / nums

memory_bank_sizes = 300

features_banks = nn.ModuleList(
    [MemoryBank(memory_bank_sizes, 1024) for _ in range(4)])  # MemoryBank(memory_bank_sizes, 1024)
label_banks = MemoryBank(memory_bank_sizes, 4)


def crosloss(outputs, targets):
    targets = targets * 4  # 0,1,2,4
    targets[targets == 4] = 3

    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)  # (B, H, W)

    # 2) reshape 成 (N, 4) vs. (N,)
    # outputs : (B, 4, H, W) → (B, H, W, 4) → (N, 4)
    # outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 4)
    # targets  : (B, H, W)   → (N,)
    # targets = targets.view(-1).long()

    # # 3) 计算交叉熵
    # return criterion(outputs, targets)

    cross = 0.0
    for j in range(targets.size(0)):
        tar = targets[j].view(-1).long()
        out = outputs[j].permute(1, 2, 0).contiguous().view(-1, 4)
        cross += criterion(out, tar)
    return cross
    # targets = targets.squeeze(0).squeeze(0).view(-1).long()
    # outputs = outputs.squeeze(0)
    # outputs = outputs.permute(1, 2, 0).contiguous().view(-1, 4)
    # return criterion(outputs, targets)


def label_norm(v):
    max_value = torch.max(v)
    min_value = torch.min(v)
    value_range = max_value - min_value
    if value_range == 0:
        v = v - 1
        return v
    else:
        norm_0_1 = (v - min_value) / value_range
    return torch.clamp(2 * norm_0_1 - 1, -1, 1)


def binary_dice_score(pred_bin: torch.Tensor,
                      true_bin: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """
    计算单通道下的 Dice 系数。pred_bin, true_bin 形状均为 (N, H, W)，
    值为 0/1 的二值掩码。返回 (N,) 长度的 Tensor，表示每张图的 Dice。
    """
    # 把 pred_bin, true_bin 都视作 float 张量
    pred_f = pred_bin.float()
    true_f = true_bin.float()

    # 按每张图分别计算 intersection 和 平台和
    # intersection_i = ∑_{h,w} pred_f[i,h,w] * true_f[i,h,w]
    # sum_i = ∑ pred_f[i] + ∑ true_f[i]
    # intersection = (pred_f * true_f).view(pred_f.shape[0], -1).sum(dim=1)  # (N,)
    # sum_pred = pred_f.view(pred_f.shape[0], -1).sum(dim=1)               # (N,)
    # sum_true = true_f.view(true_f.shape[0], -1).sum(dim=1)               # (N,)
    intersection = torch.sum(pred_f * true_f)
    dice_per_image = (2 * intersection + eps) / (torch.sum(pred_f) + torch.sum(true_f) + eps)  # (N,)
    return dice_per_image  # 返回每张图的 Dice 值

def contrastive_loss_with_clamp(features, features_banks, label_scenario, topk,
                                temperature=0.1, eps=1e-6, chunk_size=1024):
    device = features[0][0].device
    B = features[0][0].shape[0]

    # 1. 预处理 bank：扁平化、to(device)
    pos_banks = {}
    neg_banks = {}
    for u, flag in enumerate(label_scenario):
        if flag != 1:
            continue
        # 正样本
        pos = features_banks[u].memory[-topk:].view(topk, -1).to(device)
        # 手动归一化
        pos_norms = pos.norm(dim=1, keepdim=True)
        pos = pos / (pos_norms + eps)
        pos_banks[u] = pos

        # 负样本：拼接其他所有模态
        neg_list = []
        for u3, f3 in enumerate(label_scenario):
            if u3 != u:
                neg_list.append(features_banks[u3].memory[-topk:].view(topk, -1))
        if neg_list:
            neg = torch.cat(neg_list, dim=0).to(device)
            neg_norms = neg.norm(dim=1, keepdim=True)
            neg = neg / (neg_norms + eps)
            neg_banks[u] = neg

    CON_loss = torch.tensor(0., device=device)

    # 2. 保持原三重循环结构
    for idb in range(B):
        for u, flag in enumerate(label_scenario):
            if flag != 1:
                continue

            # 2.1 取 anchor 并手动归一化

            anchor = features[u][4][idb].view(1, -1).to(device)
            an_norm = anchor.norm(dim=1, keepdim=True)
            if an_norm.item() < eps:
                # 全 0 向量，跳过
                continue
            anchor = anchor / (an_norm + eps)

            # 2.2 计算正样本部分
            pos_bank = pos_banks[u]  # [topk, D]
            pos_sims = anchor @ pos_bank.T  # [1, topk]
            pos_exp = torch.exp(pos_sims / temperature).sum()

            # 2.3 计算负样本部分（chunk）
            neg = neg_banks.get(u, None)
            neg_exp_sum = torch.tensor(0., device=device)
            if neg is not None:
                for neg_chunk in neg.split(chunk_size, dim=0):
                    neg_sims = anchor @ neg_chunk.T
                    neg_exp_sum += torch.exp(neg_sims / temperature).sum()

            # 2.4 clamp 防止 pos_exp/denom 过小或过大
            denom = pos_exp + neg_exp_sum + eps
            ratio = pos_exp / denom
            ratio = torch.clamp(ratio, min=eps, max=1.0)
            loss = -torch.log(ratio)
            CON_loss += loss

    return CON_loss

# >>> Added for loss logging
def init_metrics_container():
    return {
        'epoch': [],
        'd_loss': [],
        'g_loss': [],
        'seg_loss': [],
        'dice_loss': [],
        'cross_entropy_loss': [],
        'L1_missing_loss': [],
        'L1_reconstruct_loss': [],
        'L1_local_loss': [],
        'feature_loss': [],
        'fea_loss_raw': [],
        'contrastive_loss': [],
        'cdl_loss': [],
        'val_psnr_avg': [],
        'val_ssim_avg': [],
        'best_psnr_avg': [],
        'best_ssim_avg': []
    }

def save_metrics_csv_json(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    json_path = os.path.join(save_dir, 'training_metrics.json')
    # CSV
    keys = list(metrics.keys())
    # 将 epoch 等长度对齐
    length = len(metrics['epoch'])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(length):
            row = [metrics[k][i] if i < len(metrics[k]) else '' for k in keys]
            writer.writerow(row)
    # JSON
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_metric_groups(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = metrics['epoch']
    if not epochs:
        return
    def _plot_single(y_keys, title, filename, ylabel='Loss'):
        plt.figure(figsize=(6,4))
        for k in y_keys:
            if len(metrics[k]) == len(epochs):
                plt.plot(epochs, metrics[k], label=k)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()

    _plot_single(['d_loss','g_loss'], 'D vs G Loss', 'd_g_loss.png')
    _plot_single(['seg_loss','dice_loss','cross_entropy_loss','cdl_loss'], 'Segmentation Losses', 'seg_losses.png')
    _plot_single(['L1_missing_loss','L1_reconstruct_loss','L1_local_loss'], 'L1 Losses', 'l1_losses.png')
    _plot_single(['feature_loss','contrastive_loss'], 'Feature / Contrastive ', 'feature_related.png')
    _plot_single(['val_psnr_avg','val_ssim_avg'], 'Validation Metrics', 'val_metrics.png', ylabel='Metric')

    # 组合到一张图（可选）
    plt.figure(figsize=(14,10))
    subplot_map = [
        (1, ['d_loss','g_loss'], 'D vs G'),
        (2, ['seg_loss','dice_loss','cross_entropy_loss','cdl_loss'], 'Seg Loss'),
        (3, ['L1_missing_loss','L1_reconstruct_loss','L1_local_loss'], 'L1 Loss'),
        (4, ['feature_loss','contrastive_loss'], 'Feature/Contrastive'),
        (5, ['val_psnr_avg','val_ssim_avg'], 'Val PSNR/SSIM')
    ]
    for idx,(sp, keys, title) in enumerate(subplot_map, start=1):
        plt.subplot(3,2,idx)
        for k in keys:
            if len(metrics[k]) == len(epochs):
                plt.plot(epochs, metrics[k], label=k)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.legend(fontsize=8)
    plt.savefig(os.path.join(save_dir,'all_metrics_grid.png'), dpi=180)
    plt.close()
def train(args):
    log_dir = os.path.join(args.checkpoint_dir, 'logs')
    sample_dir = os.path.join(args.checkpoint_dir, 'sample_dir')
    model_save_dir = os.path.join(args.checkpoint_dir, 'model_save_dir')
    result_dir = os.path.join(args.checkpoint_dir, 'result_dir')

    glr = args.lr
    print(glr, flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator_teacher()
    # netseg = Generator_teacher()
    # netseg.load_state_dict(checkpoint)
    netD = Discriminators()

    g_optimizier = torch.optim.Adam(netG.parameters(), lr=glr,
                                    betas=(args.betas[0], args.betas[1]))
    d_optimizier = torch.optim.Adam(netD.parameters(), lr=glr * 4, betas=(args.betas[0], args.betas[1]))

    netG.to(device)
    # netseg.to(device)
    netD.to(device)
    netG.to(device)
    # netseg.to(device)
    netD.to(device)
    features_banks.to(device)
    label_banks.to(device)
    # t_label_banks.to(device)

    if args.sepoch != 0:
        print('Resume the trained models from epoch {}...'.format(args.sepoch), flush=True)
        G_path = os.path.join(model_save_dir, 'Gseg_bank631.ckpt').replace('\\', '/')  # 解决斜杆问题
        # E_path = os.path.join(model_save_dir, 'E1028(1).ckpt').replace('\\', '/')  # 解决斜杆问题
        D_path = os.path.join(model_save_dir, 'Dseg_bank631.ckpt').replace('\\', '/')
        f_path = os.path.join(model_save_dir, 'fea_banks631.pth').replace('\\', '/')
        l_path = os.path.join(model_save_dir, 'label_banks631.pth').replace('\\', '/')
        # t_path = os.path.join(model_save_dir, 't_label_banks22.pth').replace('\\', '/')

        netG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        # net_E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
        netD.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

        g_optimizier.load_state_dict(torch.load(os.path.join(model_save_dir, 'optimizerG631.pth').replace('\\', '/'),
                                                map_location=lambda storage, loc: storage))
        d_optimizier.load_state_dict(torch.load(os.path.join(model_save_dir, 'optimizerD631.pth').replace('\\', '/'),
                                                map_location=lambda storage, loc: storage))

        features_banks.load_state_dict(torch.load(f_path, map_location=lambda storage, loc: storage))
        label_banks.load_state_dict(torch.load(l_path, map_location=lambda storage, loc: storage))
        # t_label_banks.load_state_dict(torch.load(t_path, map_location=lambda storage, loc: storage))
    print('start training...', flush=True)

    loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    mse = nn.MSELoss()

    # Data loader
    data_files = {
        'train': './partition/0-train.txt',
        'val': './partition/0-val.txt',
        'test': './partition/0-test.txt'
    }
    loaders = get_loaders(data_files, ['t1ce', 't1', 't2', 'flair'], args.batch_size, 4)
    target_path = os.path.join(sample_dir, '{}-{}-fake.png')
    real_path = os.path.join(sample_dir, '{}-{}-real.png')

    scenarios = list(map(list, itertools.product([0, 1], repeat=4)))
    scenarios.remove([0, 0, 0, 0])
    scenarios.remove([1, 1, 1, 1])
    scenarios.sort(key=lambda x: x.count(1))
    label_list = torch.from_numpy(np.zeros((args.batch_size,
                                            1,
                                            12,
                                            12))).cuda().type(torch.cuda.FloatTensor)
    label_list_r = torch.from_numpy(np.ones((args.batch_size,
                                             1,
                                             12,
                                             12))).cuda().type(torch.cuda.FloatTensor)

    SSIM_avg = 0.0
    PSNR_avg = 0.0
    best_epoch = 0
    WT = 0.0
    ET = 0.0
    TC = 0.0
    class_maping = torch.tensor([0, 1, 2, 4]).cuda()
    # >>> Added metrics container
    metrics = init_metrics_container()

    for epoch in range(args.sepoch, args.epoch):

        d_loss_total = 0.0
        g_loss_total = 0.0
        seg_loss_total = 0.0
        dice_loss_total = 0.0
        cross_entropy_total = 0.0
        L1_missing_total = 0.0
        L1_reconstruct_total = 0.0
        L1_local_total = 0.0
        feature_loss_total = 0.0   # fea_loss/(4-counts)
        contrastive_total = 0.0
        cdl_loss_total = 0.0

        steps = len(loaders['train'])
        start_time = datetime.datetime.now()
        netG.train()
        for i, batch_data in enumerate(loaders['train']):

            t1ce, t1, t2, flair, label, t_img, mask, four_masks, B_label,_ = batch_data
            flair = flair.unsqueeze(1).to(device)
            t1 = t1.unsqueeze(1).to(device)
            t1ce = t1ce.unsqueeze(1).to(device)
            t2 = t2.unsqueeze(1).to(device)
            label = label.unsqueeze(1).to(device)

            mask = mask.unsqueeze(1).to(device)

            B_label = [b_label.to(device) for b_label in B_label]

            t_imgs = [x.unsqueeze(1).to(device) for x in t_img]

            four_masks = [y.unsqueeze(1).to(device) for y in four_masks]

            small_mask = [F.interpolate(mask, size=(12, 12), mode='nearest') for mask in four_masks]

            small_mask = torch.cat(small_mask, dim=1)

            B = flair.size(0)
            impute_tensor = torch.full((B, 1, 192, 192),
                                       fill_value=-1.0,
                                       dtype=torch.float32,
                                       device=device)



            if epoch <= 10:
                curr_scenario_range = [11, 14]
                rand_val = torch.randint(low=10, high=14, size=(1,))
            if epoch > 10 and epoch <= 20:
                curr_scenario_range = [7, 14]
                rand_val = torch.randint(low=7, high=14, size=(1,))
            if epoch > 20 and epoch <= 30:
                curr_scenario_range = [3, 14]
                rand_val = torch.randint(low=3, high=14, size=(1,))
            if epoch > 30:
                curr_scenario_range = [0, 14]
                rand_val = torch.randint(low=0, high=14, size=(1,))
            # curr_scenario_range = [11, 14]
            # rand_val = torch.randint(low=10, high=14, size=(1,))

            label_scenario = scenarios[int(rand_val.numpy()[0])]

            # 't1ce', 't1', 't2', 'flair'
            x_real = [flair, t1, t1ce, t2]
            imgs = [flair, t1, t1ce, t2]
            t_imgs2 = [t_imgs[3], t_imgs[1], t_imgs[0], t_imgs[2]]
            t_small_masks = [B_label[3], B_label[1], B_label[0], B_label[2]]

            count = 0
            for idx, k in enumerate(label_scenario):
                # label_list[:, idx] = 0
                if k == 0:
                    imgs[idx] = impute_tensor
                    count = count + 1

            # 2. Train the discriminator
            # Compute loss with real whole images.
            # x_real_cat = torch.cat([x_real[0], x_real[1], x_real[2], x_real[3]], dim=1)
            out_src = netD(x_real)

            # d_loss_real =mse(out_src, label_list_r)#-torch.mean(out_src, dim=[0,2,3])#loss#

            with torch.no_grad():
                pre_label_map, _ = netG(x_real[0], x_real[1], x_real[2], x_real[3], None, None, small_mask,
                                        features_banks, label_banks, miss_index=label_scenario, epoch=epoch, seg=1,
                                        up=0)
                pre_label_map = torch.softmax(pre_label_map, dim=1)
                pre_class = torch.argmax(pre_label_map, dim=1)
                pre_label = class_maping[pre_class]
                pre_label = label_norm(pre_label).unsqueeze(1)
                # print("1,",pre_label.size())
                img_fake0, _, _ = netG(x_real[0], x_real[1], x_real[2], x_real[3], label, pre_label, small_mask,
                                       features_banks, label_banks, miss_index=label_scenario, epoch=epoch, seg=0)

            out_src1 = netD(img_fake0)

            d_loss = 0.0
            for d, k in enumerate(label_scenario):
                if k == 0:
                    real_loss = mse(out_src[d], label_list_r)
                    fake_loss = mse(out_src1[d], label_list)
                    d_loss += (real_loss + fake_loss)

            d_loss_total += d_loss.item()
            d_optimizier.zero_grad()
            d_loss.backward()
            d_optimizier.step()

            # 3. Train the generator
            # train segmentor
            seg_out, cdl_loss = netG(x_real[0], x_real[1], x_real[2], x_real[3], None, None, small_mask, features_banks,
                                     label_banks, miss_index=label_scenario, epoch=epoch, seg=1)

            di_loss = diceloss(torch.softmax(seg_out, dim=1), four_masks)
            cr_loss = crosloss(seg_out, denorm(label))
            seg_loss = 5*di_loss + cr_loss + 2*cdl_loss
            g_optimizier.zero_grad()
            seg_loss.backward()
            g_optimizier.step()

            with torch.no_grad():
                pre_label_map2, _ = netG(x_real[0], x_real[1], x_real[2], x_real[3], None, None, small_mask,
                                         features_banks, label_banks, miss_index=label_scenario, epoch=epoch, seg=1)
                pre_label_map2 = torch.softmax(pre_label_map2, dim=1)
                pre_class2 = torch.argmax(pre_label_map2, dim=1)
                pre_label2 = class_maping[pre_class2]
                pre_label2 = label_norm(pre_label2).unsqueeze(1)
            img_fake1, features, fea_loss = netG(x_real[0], x_real[1], x_real[2], x_real[3], label, pre_label2,
                                                 small_mask,
                                                 features_banks, label_banks, miss_index=label_scenario, epoch=epoch,
                                                 seg=0, up=1)

            # contrastive_loss_pull_only_vicreg(features, features_banks, label_scenario, topk,
            #                           eps=1e-6,
            #                           inv_w=1.0, var_w=1.0, cov_w=1.0,
            #                           gamma=1.0, chunk_layer_idx=4)
            CON_loss = contrastive_loss_with_clamp(
                features, features_banks, label_scenario,
                topk=memory_bank_sizes,
                temperature=0.1, eps=1e-6, chunk_size=1024
            )

            d_g_loss = 0.0
            out_src2 = netD(img_fake1)
            for d, k in enumerate(label_scenario):
                if k == 0:
                    fake_loss = mse(out_src2[d], label_list)
                    d_g_loss += fake_loss

            L1_loss = 0.0
            L1_loss2 = 0.0
            L1_rec = 0.0

            for idx_curr_label, m in enumerate(label_scenario):
                if m == 0:
                    fakel = img_fake1[idx_curr_label]
                    lossl = l1_loss(fakel, x_real[idx_curr_label])
                    L1_loss += lossl

                    lossl2 = l1_loss(fakel * mask, t_imgs2[idx_curr_label])
                    L1_loss += lossl
                    L1_loss2 += lossl2


                else:
                    fakel2 = img_fake1[idx_curr_label]
                    lossl_2 = l1_loss(fakel2, x_real[idx_curr_label])
                    L1_rec += lossl_2

            counts = sum(label_scenario)
            g_loss = d_g_loss + 100 * L1_loss / (4 - counts) + 30 * L1_rec / counts + 10 * L1_loss2 / (
                    4 - counts) + 10 * fea_loss / (4 - counts) + CON_loss / counts
            # d_g_loss + 100 * L1_loss + 30 * L1_rec + 10 * L1_loss2 + fea_loss

            g_loss_total += g_loss.item()
            g_optimizier.zero_grad()
            g_loss.backward()
            g_optimizier.step()

            # 更新bank
            label_banks.initialize_memory_bank(small_mask)

            for j2 in range(4):
                features_banks[j2].initialize_memory_bank(features[j2][4])
                # t_label_banks[j2].initialize_memory_bank(t_small_masks[j2])

                # >>> accumulate other losses

            seg_loss_total += seg_loss.item()
            dice_loss_total += di_loss.item()
            cross_entropy_total += cr_loss.item()
            L1_missing_total += L1_loss.item()/(4 - counts)
            L1_reconstruct_total += L1_rec.item()/sum(label_scenario)
            L1_local_total += L1_loss2.item()/(4 - counts)
            feature_loss_total += (fea_loss / (4 - counts)).item()
            contrastive_total += (CON_loss / counts).item()
            cdl_loss_total += (cdl_loss.item() if isinstance(cdl_loss, torch.Tensor) else cdl_loss)

            if i % 50 == 0:
                torch.cuda.empty_cache()

            if i % 100 == 0:
                if isinstance(cdl_loss, torch.Tensor):
                    cdl_loss = cdl_loss.item()
                if isinstance(fea_loss, torch.Tensor):
                    fea_loss = fea_loss.item()
                print(
                    f'Epoch: [{epoch + 1}], Step: [{i}], D-Loss: {d_loss.item():.4f}, G-Loss: {g_loss.item():.4f}, L1-Loss: {L1_loss.item():.4f},'
                    f'L1-loss_rec:{L1_rec.item():.4f},local-Loss: {L1_loss2.item():.8f},cdl-Loss: {cdl_loss:.8f},'  # emb-Loss: {emb_loss.item():.8f},
                    f'seg-Loss: {seg_loss.item():.8f},fea-Loss: {fea_loss / (4 - counts):.8f},contra-Loss: {CON_loss.item() / counts:.8f}')  # cf-Loss: {feature_loss:.8f}cf-Loss: {cf_loss2:.8f}
        end_time = datetime.datetime.now()
        print(
            f'Epoch [{epoch + 1}/{args.epoch}] - D-Loss: {d_loss_total / len(loaders["train"]):.4f}, G-Loss: {g_loss_total / len(loaders["train"]):.4f}, Time: {end_time - start_time}')
        print('----------------------------------Epoch: [%2d] finsih' % (epoch + 1))
        if (epoch + 1) % 1 == 0:
            netG.eval()
            with torch.no_grad():
                s = [0, 0, 0, 0]
                t = [0, 0, 0, 0]
                for v, batch_data in enumerate(loaders['val']):
                    t1ce, t1, t2, flair, label, t_img, mask, four_masks2, B_label,_ = batch_data
                    flair = flair.unsqueeze(1).to(device)
                    t1ce = t1ce.unsqueeze(1).to(device)
                    t1 = t1.unsqueeze(1).to(device)
                    t2 = t2.unsqueeze(1).to(device)
                    label = label.unsqueeze(1).to(device)
                    # B_label = B_label.unsqueeze(0).to(device)
                    B_label = [b_label.to(device) for b_label in B_label]
                    four_masks2 = [y.unsqueeze(1).to(device) for y in four_masks2]
                    small_mask2 = [F.interpolate(mask, size=(12, 12), mode='nearest') for mask in
                                   four_masks2]
                    small_mask2 = torch.cat(small_mask2, dim=1)
                    t_small_mask2 = [B_label[3], B_label[1], B_label[0], B_label[2]]
                    x_real = [flair, t1, t1ce, t2]
                    B = flair.size(0)
                    impute_tensor = torch.full((B, 1, 192, 192),
                                               fill_value=-1.0,
                                               dtype=torch.float32,
                                               device=device)
                    # _, realf = net_E(x_real[0], x_real[1], x_real[2], x_real[3], label, miss_index=misss)

                    for j in range(0, 4):
                        imgs = [flair, t1, t1ce, t2]
                        misss = [1, 1, 1, 1]
                        misss[j] = 0
                        imgs[j] = impute_tensor

                        pre_label3_map, _ = netG(imgs[0], imgs[1], imgs[2], imgs[3], label, None, small_mask2,
                                                 features_banks, label_banks, miss_index=misss, epoch=100, seg=1)
                        pre_label3_map = torch.softmax(pre_label3_map, dim=1)
                        pre_class3 = torch.argmax(pre_label3_map, dim=1)
                        pre_label3 = class_maping[pre_class3]
                        pre_label3 = label_norm(pre_label3).unsqueeze(1)
                        img_fake0, _, _ = netG(imgs[0], imgs[1], imgs[2], imgs[3], label, pre_label3, small_mask2,
                                               features_banks, label_banks, miss_index=misss, epoch=epoch, seg=0)

                        x_real[j] = denorm(x_real[j])
                        img_fake0[j] = denorm(img_fake0[j])
                        for b in range(B):
                            target_img = img_fake0[j][b].squeeze().cpu().numpy()
                            x_real_img = x_real[j][b].squeeze().cpu().numpy()
                            # 计算psnr
                            # print('{}-{}-psnr={}'.format(i,j, psnr(target_img, x_real[j], data_range=1)))
                            # print('{}-{}-ssim={}'.format(i,j, ssim(target_img, x_real[j], data_range=1))) #, multichannel=True
                            s[j] = s[j] + psnr(target_img, x_real_img, data_range=1)
                            t[j] = t[j] + ssim(target_img, x_real_img, data_range=1)

                mod = ['flair', 't1', 't1ce', 't2']
                SS = 0
                PS = 0
                for r in range(4):
                    SS += s[r] / 400
                    PS += t[r] / 400
                    # print('{}:psnr-avg={}'.format(mod[r], s[r] / 200))
                    # print('{}:ssim-avg={}'.format(mod[r], t[r] / 200))
                # print(SS/4)
                # print(PS/4)
                if (PS / 4 > PSNR_avg):
                    SSIM_avg = SS / 4
                    PSNR_avg = PS / 4
                    best_epoch = epoch + 1
                    G_path = os.path.join(model_save_dir, 'Gseg_bank915.ckpt')  # 0111

                    D_path = os.path.join(model_save_dir, 'Dseg_bank915.ckpt')
                    torch.save(netG.state_dict(), G_path)
                    torch.save(netD.state_dict(), D_path)

                    # 保存优化器参�?
                    torch.save(g_optimizier.state_dict(), os.path.join(model_save_dir, 'optimizerG915.pth'))
                    torch.save(d_optimizier.state_dict(), os.path.join(model_save_dir, 'optimizerD915.pth'))

                    f_path = os.path.join(model_save_dir, 'fea_banks915.pth')  # banks3
                    torch.save(features_banks.state_dict(), f_path)
                    l_path = os.path.join(model_save_dir, 'label_banks915.pth')
                    torch.save(label_banks.state_dict(), l_path)
                    # t_path = os.path.join(model_save_dir, 't_label_banks25.pth')
                    # torch.save(t_label_banks.state_dict(), t_path)

                print('psnr-avg={}'.format(SS / 4))
                print('ssim-avg={}'.format(PS / 4))
                print('best_epoch={}'.format(best_epoch))

                # >>> Epoch 级别记录
                metrics['epoch'].append(epoch + 1)
                metrics['d_loss'].append(d_loss_total / steps)
                metrics['g_loss'].append(g_loss_total / steps)
                metrics['seg_loss'].append(seg_loss_total / steps)
                metrics['dice_loss'].append(dice_loss_total / steps)
                metrics['cross_entropy_loss'].append(cross_entropy_total / steps)
                metrics['L1_missing_loss'].append(L1_missing_total / steps)
                metrics['L1_reconstruct_loss'].append(L1_reconstruct_total / steps)
                metrics['L1_local_loss'].append(L1_local_total / steps)
                metrics['feature_loss'].append(feature_loss_total / steps)
                metrics['contrastive_loss'].append(contrastive_total / steps)
                metrics['cdl_loss'].append(cdl_loss_total / steps)
                metrics['val_psnr_avg'].append(SSIM_avg)
                metrics['val_ssim_avg'].append(PSNR_avg)

                # 每个 epoch 即时存储 & 画图
                save_metrics_csv_json(metrics, log_dir)
                plot_metric_groups(metrics, log_dir)

        if(epoch + 1) % 1 == 0:
            missing_combinations = [
                [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1],
                [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 0, 1]
            ]
            com_wt=0.0
            com_et=0.0
            com_tc=0.0
            with torch.no_grad():
                for combo in missing_combinations:
                    s = [0, 0, 0, 0]
                    t = [0, 0, 0, 0]
                    dice = [0, 0, 0]
                    a = 0
                    b = 0
                    c = 0
                    for i, batch_data in enumerate(loaders['test']):

                        t1ce, t1, t2, flair, label, t_img, mask, four_masks, B_label, truelabel = batch_data
                        flair = flair.unsqueeze(1).to(device)
                        t1 = t1.unsqueeze(1).to(device)
                        t1ce = t1ce.unsqueeze(1).to(device)
                        t2 = t2.unsqueeze(1).to(device)
                        label = label.unsqueeze(1).to(device)
                        truelabel.unsqueeze(1).to(device)
                        four_masks = [y.to(device) for y in four_masks]

                        B_label = [b_label.to(device) for b_label in B_label]

                        small_mask = [F.interpolate(mask.unsqueeze(1), size=(12, 12), mode='nearest') for mask in
                                      four_masks]
                        small_mask = torch.cat(small_mask, dim=1)

                        x_real = [flair, t1, t1ce, t2]

                        with torch.no_grad():
                            pre_label_map2, _ = netG(x_real[0], x_real[1], x_real[2], x_real[3], None, None, small_mask,
                                                     features_banks, label_banks, miss_index=combo, epoch=200, seg=1)
                        output = pre_label_map2

                        probs = F.softmax(output, dim=1)  # (N,4,H,W)
                        preds = torch.argmax(probs, dim=1)  # (N,H,W)，值在 {0,1,2,3}

                        # ———— 2. 把 preds 中的 “3” 映射回原始标签 4 ————
                        # 创建 preds_orig（仍保留在 GPU 上）
                        preds_orig = preds.clone().to(device)  # (N,H,W)，当前值在 {0,1,2,3}
                        preds_orig[preds_orig == 3] = 4
                        truelabel[truelabel == 3] = 4

                        # WT_true = (truelabel > 0).long().to(device)      # (N,H,W)，1 表示是 WT 区域
                        WT_true = four_masks[1] + four_masks[2] + four_masks[3]
                        WT_pred = (preds_orig > 0).long()  # (N,H,W)

                        # # ET: enhancing tumor → 标签 4
                        ET_true = four_masks[3]  # (N,H,W)
                        ET_pred = (preds_orig == 4).long()  # (N,H,W)

                        # # TC: tumor core → 标签 1 或 4
                        # # 注意：不包括原始标签 2
                        TC_true = four_masks[1] + four_masks[3]  # (N,H,W)
                        TC_pred = ((preds_orig == 1) | (preds_orig == 4)).long()

                        score1 = binary_dice_score(WT_pred, WT_true)
                        score2 = binary_dice_score(ET_pred, ET_true)
                        score3 = binary_dice_score(TC_pred, TC_true)
                        if WT_true.sum() != 0 and score1 >= 0.01:
                            a += 1
                            dice[0] += score1
                        if ET_true.sum() != 0 and score2 >= 0.01:
                            b += 1
                            dice[1] += score2
                        if TC_true.sum() != 0 and score3 >= 0.01:
                            c += 1
                            dice[2] += score3

                        # print('True ',i,':', WT_true.sum(),ET_true.sum(), TC_true.sum())

                    out1 = dice[0] / a
                    if b!=0:
                        out2 = dice[1] / b
                    else:
                        out2 = 0
                    out3 = dice[2] / c

                    com_wt = com_wt + out1
                    com_et = com_et + out2
                    com_tc = com_tc + out3


                    # print(out1.item())
                    # print(out2.item())
                    # print(out3.item())
                if (com_wt/14 > WT and com_et/14 > ET and com_tc/14 > TC):
                    WT = com_wt/14
                    ET = com_et/14
                    TC = com_tc/14
                    torch.save(netG.state_dict(), os.path.join(model_save_dir, 'Gseg_best.ckpt'))
                    print('best_seg_epoch={}'.format(epoch + 1))
                print('WT:', com_wt/14, 'ET:', com_et/14, 'TC:', com_tc/14)

        if (epoch + 1) == args.epoch:
            print('best_epoch={}'.format(best_epoch))
            print('best_psnr-avg={}'.format(PSNR_avg))
            print('best_ssim-avg={}'.format(SSIM_avg))
            print('best_WT={}'.format(WT))
            print('best_ET={}'.format(ET))
            print('best_TC={}'.format(TC))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datasets', type=str, default='BraTs')
    parser.add_argument('-save_path', type=str, default='checkpoint')
    # parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-gan_version', type=str, default='Generator[2/3]+shapeunet+D')
    parser.add_argument('-epoch', type=int, default=900)
    parser.add_argument('-sepoch', type=int, default=0)
    parser.add_argument('-lr', type=float, default=2e-4)

    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-note', type=str, default='affine:True;')
    parser.add_argument('-random_seed', type=int, default='1234')

    parser.add_argument('-betas', type=tuple, default=(0.5, 0.999))

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')

    # Data Loader.
    parser.add_argument('--phase', type=str, default='train')

    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser = parser.parse_args()
    print(parser, flush=True)
    train(parser)


