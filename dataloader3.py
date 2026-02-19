import cv2
from torch.utils import data
from torchvision import transforms as T
import os
import numpy as np
import SimpleITK as sitk
from tools import tsfm_tfusion
import torch
from scipy.ndimage import zoom
import torch.nn.functional as F
import skimage
from skimage.feature import graycoprops, graycomatrix, local_binary_pattern
from skimage.filters import gabor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class Brain(data.Dataset):
    def __init__(self, data_file, selected_modal, brain_dir, inputs_transform=None,
                 labels_transform=None, t_join_transform=None, join_transform=None, phase='train'):
        self.selected_modal = selected_modal  # 选定模态
        self.c_dim = len(self.selected_modal)  # 长度
        self.inputs_transform = inputs_transform  # 输入变换
        self.labels_transform = labels_transform  # 标签变换
        self.join_transform = join_transform
        self.t_join_transform = t_join_transform
        self.data_file = data_file  # 数据文件
        self.dataset = {}
        self.phase = phase
        self.brain_dir = brain_dir
        self.init()

    def init(self):

        self.dataset['data'] = []
        lines = [line.rstrip() for line in open(self.data_file, 'r')]  # 读取指定文件的每一行并去除行尾的空白符 存储在lines
        for i, image_path in enumerate(lines):
            image_name = os.path.basename(image_path)
            pid = image_name.split('_')[-1]

            self.dataset['data'].append([image_path, pid])

        print('[*] Load {}, which contains {} paired volumes with radom missing modilities, {}'.format(self.data_file,
                                                                                                       len(self.dataset[
                                                                                                               'data']),
                                                                                                       self.selected_modal))

    def __getitem__(self, idex):
        patient_idx = idex // 10  # 得到当前患者的索引
        slice_idx = idex % 10 + 70  # 得到当前切片的索引，从70开始到80
        image_path, pid = self.dataset['data'][patient_idx]

        label_path = image_path + '/BraTS20_Training_{}_{}.nii.gz'.format(pid, 'seg')
        volume_label = (sitk.GetArrayFromImage(sitk.ReadImage(label_path))[slice_idx]).astype(
            np.float32)  # 读取并转换标签图像的数据
        volume_label = volume_label[30:222, 20:212]

        masks = [(volume_label == i).astype(np.float32) for i in [0, 1, 2, 4]]
        four_mask = [torch.from_numpy(mask) for mask in masks]

        # zoom_factor = 256 / 192
        # volume_label = zoom(volume_label, zoom_factor, order=0)
        # volume_label = pad_image_to_size(volume_label, 256, 256)

        volume_label2 = volume_label
        volume_label2[volume_label2 == 4] = 3

        maskT = (volume_label > 0).astype(np.float32)

        volumes = []
        t_imgs = []

        for modal in self.selected_modal:
            m_path = image_path + '/BraTS20_Training_{}_{}.nii.gz'.format(pid, modal)
            volume = (sitk.GetArrayFromImage(sitk.ReadImage(m_path))[slice_idx]).astype(np.float32)
            volume = volume[30:222, 20:212]
            # volume = zoom(volume, zoom_factor, order=2)
            volumes.append(volume)
            # t_imgs.append(volume*mask)

        # if self.join_transform:
        #     volumes, volume_label, crop_size = self.join_transform(volumes, volume_label, self.phase)
        if self.t_join_transform:
            volumes, volume_label, _ = self.t_join_transform(volumes, volume_label, self.phase)

        if self.inputs_transform:
            volumes = [self.inputs_transform(vol) for vol in volumes]

        t_imgs = [vol * maskT for vol in volumes]

        if self.labels_transform:
            volume_label = self.labels_transform(volume_label)
        #     t_imgs[0] = self.labels_transform(t_imgs[0])
        #     t_imgs[1] = self.labels_transform(t_imgs[1])
        #     t_imgs[2] = self.labels_transform(t_imgs[2])
        #     t_imgs[3] = self.labels_transform(t_imgs[3])

        bins = np.arange(-1.0, 1.1, 0.2)
        B_masks = []
        for j in range(4):
            masks = []
            # 背景 mask（仅包括 -1）
            small_vol = F.avg_pool2d(volumes[j].unsqueeze(0), kernel_size=2, stride=2)  # 第一次池化，变为96x96
            small_vol = F.avg_pool2d(small_vol, kernel_size=2, stride=2)  # 第二次池化，变为48x48
            small_vol = F.avg_pool2d(small_vol, kernel_size=2, stride=2)  # 第三次池化，变为24x24
            small_vol = F.avg_pool2d(small_vol, kernel_size=2, stride=2)  # 第四次池化，变为12x12

            # background_mask = (small_vol == -1).float()  # 确保是 (1, 192, 192)
            #
            #
            # # 如果是从一个多通道的输入开始，可以调整`unsqueeze`和`squeeze`的维度
            # background_mask = background_mask.squeeze(0)  # 如果需要去掉batch维度
            # masks.append(background_mask)
            # # 逐步遍历每个区间，生成 0-1 mask
            # for i in range(len(bins) - 1):
            #     mask = ( (small_vol != -1)&(small_vol >= bins[i]) & (small_vol < bins[i + 1])).float()  # 变为 float32 Tensor
            #     # mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(12, 12), mode='nearest').squeeze(0)  # 保持形状一致
            #     # mask = F.avg_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)  # 第一次池化，变为96x96
            #     # mask = F.avg_pool2d(mask, kernel_size=2, stride=2)  # 第二次池化，变为48x48
            #     # mask = F.avg_pool2d(mask, kernel_size=2, stride=2)  # 第三次池化，变为24x24
            #     # mask = F.avg_pool2d(mask, kernel_size=2, stride=2)  # 第四次池化，变为12x12
            #     mask = mask.squeeze(0)  # 如果需要去掉batch维度
            #     masks.append(mask)
            # # 拼接所有 mask（在第 0 维度）
            # result =torch.cat(masks, dim=0)
            B_masks.append(small_vol)

        return volumes[0], volumes[1], volumes[2], volumes[
            3], volume_label, t_imgs, maskT, four_mask, B_masks, torch.tensor(volume_label2)

    def __len__(self):
        return len(self.dataset['data']) * 10  # 取多少切片 *多少


def get_loaders(data_files, selected_modals, batch_size=1, num_workers=0):
    rs = np.random.RandomState(1234)

    train_join_tsfm = tsfm_tfusion.Compose([
        tsfm_tfusion.RandomFlip(rs),  # 随机翻转数据体积
        tsfm_tfusion.RandomRotate(rs, angle_spectrum=10),  # 角度+-10°
    ])
    input_tsfm = T.Compose([
        tsfm_tfusion.Normalize(),
        tsfm_tfusion.NpToTensor()
    ])
    label_tsfm = T.Compose([
        tsfm_tfusion.Normalize2(),  # 新加
        tsfm_tfusion.NpToTensor()
    ])

    brain_dir = '/root/lanyun-fs/MICCAI_BraTS2020_TrainingData'

    datasets = dict(train=Brain(data_files['train'], selected_modals, brain_dir, inputs_transform=input_tsfm,
                                labels_transform=label_tsfm, t_join_transform=train_join_tsfm, join_transform=None,
                                phase='train'),  # train_join_tsfm
                    val=Brain(data_files['val'], selected_modals, brain_dir, inputs_transform=input_tsfm,
                              labels_transform=label_tsfm, t_join_transform=None, join_transform=None, phase='val'),
                    test=Brain(data_files['test'], selected_modals, brain_dir, inputs_transform=input_tsfm,
                               labels_transform=label_tsfm, t_join_transform=None, join_transform=None, phase='test')
                    )
    loaders = {x: data.DataLoader(dataset=datasets[x], batch_size=batch_size,
                                  shuffle=(x == 'train'),
                                  num_workers=num_workers)
               for x in ('train', 'val', 'test')}
    return loaders

#