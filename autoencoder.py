import json
import shutil
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda import cudart
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime
from tqdm import tqdm
from os import makedirs

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.loss_utils import mse_loss, l1_loss

from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.save_and_load import *


class Autoencoder(nn.Module):
    def __init__(self, feat_dim=56, hidden=32):
        super(Autoencoder, self).__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Linear(self.feat_dim, 512),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(512, self.hidden)  # 隐藏层到潜在空间
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 512),  # 潜在空间到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(512, self.feat_dim)  # 隐藏层到输出层
        )

    def forward(self, x):
        z = self.encoder(x)  # 编码器部分：输入到潜在空间的映射
        reconstructed = self.decoder(z)  # 解码器部分：潜在空间到输出的映射
        return reconstructed, z


def train_model(model, gs: GaussianModel, gs_fea, first, epochs, path, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, epochs[-1]), desc="Training progress")
    model.train()
    for epoch in range(first, epochs[-1] + 1):
        if epoch in epochs:
            reconstructed, latent_space = model(gs_fea)
            # 保存AE模型
            model_path = os.path.join(path, "model", "autoencoder_" + str(epoch) + ".pth")
            save_model(model, model_path)
            # 保存压缩存储的点云
            compress_ply_path = os.path.join(path, "compress", "compress_" + str(epoch) + ".ply")
            save_compress_latent_ply(latent_space, gs, compress_ply_path)
            # 从压缩点云中重建高斯
            reconstructed_gs = load_compress_latent_ply(model, compress_ply_path)
            # 保存重建高斯到点云文件
            reconstructed_path = os.path.join(path, "reconstructed",
                                              "point_cloud_" + str(epoch) + ".ply")
            reconstructed_gs.save_ply(reconstructed_path)
            # 训练结束时，保存一个命名好的点云，方便使用
            if epoch == epochs[-1]:
                reconstructed_gs.save_ply(os.path.join(path, "point_cloud", "iteration_30000", "point_cloud.ply"))

        total_loss = 0
        optimizer.zero_grad()  # 清除梯度
        reconstructed, _ = model(gs_fea)
        # loss = l1_loss(reconstructed,gs_fea)
        loss = mse_loss(reconstructed, gs_fea)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if epoch % 10 == 0:
            progress_bar.set_postfix(
                {"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if epoch == epochs:
            progress_bar.close()


def get_gs_fea(gaussians: GaussianModel):
    g_xyz = gaussians.get_xyz.detach()  # 获取3D点坐标数据并从计算图中分离
    n_gaussian = g_xyz.shape[0]  # 获取点云数据中点的数量

    # 每步处理的点的数量
    per_step_size = 100_0000  # 默认每步处理100万点
    if 100_0000 < n_gaussian < 110_0000:  # 如果点数在100万到110万之间，调整每步处理的点数
        per_step_size = 110_0000
    # 计算需要的训练步数
    step_num = int(np.ceil(n_gaussian / per_step_size))  # 总步数，向上取整

    _features_dc = gaussians._features_dc.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 3]
    _features_rest = gaussians._features_rest.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 45]
    _opacity = gaussians._opacity.detach()  # 从计算图中分离透明度数据 [N, 1]
    _scaling = gaussians._scaling.detach()  # 从计算图中分离缩放数据 [N, 3]
    _rotation = gaussians._rotation.detach()  # 从计算图中分离旋转数据 [N, 4]
    gs_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation],
                       dim=-1)  # 将各类特征拼接成一个大的特征向量 [N, 56]

    return step_num, gs_fea


# # 预处理高斯模型，用于后续输入训练
# def handle_train(gaussians: GaussianModel, first, epochs, path):
#     _, gs_fea = get_gs_fea(gaussians)
#     model = Autoencoder().cuda()
#     train_model(model, gaussians, gs_fea=gs_fea, first = first, epochs=epochs, path=path, learning_rate=1e-3)

def handle_train(gaussians: GaussianModel, first, epochs, path):
    # 创建目标文件夹ae_output，如果不存在的话
    output_dir = os.path.join('ae_output', path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 递归复制path目录下的所有文件和子目录到ae_output文件夹
    def recursive_copy(src, dst):
        # 检查源目录是否存在
        if not os.path.exists(src):
            print(f"Source path {src} does not exist.")
            return
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)
            # 如果是文件，则直接复制
            if os.path.isfile(src_item):
                shutil.copy(src_item, dst_item)
            # 如果是目录，则递归调用
            elif os.path.isdir(src_item):
                if not os.path.exists(dst_item):
                    os.makedirs(dst_item)  # 创建目标子目录
                recursive_copy(src_item, dst_item)  # 递归复制

    # 调用递归函数复制文件和子目录
    recursive_copy(path, output_dir)

    # 获取高斯特征
    _, gs_fea = get_gs_fea(gaussians)
    # 初始化模型
    model = Autoencoder().cuda()
    # 训练模型
    train_model(model, gaussians, gs_fea=gs_fea, first=first, epochs=epochs, path=output_dir, learning_rate=1e-3)


def get_gs_model(path):
    full_path = os.path.join(path, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
    gs = GaussianModel(sh_degree=3)
    gs.load_ply(full_path)
    return gs


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--first_iteration", nargs="+", type=int, default=0)
    parser.add_argument("--train_iteration", nargs="+", type=int, default=[1000])
    parser.add_argument('--hidden', type=int, default=32)
    args = parser.parse_args(sys.argv[1:])

    gaussians = get_gs_model(args.model_path)

    # gaussians.save_ply("ae/model/source_gs.ply")
    handle_train(gaussians, args.first_iteration, args.train_iteration, args.model_path)

    print("\nTraining complete.")
