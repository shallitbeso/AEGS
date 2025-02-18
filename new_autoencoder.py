import json
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
    def __init__(self, color_dim = 49, color_hidden = 32, position_dim = 7, position_hidden = 5):
        super(Autoencoder, self).__init__()
        self.color_dim = color_dim
        self.color_hidden = color_hidden
        self.position_dim = position_dim
        self.position_hidden = position_hidden

        self.color_encoder = nn.Sequential(
            nn.Linear(self.color_dim, 512),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(512, self.color_hidden)  # 隐藏层到潜在空间
        )
        self.color_decoder = nn.Sequential(
            nn.Linear(self.color_hidden, 512),  # 潜在空间到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(512, self.color_dim)  # 隐藏层到输出层
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, 128),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(128, self.position_hidden)  # 隐藏层到潜在空间
        )
        self.position_decoder = nn.Sequential(
            nn.Linear(self.position_hidden, 128),  # 潜在空间到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(128, self.position_dim)  # 隐藏层到输出层
        )

    def color_forward(self, x):
        z = self.color_encoder(x)  # 编码器部分：输入到潜在空间的映射
        reconstructed = self.color_decoder(z)  # 解码器部分：潜在空间到输出的映射
        return reconstructed, z

    def position_forward(self, x):
        z = self.position_encoder(x)  # 编码器部分：输入到潜在空间的映射
        reconstructed = self.position_decoder(z)  # 解码器部分：潜在空间到输出的映射
        return reconstructed, z


def train_model(model, gs: GaussianModel, gs_color, gs_position, first, epochs, path, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, epochs[-1]), desc="Training progress")
    model.train()

    for epoch in range(first, epochs[-1] + 1):
        if epoch in epochs:
            reconstructed_color, latent_color = model.color_forward(gs_color)
            reconstructed_position, latent_position = model.position_forward(gs_position)
            reconstructed = torch.cat([reconstructed_color, reconstructed_position], dim=-1)
            latent_space = torch.cat([latent_color, latent_position], dim=-1)
            # 保存AE模型
            model_path = os.path.join(path, "model", "autoencoder_" + str(epoch) + ".pth")
            save_model(model, model_path)
            # 保存压缩存储的点云
            compress_ply_path = os.path.join(path, "compress", "compress_" + str(epoch) + ".ply")
            save_compress_latent_ply(latent_space, gs, compress_ply_path)

            set_gs_with_reconstructed(reconstructed, gs)
            gs.save_ply("output/tandt/train/new_ae_gs.ply")

            # # 从压缩点云中重建高斯
            # reconstructed_gs = load_compress_latent_ply(model, compress_ply_path)
            # # 保存重建高斯到点云文件
            # reconstructed_path = os.path.join(path, "reconstructed",
            #                                                       "point_cloud_" + str(epoch) + ".ply")
            # reconstructed_gs.save_ply(reconstructed_path)
            # # 训练结束时，保存一个命名好的点云，方便使用
            # if epoch == epochs[-1]:
            #     reconstructed_gs.save_ply(os.path.join(path, "reconstructed", "point_cloud.ply"))

        total_loss = 0
        optimizer.zero_grad()  # 清除梯度
        reconstructed_color, _ = model.color_forward(gs_color)
        reconstructed_position, _ = model.position_forward(gs_position)
        loss1 = mse_loss(reconstructed_color, gs_color)
        loss2 = mse_loss(reconstructed_position, gs_position)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        ema_loss_for_log = 0.4 * total_loss + 0.6 * ema_loss_for_log
        if epoch % 10 == 0:
            progress_bar.set_postfix(
                {"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if epoch == epochs:
            progress_bar.close()


def get_gs_fea(gaussians: GaussianModel):
    g_xyz = gaussians.get_xyz.detach()  # 获取3D点坐标数据并从计算图中分离
    n_gaussian = g_xyz.shape[0]  # 获取点云数据中点的数量

    _features_dc = gaussians._features_dc.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 3]
    _features_rest = gaussians._features_rest.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 45]
    _opacity = gaussians._opacity.detach()  # 从计算图中分离透明度数据 [N, 1]
    _scaling = gaussians._scaling.detach()  # 从计算图中分离缩放数据 [N, 3]
    _rotation = gaussians._rotation.detach()  # 从计算图中分离旋转数据 [N, 4]

    gs_color = torch.cat([_features_dc, _features_rest, _opacity], dim=-1) # [N, 49]
    gs_position = torch.cat([_scaling, _rotation], dim=-1) # [N, 7]

    return gs_color, gs_position


# 预处理高斯模型，用于后续输入训练
def handle_train(gaussians: GaussianModel, first, epochs, path):
    gs_color, gs_position = get_gs_fea(gaussians)
    model = Autoencoder().cuda()
    train_model(model, gaussians, gs_color=gs_color, gs_position = gs_position, first = first, epochs=epochs, path=path, learning_rate=1e-3)


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
    parser.add_argument("--train_iteration", nargs="+", type=int, default=[5000])
    parser.add_argument('--hidden', type=int, default=32)
    args = parser.parse_args(sys.argv[1:])

    gaussians = get_gs_model(args.model_path)
    gaussians.save_ply(os.path.join(args.model_path, 'source_gs.ply'))
    handle_train(gaussians, args.first_iteration, args.train_iteration, args.model_path)

    print("\nTraining complete.")
