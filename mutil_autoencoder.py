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
    # def __init__(self, feat_dim=56, hidden=32):
    def __init__(self, fea_dim=48, op_dim=1, sc_dim=3, rt_dim=4, fea_latent=32, op_latent=1, sc_latent=3,
                 rt_latent=4, fea_hidden=512, op_hidden=64, sc_hidden=128, rt_hidden=128):
        super(Autoencoder, self).__init__()

        self.fea_dim = fea_dim
        self.op_dim = op_dim
        self.sc_dim = sc_dim
        self.rt_dim = rt_dim

        self.fea_latent = fea_latent
        self.op_latent = op_latent
        self.sc_latent = sc_latent
        self.rt_latent = rt_latent

        self.fea_hidden = fea_hidden
        self.op_hidden = op_hidden
        self.sc_hidden = sc_hidden
        self.rt_hidden = rt_hidden

        # sh特征
        self.fea_encoder = nn.Sequential(
            nn.Linear(self.fea_dim, fea_hidden),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(fea_hidden, self.fea_latent)  # 隐藏层到潜在空间
        )
        self.fea_decoder = nn.Sequential(
            nn.Linear(self.fea_latent, fea_hidden),  # 潜在空间到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(512, self.fea_dim)  # 隐藏层到输出层
        )

        # 透明度
        self.op_encoder = nn.Sequential(
            nn.Linear(self.op_dim, op_hidden),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(op_hidden, self.op_latent)  # 隐藏层到潜在空间
        )
        self.op_decoder = nn.Sequential(
            nn.Linear(self.op_latent, op_hidden),  # 潜在空间到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(op_hidden, self.op_dim)  # 隐藏层到输出层
        )

        # 缩放
        self.sc_encoder = nn.Sequential(
            nn.Linear(self.sc_dim, sc_hidden),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(sc_hidden, self.sc_latent)  # 隐藏层到潜在空间
        )
        self.sc_decoder = nn.Sequential(
            nn.Linear(self.sc_latent, sc_hidden),  # 潜在空间到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(sc_hidden, self.sc_dim)  # 隐藏层到输出层
        )

        # 旋转
        self.rt_encoder = nn.Sequential(
            nn.Linear(self.rt_dim, rt_hidden),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(rt_hidden, self.rt_latent)  # 隐藏层到潜在空间
        )
        self.rt_decoder = nn.Sequential(
            nn.Linear(self.rt_latent, rt_hidden),  # 潜在空间到隐藏层
            nn.LeakyReLU(),  # 非线性激活函数
            nn.Linear(rt_hidden, self.rt_dim)  # 隐藏层到输出层
        )

    # sh forward
    def fea_forward(self, x):
        z = self.fea_encoder(x)
        reconstructed = self.fea_decoder(z)
        return reconstructed, z

    # 透明度forward
    def op_forward(self, x):
        z = self.op_encoder(x)
        reconstructed = self.op_decoder(z)
        return reconstructed, z

    # 缩放forward
    def sc_forward(self, x):
        z = self.sc_encoder(x)
        reconstructed = self.sc_decoder(z)
        return reconstructed, z

    # 旋转forward
    def rt_forward(self, x):
        z = self.rt_encoder(x)
        reconstructed = self.rt_decoder(z)
        return reconstructed, z

    def encoder(self, fea_all):

        gs_dc, gs_rest, gs_op, gs_sc, gs_rt = all_to_separate(fea_all)
        gs_fea = torch.cat([gs_dc, gs_rest], dim=-1)
        r_fea, l_fea = self.fea_encoder(gs_fea)
        l_dc = l_fea[:, :3]  # 提取前 3 列 [N, 3]
        l_rest = l_fea[:, 3:48]  # 提取第 4 到第 48 列 [N, 45]
        r_op, l_op = self.op_encoder(gs_op)
        r_sc, l_sc = self.sc_encoder(gs_sc)
        r_rt, l_rt = self.rt_encoder(gs_rt)
        l = separate_to_all(l_dc, l_rest, l_op, l_sc, l_rt)
        return l

    def decoder(self, lat_all):


        pass


def train_model(model, gs: GaussianModel, gs_all, gs_fea, gs_op, gs_sc, gs_rt, first, epochs, path, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, epochs[-1]), desc="Training progress")
    model.train()
    for epoch in range(first, epochs[-1] + 1):
        if epoch in epochs:

            r_fea, l_fea = model.fea_forward(gs_fea)
            r_op, l_op = model.op_forward(gs_op)
            r_sc, l_sc = model.sc_forward(gs_sc)
            r_rt, l_rt = model.rt_forward(gs_rt)

            reconstructed = torch.cat([r_fea, r_op, r_sc, r_rt], dim=-1)
            latent_space = torch.cat([l_fea, l_op, l_sc, l_rt], dim=-1)

            # 保存AE模型
            model_path = os.path.join(path, "model", "autoencoder_" + str(epoch) + ".pth")
            save_model(model, model_path)
            # 保存压缩存储的点云
            compress_ply_path = os.path.join(path, "compress", "compress_" + str(epoch) + ".ply")
            save_compress_latent_ply(latent_space, gs, compress_ply_path)

            set_gs_with_reconstructed(reconstructed,gs)
            gs.save_ply("output/tandt/train/tempgs.ply")

            # # 从压缩点云中重建高斯
            # reconstructed_gs = load_compress_latent_ply(model, compress_ply_path)
            # # 保存重建高斯到点云文件
            # reconstructed_path = os.path.join(path, "reconstructed",
            #                                   "point_cloud_" + str(epoch) + ".ply")
            # reconstructed_gs.save_ply(reconstructed_path)
            # # 训练结束时，保存一个命名好的点云，方便使用
            # if epoch == epochs[-1]:
            #     reconstructed_gs.save_ply(os.path.join(path, "reconstructed", "point_cloud.ply"))

        total_loss = 0
        optimizer.zero_grad()  # 清除梯度

        r_fea, _ = model.fea_forward(gs_fea)
        r_op, _ = model.op_forward(gs_op)
        r_sc, _ = model.sc_forward(gs_sc)
        r_rt, _ = model.rt_forward(gs_rt)
        reconstructed = torch.cat([r_fea, r_op, r_sc, r_rt], dim=-1)

        # reconstructed, _ = model(gs_fea)
        # loss = l1_loss(reconstructed,gs_fea)
        loss = mse_loss(reconstructed, gs_all)
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

    _features_dc = gaussians._features_dc.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 3]
    _features_rest = gaussians._features_rest.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 45]
    _opacity = gaussians._opacity.detach()  # 从计算图中分离透明度数据 [N, 1]
    _scaling = gaussians._scaling.detach()  # 从计算图中分离缩放数据 [N, 3]
    _rotation = gaussians._rotation.detach()  # 从计算图中分离旋转数据 [N, 4]

    gs_fea = torch.cat([_features_dc, _features_rest], dim=-1)

    gs_all = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation],
                       dim=-1)  # 将各类特征拼接成一个大的特征向量 [N, 56]

    return gs_all, gs_fea, _opacity, _scaling, _rotation


# 预处理高斯模型，用于后续输入训练
def handle_train(gaussians: GaussianModel, first, epochs, path):
    gs_all, gs_fea, gs_op, gs_sc, gs_rt = get_gs_fea(gaussians)
    model = Autoencoder().cuda()
    train_model(model, gaussians, gs_all=gs_all, gs_fea=gs_fea, gs_op=gs_op, gs_sc=gs_sc, gs_rt=gs_rt, first=first,
                epochs=epochs,
                path=path, learning_rate=1e-3)


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

    gaussians.save_ply("ae/model/source_gs.ply")
    handle_train(gaussians, args.first_iteration, args.train_iteration, args.model_path)

    print("\nTraining complete.")
