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

from vector_quantize_pytorch import ResidualVQ

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.loss_utils import mse_loss

from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.save_and_load import *


class RVQuantizer(nn.Module):
    def __init__(self, latent_dim=32):
        super(RVQuantizer, self).__init__()
        self.vq = ResidualVQ(
            dim=latent_dim,
            codebook_size=512,
            num_quantizers=10,
            commitment_weight=0.0,
            kmeans_init=True,
            kmeans_iters=1,
            ema_update=False,
            learnable_codebook=True,
            in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)
        )

    def forward(self, z):
        # 量化码本，损失，量化码本索引
        quantized, indices, commit_loss = self.vq(z)
        return quantized, indices, commit_loss


class Autoencoder(nn.Module):
    def __init__(self, feat_dim=56, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        # 编码器：提取点云特征
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, latent_dim)  # 映射到潜在空间
        )

        # 解码器：重建点云数据
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, feat_dim)  # 还原到原始点云特征维度
        )

    def forward(self, x):
        z = self.encoder(x)  # 编码
        reconstructed = self.decoder(z)  # 解码
        return reconstructed, z


def train_model(ae_model: Autoencoder, rvq_model: RVQuantizer, gs: GaussianModel, gs_fea, ae_first, ae_epochs, rvq_first, rvq_epochs, path,
                learning_rate=1e-3):
    optimizer_ae = torch.optim.Adam(ae_model.parameters(), lr=learning_rate)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, ae_epochs[-1]), desc="Training progress")
    ae_model.train()
    for epoch in range(ae_first, ae_epochs[-1] + 1):

        total_loss = 0
        optimizer_ae.zero_grad()  # 清除梯度
        reconstructed, z = ae_model(gs_fea)

        # 损失
        loss_mse = mse_loss(reconstructed, gs_fea)
        # loss_entropy = entropy_loss(z)

        # loss = loss_mse + 0.1 * loss_entropy
        loss = loss_mse

        loss.backward()
        optimizer_ae.step()
        total_loss += loss.item()
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if epoch % 10 == 0:
            progress_bar.set_postfix(
                {"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if epoch == ae_epochs[-1]:
            progress_bar.close()

    final_reconstructed, final_latent = ae_model(gs_fea)
    # progress_bar = tqdm(range(0, rvq_epochs[-1]), desc="Training RVQuantizer")
    rvq_model.eval()
    # for epoch in range(rvq_first, rvq_epochs[-1] + 1):
    #     quantized, indices, commit_loss = rvq_model(final_latent)
    #     if epoch % 10 == 0:
    #         progress_bar.update(10)
    #     if epoch == rvq_epochs[-1]:
    #         progress_bar.close()

    # 保存AE模型
    model_path = os.path.join(path, "model", "autoencoder_" + str(epoch) + ".pth")
    save_model(ae_model, model_path)

    final_quantized, final_indices, final_commit_loss = rvq_model(final_latent)
    final_codebook = rvq_model.vq.codebooks
    rvq_dir = os.path.join(path, "rvq")
    os.makedirs(rvq_dir, exist_ok=True)
    # 将模型保存到 'codebook.pt' 文件中
    torch.save(rvq_model.vq.state_dict(), os.path.join(rvq_dir, "rvq_model.pt"))
    # 将码本和索引保存到 'codebook.pt' 文件中
    torch.save(dict(codebook=final_codebook, indices=final_indices), os.path.join(rvq_dir, "codebook.pt"))

    # 使用索引从码本中重建量化向量
    reconstructed_quantized = rvq_model.vq.get_codes_from_indices(final_indices).sum(dim=0)
    # 保存压缩存储的compress点云ply
    compress_ply_path = os.path.join(path, "compress", "compress_" + str(epoch) + ".ply")
    # save_compress_latent_ply(final_latent, gs, compress_ply_path)
    save_compress_latent_ply(reconstructed_quantized, gs, compress_ply_path)
    # 从compress压缩点云中重建高斯点云
    reconstructed_gs = load_compress_latent_ply(ae_model, compress_ply_path)
    # 保存重建高斯到point_cloud点云文件
    reconstructed_path = os.path.join(path, "reconstructed",
                                      "point_cloud_" + str(epoch) + ".ply")
    reconstructed_gs.save_ply(reconstructed_path)

    # 训练结束时，保存一个命名好的点云，方便使用
    reconstructed_gs.save_ply(os.path.join(path, "point_cloud", "iteration_30000", "point_cloud.ply"))


def get_gs_fea(gaussians: GaussianModel):
    g_xyz = gaussians.get_xyz.detach()  # 获取3D点坐标数据并从计算图中分离
    n_gaussian = g_xyz.shape[0]  # 获取点云数据中点的数量
    _features_dc = gaussians._features_dc.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 3]
    _features_rest = gaussians._features_rest.detach().view(n_gaussian, -1)  # 从计算图中分离并重塑为[N, 45]
    _opacity = gaussians._opacity.detach()  # 从计算图中分离透明度数据 [N, 1]
    _scaling = gaussians._scaling.detach()  # 从计算图中分离缩放数据 [N, 3]
    _rotation = gaussians._rotation.detach()  # 从计算图中分离旋转数据 [N, 4]
    gs_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation],
                       dim=-1)  # 将各类特征拼接成一个大的特征向量 [N, 56]

    return gs_fea


def handle_train(gaussians: GaussianModel, ae_first, ae_epochs, rvq_first, rvq_epochs, path):
    # 创建目标文件夹ae_output，如果不存在的话
    output_dir = os.path.join('ae_output', path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 调用递归函数复制文件和子目录
    recursive_copy(path, output_dir)

    # 获取高斯特征
    gs_fea = get_gs_fea(gaussians)
    # 初始化模型
    ae_model = Autoencoder().cuda()
    rvq_model = RVQuantizer().cuda()
    # 训练模型
    train_model(ae_model=ae_model, rvq_model=rvq_model, gs=gaussians, gs_fea=gs_fea, ae_first=ae_first,
                ae_epochs=ae_epochs, rvq_first=rvq_first, rvq_epochs=rvq_epochs, path=output_dir, learning_rate=1e-3)


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

    parser.add_argument("--ae_first_iteration", nargs="+", type=int, default=0)
    parser.add_argument("--ae_iteration", nargs="+", type=int, default=[10000])

    parser.add_argument("--rvq_first_iteration", nargs="+", type=int, default=0)
    parser.add_argument("--rvq_iteration", nargs="+", type=int, default=[10])

    parser.add_argument('--hidden', type=int, default=32)
    args = parser.parse_args(sys.argv[1:])

    gaussians = get_gs_model(args.model_path)

    # gaussians.save_ply("ae/model/source_gs.ply")
    handle_train(gaussians, ae_first=args.ae_first_iteration, ae_epochs=args.ae_iteration,
                 rvq_first=args.rvq_first_iteration, rvq_epochs=args.rvq_iteration, path=args.model_path)

    print("\nTraining complete.")
