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
from utils.loss_utils import mse_loss, l1_loss

from scene.gaussian_model import GaussianModel
from utils.save_and_load import *


# class Autoencoder(nn.Module):
#     def __init__(self, feat_dim=56, hidden=32):
#         super(Autoencoder, self).__init__()
#         self.feat_dim = feat_dim
#         self.hidden = hidden
#         self.encoder = nn.Sequential(
#             nn.Linear(self.feat_dim, 512),  # 输入层到隐藏层
#             nn.LeakyReLU(),  # 非线性激活函数
#             nn.Linear(512, self.hidden)  # 隐藏层到潜在空间
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(self.hidden, 512),  # 潜在空间到隐藏层
#             nn.LeakyReLU(),  # 非线性激活函数
#             nn.Linear(512, self.feat_dim)  # 隐藏层到输出层
#         )
#
#     def forward(self, x):
#         z = self.encoder(x)  # 编码器部分：输入到潜在空间的映射
#         reconstructed = self.decoder(z)  # 解码器部分：潜在空间到输出的映射
#         return reconstructed, z

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

def train_model(model, gs: GaussianModel, gs_fea, epochs=100, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, epochs), desc="Training progress")
    model.train()
    for epoch in range(epochs):
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

        # print(f'Epoch {epoch + 1}, Loss: {total_loss}')
    print("VAE Train Finish.")
    save_model(model)
    reconstructed, latent_space = model(gs_fea)
    save_latent_space(latent_space)
    latent_ply(reconstructed, gs, "ae/model/latent.ply")
    save_compress_latent_ply(latent_space, gs, "ae/model/compress_latent.ply")
    new_gs = load_compress_latent_ply(model, "ae/model/compress_latent.ply")
    new_gs.save_ply("ae/model/point_cloud.ply")


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


# 预处理高斯模型，用于后续输入训练
def handle_train(gaussians: GaussianModel, epochs):
    _, gs_fea = get_gs_fea(gaussians)
    model = Autoencoder().cuda()
    train_model(model, gaussians, gs_fea=gs_fea, epochs=epochs, learning_rate=1e-3)


def get_gs_model(path):
    gs = GaussianModel(sh_degree=3)
    gs.load_ply(path)
    return gs


def restore_gs_from_latent_space():
    pass


if __name__ == "__main__":
    gaussians = get_gs_model(
        "output/mipnerf360/bicycle/point_cloud/iteration_30000/point_cloud.ply")
    gaussians.save_ply("ae/model/source_gs.ply")
    handle_train(gaussians, 3)
