from torch import dtype

from scene.gaussian_model import GaussianModel
from utils.system_utils import mkdir_p
import os as os
import numpy as np
from plyfile import PlyData, PlyElement
from torch import nn

import torch


def construct_list_of_attributes(hidden):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

    # 为每个隐藏层创建一个保存用ply
    for i in range(hidden):
        l.append('latent_{}'.format(i))
    return l

def save_model(model, model_filename='model/autoencoder.pth'):
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")


def save_latent_space(latent_space, latent_filename='model/latent_space.pth'):
    torch.save(latent_space, latent_filename)
    print(f"Latent space saved to {latent_filename}")


def load_model(model, model_filename='model/autoencoder.pth'):
    model.load_state_dict(torch.load(model_filename))
    model.eval()  # 设置为评估模式
    print(f"Model loaded from {model_filename}")
    return model


def load_latent_space(latent_filename='model/latent_space.pth'):
    """加载潜在空间"""
    latent_space = torch.load(latent_filename)
    print(f"Latent space loaded from {latent_filename}")
    return latent_space


def latent_ply(reconstructed, gs: GaussianModel, path):
    set_gs_with_reconstructed(reconstructed, gs)
    gs.save_ply(path)

def set_gs_with_reconstructed(reconstructed, gs:GaussianModel):
    N = reconstructed.size(0)

    _features_dc = reconstructed[:, :3]  # 提取前 3 列 [N, 3]
    _features_rest = reconstructed[:, 3:48]  # 提取第 4 到第 48 列 [N, 45]

    _features_dc = _features_dc.view(N, 1, 3)  # 这样会得到 [N, 1, 3]
    _features_rest = _features_rest.view(N, 15, 3)  # 这样会得到 [N, 15, 3]

    _opacity = reconstructed[:, 48:49]  # 提取第 49 列 [N, 1]
    _scaling = reconstructed[:, 49:52]  # 提取第 50 到第 52 列 [N, 3]
    _rotation = reconstructed[:, 52:]  # 提取剩余部分（第 53 到第 56 列）[N, 4]

    gs._features_dc = _features_dc
    gs._features_rest = _features_rest
    gs._opacity = _opacity
    gs._scaling = _scaling
    gs._rotation = _rotation

# 保存压缩后的ply文件
def save_compress_latent_ply(latent_space, gs: GaussianModel, path):
    mkdir_p(os.path.dirname(path))

    xyz = gs._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    latent = latent_space.detach().cpu().numpy()

    latent_space_size = latent_space.size(1)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(latent_space_size)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, latent), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


# 恢复压缩过的ply文件到gs模型
def load_compress_latent_ply(model, path):
    gs = GaussianModel(3)
    plydata = PlyData.read(path)
    gs.pretrained_exposures = None
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)

    latent_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("latent_")]
    latent_name = sorted(latent_name, key=lambda x: int(x.split('_')[-1]))
    latent = np.zeros((xyz.shape[0], len(latent_name)))
    for idx, attr_name in enumerate(latent_name):
        latent[:, idx] = np.asarray(plydata.elements[0][attr_name])

    latent = nn.Parameter(torch.tensor(latent, dtype=torch.float, device="cuda").requires_grad_(True))
    reconstructed = model.decoder(latent)
    set_gs_with_reconstructed(reconstructed, gs)
    gs._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    gs.active_sh_degree = 3
    return gs
