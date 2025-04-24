'''
ノイズ除去モデル
diffusersライブラリのUNet2DConditionModelを使用
低解像度画像、時刻、ノイズ画像（t）→ノイズ除去画像(tー1)
CLIPによる低解像度画像、日付の埋め込み
'''

import random
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd

from diffusers import UNet2DConditionModel
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn

import math
from tqdm import tqdm

configuration = CLIPVisionConfig(
    projection_dim = 512,
    num_channels = 1,
    image_size = 64
)

class DateTimeEmbeddingWithPeriodicity(nn.Module):
    def __init__(self, embedding_dim=256):
        super(DateTimeEmbeddingWithPeriodicity, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(6, self.embedding_dim)
        
    def forward(self, date):
        """
        date: pandas.Timestamp or datetime-like object
        """
        # 日付から年、月、日、曜日を抽出
        month = date.month
        day = date.day

        # 時間から時、分、秒を抽出
        hour = date.hour
        minute = date.minute
        second = date.second
        
        # 月の周期的な埋め込み
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # 日の周期的な埋め込み
        day_sin = np.sin(2 * np.pi * day / 30)
        day_cos = np.cos(2 * np.pi * day / 30)
        
        # 時間の周期的な埋め込み
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # 埋め込みベクトルとして結合
        embedding = torch.tensor(np.stack([month_sin, month_cos, 
                                 day_sin, day_cos,
                                 hour_sin, hour_cos,
                                    ],axis=1), dtype=torch.float32)
        #　次元を調整して返す
        return self.fc(embedding)

class DenoisingModel(nn.Module):
    def __init__(self):
        super(DenoisingModel, self).__init__()
        self.lr_encoder = CLIPVisionModelWithProjection(configuration)
        self.unet = UNet2DConditionModel(
                        sample_size=config.image_size,  # the target image resolution
                        #入出力サイズ
                        in_channels=1,
                        out_channels=1, 
                        #モデル構造の指定    
                        layers_per_block=2,  # how many ResNet layers to use per UNet block
                        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
                        down_block_types=(
                            "DownBlock2D",  # a regular ResNet downsampling block
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                            "DownBlock2D",
                        ),
                        up_block_types=(
                            "UpBlock2D",  # a regular ResNet upsampling block
                            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                        ),
                        #埋め込みデータ指定
                        encoder_hid_dim_type = 'image_proj',
                        encoder_hid_dim = 512,
                        #addition_embed_type = 'text'
                    )

    def forward(self, t, noise, lr):
        lr_encoded = self.lr_encoder(lr).image_embeds.unsqueeze(1)
        denoised = self.unet(noise, t, lr_encoded, added_cond_kwargs={'image_embeds':lr_encoded}).sample
        return denoised 