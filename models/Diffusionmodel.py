import random
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd

import torch
import torchvision
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn

from scipy.interpolate import griddata
import cv2
from tqdm import tqdm
'''
DDPM(Ho et al., 2020)
'''

class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and ((t <= self.num_timesteps).all())
        t_idx = t - 1
        
        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise
    
    def denoise(self, model, x, cond, t, gamma):
        T = self.num_timesteps
        assert (t >= 1).all() and ((t <= self.num_timesteps).all())
        
        t_idx = t - 1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]
        
        N = alpha_bar.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)
        
        model.eval()
        with torch.no_grad():
                
            eps_cond = model(t, x, cond)
            
            nocond = torch.zeros_like(x, device=self.device)
                
            eps_uncond = model(t, x, nocond)
            
            eps = eps_uncond + gamma * (eps_cond - eps_uncond)

        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps ) / torch.sqrt(alpha)
    
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))

        return mu + noise * std
    
    def reverse_to_data(self, x):
        #標準化の解除、numpy化
        invTrans = transforms.Compose([ transforms.Normalize(mean = 0., std = 1/hr_std),
                                        transforms.Normalize(mean = -hr_mean, std = 1), ])
        
        image = x.to('cpu')
        image = invTrans(image)
        image = image.numpy()

        return np.transpose(image,(1,2,0))

    def sample_asim(self, model, cond, obs, x_shape=(16, 1, 64, 64),gamma=0.0, asim_sample=100):
        #データ同化処理
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        obs = obs.to('cpu')
        
        obs_points = self.rsampling(obs.detach().numpy().copy(), asim_sample)
        interpolate = torch.tensor(self.makeinterpolategrid(obs_points, size=x_shape[2:]),device=self.device)
        mask = torch.tensor(self.positionmask(obs_points, size=x_shape[2:]),device=self.device)
        
        for i in tqdm(range(self.num_timesteps, 0, -1)): 
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)    
            x = self.denoise(model, x, cond, t, gamma)
            
            noised_interp, noise = self.add_noise(interpolate, t)
            
            xunknown = (1-mask) * x
            xknown = interpolate * mask


                
            x = (xunknown + xknown).to(torch.float32)
            #print(x.shape)
        
        images = [self.reverse_to_data(x[i]) for i in range(batch_size)]
        return x, images

    def rsampling(self, darray, num_sample):
        #観測データサンプリング（ランダム）
        x_size, y_size = darray.shape[2:]
        a = np.arange(x_size*y_size)
        np.random.shuffle(a)
        point = a[:num_sample]
        sample_array = []
        for i in point:
            y_idx = i // x_size
            x_idx = i % x_size
            sample = y_idx, x_idx, darray[:,:, x_idx, y_idx]
            sample_array.append(sample)
        return sample_array

    def fixedsampling(self, darray,coordarr):
        #観測データサンプリング（固定）
        sample_array = []
        for i in coordarr:
            y_idx = i[0]
            x_idx = i[1]
            sample = y_idx, x_idx, darray[:,:, x_idx, y_idx]
            sample_array.append(sample)
        return sample_array

    def makeinterpolategrid(self, obsarr, size=(64,64)):
        #観測値補間
        coordarr = [i[0:2] for i in obsarr] 
        data = [i[2] for i in obsarr] 
        latarr, longarr = np.meshgrid(range(size[0]),range(size[1]))
        
        #内挿値
        result1 = griddata(points=coordarr, values=data, xi=(latarr, longarr),method='cubic', fill_value=0).transpose(2,3,0,1)
        
        #外挿値
        result2 = griddata(points=coordarr, values=data, xi=(latarr, longarr),method='nearest').transpose(2,3,0,1)
    
        nan_mask = result1[0] == 0
        result = result2 * nan_mask + result1
        return result
    
    #ガウシアンマスクの作成
    def _positionmask(self, coord, maskarr, kernel):
        temp = np.zeros(maskarr.shape)
        
        latidx = coord[0] 
        lonidx = coord[1]
        
        temp[lonidx,latidx] = 1
        temp = cv2.filter2D(temp,-1,kernel,borderType=cv2.BORDER_ISOLATED)
        maskarr = np.fmax(temp,maskarr)
        return maskarr
    
    def positionmask(self, obsarr, size=(64,64)):
        kernelg=cv2.getGaussianKernel(11,2.)
        gaussian_2d = kernelg * kernelg.T
        
        maskarr = np.zeros(size)
        coordarr = [i[0:2] for i in obsarr]
        for coord in coordarr:
            maskarr = self._positionmask(coord, maskarr, gaussian_2d)
    
        return maskarr