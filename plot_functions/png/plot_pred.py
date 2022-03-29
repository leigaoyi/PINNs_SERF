# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:39:53 2022

@author: leisir
绘制沿着x轴的极化率分布曲线，间隔0.05cm
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

X_data_list = ['axes_0.8.npy', 'axes_0.4.npy', 'axes_0.npy',\
               'axes_-0.4.npy', 'axes_-0.8.npy']
    
Pz_list = ['pred_0.8.npy', 'pred_0.4.npy', 'pred_0.npy',\
           'pred_-0.4.npy', 'pred_-0.8.npy']

save_list = ['PINNs_0.8.png', 'PINNs_0.4.png', 'PINNs_0.png',\
             'PINNs_-0.4.png', 'PINNs_-0.8.png']    
# pred_vector_path = './result/Z0.8_pred'
# X_data = np.load('./result/axes_0.8.npy')
# Pz = np.load('./result/pred_0.8.npy')#[:1000]

for i in range(len(X_data_list)):
    X_data = np.load('./result/'+X_data_list[i])
    Pz = np.load('./result/'+Pz_list[i])
    
    x, y = X_data[:, 0], X_data[:, 1]
    
    
    ## 放到区间下，
    
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, s=50, c=Pz, alpha=0.7, cmap='rainbow')
    #plt.colorbar(label="Polarization Ratio") 
    plt.title(Pz_list[i])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    #plt.show()
    plt.savefig('./draw_figures/'+save_list[i], dpi=320)
