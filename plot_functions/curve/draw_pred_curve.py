# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:39:53 2022

@author: leisir
绘制沿着x轴的极化率分布曲线，间隔0.05cm
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

pred_list = ['Z0.8_pred', 'Z0.4_pred', 'Z0_pred',\
             'Z-0.4_pred', 'Z-0.8_pred']
X_data_list = ['axes_0.8.npy', 'axes_0.4.npy', 'axes_0.npy',\
               'axes_-0.4.npy', 'axes_-0.8.npy']
    
Pz_list = ['pred_0.8.npy', 'pred_0.4.npy', 'pred_0.npy',\
           'pred_-0.4.npy', 'pred_-0.8.npy']

save_list = ['PINNs_0.8.png', 'PINNs_0.4.png', 'PINNs_0.png',\
             'PINNs_-0.4.png', 'PINNs_-0.8.png']    

# pred_vector_path = './result/Z0.8_pred'
# X_data = np.load('./result/axes_0.8.npy')
# z = np.load('./result/pred_0.8.npy')#[:1000]
for i in range(len(pred_list)):
    pred_vector_path = './result/' + pred_list[i]
    X_data = np.load('./result/' + X_data_list[i])
    z = np.load('./result/'+Pz_list[i])

    x, y = X_data[:, 0], X_data[:, 1]
    
    
    ## 放到区间下，
    Pz = z
    num_space = 30
    X_lim = np.linspace(-1.5, 1.5, num_space+1)
    x_space = 3/num_space
    Pz_dist = np.zeros(num_space) ## Pz value add
    Pz_space_num = np.zeros(num_space) ## Pz number
    
    for x_idx in range(len(x)):
          Pz_idx = int((x[x_idx]+1.5)/x_space-1)
          Pz_dist[Pz_idx] += Pz[x_idx]
          Pz_space_num[Pz_idx] += 1
     
    Pz_density = Pz_dist/Pz_space_num      
    x_plot = (X_lim+x_space)[:-1]
    p_plot = Pz_density
    np.savez(pred_vector_path, x_axis = x_plot, polarization=p_plot)
    ## 读取的时候，data = np.load('xx.npz'), x = data['x_axis']
    plt.figure(figsize=(10,10))     
    plt.plot((X_lim+x_space)[:-1], Pz_density)
    plt.xlim(-1.5, 1.5)
    plt.show()      

