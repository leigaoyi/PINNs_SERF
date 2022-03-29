# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:39:53 2022

@author: leisir
绘制沿着x轴的极化率分布曲线，间隔0.05cm
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import pandas as pd

filename = './result/COMSOL_Rop_32/z0.csv'
df = pd.read_csv(filename, header=9, names=['x', 'y', 'z', 'Pz'])
## (x, y, z) u1, 20796*4

x = np.asarray(df['x'])
y = np.asarray(df['y'])
z = np.asarray(df['z'])
Pz = np.asarray(df['Pz'])

## 放到区间下，
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
np.savez('./result/Z0', x_axis = x_plot, polarization=p_plot)
plt.figure()      
plt.plot((X_lim+x_space)[:-1], Pz_density)
plt.xlim(-1.5, 1.5)
plt.show()        
