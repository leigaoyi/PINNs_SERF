# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:08:34 2022

@author: leisir
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import pandas as pd
import os
import csv

file_list = ['z0.8.csv', 'z0.4.csv', 'z0.csv']

for i in file_list:
    filebase = os.path.basename(i)
    filename = './result/COMSOL_Rop_32/'+i
#filename = './result/COMSOL_Rop_32/z0.8.csv'
    
    df = pd.read_csv(filename, header=9, names=['x', 'y', 'z', 'Pz'])
    ## (x, y, z) u1, 20796*4
    
    x = np.asarray(df['x'])
    y = np.asarray(df['y'])
    z = np.asarray(df['z'])
    Pz = np.asarray(df['Pz'])
    
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, s=50, c=Pz, alpha=0.7, cmap='rainbow')
    #plt.colorbar(label="Polarization Ratio") 
    plt.title('COMSOL {0}'.format(filebase))
    #plt.show()
    plt.savefig('./draw_figures/COMSOL_{0}.png'.format(filebase), dpi=320)
