# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:14:55 2022

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import random


rcell = 1.5
sample_num = 20000

def random_circle_point(radius):
    '''
    Generate uniform circle sample with radius r

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    theta = random.uniform(0, 2*np.pi)
    r = random.uniform(0, radius)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    return x, y

def random_sphere_points(z):
    radius_z = np.sqrt(rcell**2 - z**2)
    sample_data = []
    
    for i in range(sample_num):
        (x, y) = random_circle_point(radius_z)
        sample_data.append((x, y, z))
    
    for theta in np.linspace(0, 2*np.pi, 100):
        x = radius_z * np.cos(theta)
        y = radius_z * np.sin(theta)
        sample_data.append((x, y, z))
        
    sample_data = np.asarray(sample_data, dtype=np.float32)
    return sample_data

def uniform_sample_points(z):
      radius_z = np.sqrt(rcell**2 - z**2)
      sample_data = []
      radius_list = np.sqrt(np.linspace(0, radius_z, 150))
      ## 边界上完整采样
      ## 内部，角度、尺寸遍历
      for theta in np.linspace(0, 2*np.pi, 150):
            for r in radius_list:
                  x = r*np.cos(theta)
                  y = r*np.sin(theta)
                  sample_data.append((x, y, z))
      sample_data = np.asarray(sample_data, dtype=np.float32)
      return sample_data  

    
# def plot_single(z_axis):
#     z_axis = -0.8
#     X = random_sphere_points(z= z_axis)
#     y_pred = model.predict(X)[:, 0]
#     x_axis, y_axis = X[:, 0], X[:, 1]
#     plt.scatter(x_axis, y_axis, s=50, c= y_pred, cmap='rainbow')
#     plt.colorbar(label="Polarization Ratio") 
#     plt.title('Prediction at Z axis of {0} cm'.format(z_axis))

