# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:45:06 2022

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

location = '0'
COMSOL_result_path = './draw_figures/Z{0}.npz'.format(location)
pred_result_path = './draw_figures/Z{0}_pred.npz'.format(location)

COMSOL_data = np.load(COMSOL_result_path)
pred_data = np.load(pred_result_path)

comsol_x = COMSOL_data['x_axis']
comsol_p = COMSOL_data['polarization']

pred_x = pred_data['x_axis']
pred_p = pred_data['polarization']

plt.figure(figsize=(10,10))
plt.plot(comsol_x, comsol_p, label='COMSOL')
plt.plot(pred_x, pred_p, marker='x', linestyle='', label='PINNs')
plt.title('Polarization distribution at Z axis {0} cm'.format(location))
plt.legend()
plt.show()
plt.savefig('./draw_figures/polarization_dist_z_{0}.png'.format(location), dpi=320)