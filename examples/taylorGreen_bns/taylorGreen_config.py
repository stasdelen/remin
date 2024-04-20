# Geometric Parameters
import numpy as np
PI = np.pi

xmin = -PI
xmax = PI
ymin = -PI
ymax = PI

tstart = 0
tfinal = 10.0

# Flow Parameters
# Re = 10
# u0 = 1.0
RT = 100
# c  = 10.0
# L  = 1

sqrtRT = 10

nu  = 0.01
tau = nu / RT

save_folder = './outputs/ldc_pinnbgk_steady'

# Learning Parameters
epochs = 40000
lr     = 5e-4
gamma  = 0.99

mesh_file = 'taylorGreen.msh'
