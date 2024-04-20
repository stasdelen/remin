# Geometric Parameters

xmin = -0.5
xmax =  2.0
ymin = -0.5
ymax =  1.5

# tstart = 0
# tfinal = 1.0

# Flow Parameters
Re = 10
u0 = 0.1581
p0 = 0.05
RT = 100
c  = 17.3205
L  = 1

sqrtRT = 10

nu  = L * u0 / Re
tau = nu / RT

save_folder = './outputs/kovasznay_pinnbgk2'

# Learning Parameters
epochs = 20000
lr     = 1e-4
gamma  = 0.99

mesh_file = 'kovasznay.msh'
