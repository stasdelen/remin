# Geometric Parameters
xmin = 	0.0
xmax =  5.0
ymin = -1.0
ymax =  1.0

tstart = 0
tfinal = 5.0


# Flow Parameters
uBar = 1.0
vBar = 0.0
rBar = 1.0
c    = 10.0
Ma   = uBar / c
RT   = c*c
Re   = 50
nu   = uBar*1.0 / Re
tau  = nu / (c*c)

save_folder = './outputs/channel_pinnbgk'

steady = False

# Learning Parameters
epochs = 50000
lr     = 1e-3
gamma  = 0.99

mesh_file = 'channel.msh'
