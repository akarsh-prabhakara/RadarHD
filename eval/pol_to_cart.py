# Uses the generated polar upsampled radar images from the output of 'test_radarhd.py' and converts it into cartesian 
# format in a specific folder structure

from PIL import Image
import numpy as np
import glob as glob
import sys
import os

params = {
    'model_name': '13',
    'expt': 1,
    'dt': '20220320-034822',
    'epoch_num': 120,
}

RMAX = 10.8
RBINS = 256
ABINS = 512

# ########################################

def convert_pol2cart(a):
    b = np.zeros((RBINS, ABINS))
    loc = np.argwhere(a>0)
    xloc = loc[:,0]
    yloc = loc[:,1]
    x = x_axis[xloc,yloc]
    y = y_axis[xloc,yloc]
    new_xloc = [np.argmax(x_axis_grid>=x[i]) for i in range(len(x))]
    new_yloc = [np.argmax(y_axis_grid>=y[i]) for i in range(len(y))]
    b[new_xloc,new_yloc] = a[xloc,yloc]
    return b

# ########################################

name_str = params['model_name'] + '_' + str(params['expt']) + '_' + params['dt']
img_dir = '../logs/' + name_str + '/test_imgs/'
save_path = './processed_imgs_' + name_str + '_test_imgs/'

os.system('rm -rf ' + save_path)
os.makedirs(save_path)

agrid = np.linspace(-90,90,ABINS)
rgrid = np.linspace(0,RMAX,RBINS)

cosgrid = np.cos(agrid * np.pi / 180)
singrid = np.sin(agrid * np.pi / 180)

sine_theta,range_d = np.meshgrid(singrid,rgrid)
cos_theta = np.sqrt(1-sine_theta**2)

x_axis = np.multiply(range_d, cos_theta)
y_axis = np.multiply(range_d, sine_theta)

x_axis_grid = np.linspace(0,RMAX,RBINS)
y_axis_grid = np.linspace(-RMAX,RMAX,ABINS)

# Convert pol to cartesian
files = sorted(glob.glob(img_dir + '*'))
for file in files:
    name = file[file.rfind('/')+1:-4]
    a = Image.open(file)
    a = np.asarray(a)
    a = convert_pol2cart(a)
    a = a.astype(np.uint8)
    a = Image.fromarray(a)
    savename  = save_path + name + '.png'
    a.save(savename)

# Some automated organization
files = glob.glob(save_path + '*.png')
traj_num = [file[file.rfind('/')+5:file.rfind('/')+8] for file in files]
trajs = set(traj_num)

epoch_num = [file[file.rfind('/')+1:file.rfind('/')+4] for file in files]
epochs = set(epoch_num)

for traj in trajs:
    TRAJ_DIR = save_path + traj + '/'
    if not os.path.exists(TRAJ_DIR):
        os.makedirs(TRAJ_DIR)  
    for epoch in epochs:
        EPOCH_DIR = TRAJ_DIR + epoch + '/'
        if not os.path.exists(EPOCH_DIR):
            os.makedirs(EPOCH_DIR)
        PRED_DIR = EPOCH_DIR + 'pred/'
        if not os.path.exists(PRED_DIR):
            os.makedirs(PRED_DIR)   
        LABEL_DIR = EPOCH_DIR + 'label/'
        if not os.path.exists(LABEL_DIR):
            os.makedirs(LABEL_DIR)
        os.system('mv ' + save_path + epoch + '_' + traj + '*_pred.png '+PRED_DIR)
        os.system('mv ' + save_path + epoch + '_' + traj + '*_label.png '+LABEL_DIR)

    # ALL_LABEL_DIR = TRAJ_DIR + 'label/'
    # if not os.path.exists(ALL_LABEL_DIR):
    #     os.makedirs(ALL_LABEL_DIR)      
    # os.system('cp ' + LABEL_DIR + '*' + ' ' + ALL_LABEL_DIR)