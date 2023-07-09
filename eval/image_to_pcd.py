# Uses the cartesian format images output from 'pol_to_cart.py', thresholds it and converts the image to a point cloud
# for evaluation 

import glob, os
import cv2
import numpy as np
import open3d as o3d

params = {
    'model_name': '13',
    'expt': 1,
    'dt': '20220320-034822',
    'epoch_num': 120,
}

RMAX = 10.8
RBINS = 256
ABINS = 512
MIN_THRESHOLD = 1
MAX_THRESHOLD = 255

# ########################################

def convert2pcd(img_files, DIR):
    print(DIR)

    PCD_DIR = DIR + 'pcd/'
    if not os.path.exists(PCD_DIR):
        os.makedirs(PCD_DIR)

    for i in range(len(img_files)):
        filename = os.path.basename(img_files[i]).split('.')[0]
        img = cv2.imread(img_files[i], 0)
        ret, thresh_img = cv2.threshold(img,MIN_THRESHOLD,MAX_THRESHOLD,cv2.THRESH_TOZERO)

        # 0 dim is azimuth, 1 dim is range
        location = np.squeeze(cv2.findNonZero(thresh_img))

        if location.size == 1:
            dummy = np.column_stack((np.array([0]), np.array([0]), np.array([0])))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dummy)     
        else:       
            y_location = y_axis_grid[location[:,0]]
            x_location = x_axis_grid[location[:,1]]
            point_loc_3d = np.column_stack((x_location, y_location, np.zeros(location.shape[0])))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_loc_3d)

        o3d.io.write_point_cloud(PCD_DIR + filename + ".pcd", pcd)

# ########################################

name_str = params['model_name'] + '_' + str(params['expt']) + '_' + params['dt']
save_path = './processed_imgs_' + name_str + '_test_imgs/'
print(save_path)

trajs = sorted(glob.glob(save_path + '*'), key=lambda x:int(os.path.basename(x)))
print(trajs)

epoch = '%03d' % params['epoch_num']

x_axis_grid = np.linspace(0,RMAX,RBINS)
y_axis_grid = np.linspace(-RMAX,RMAX,ABINS)

for traj in trajs:
    TRAJ_DIR = traj + '/'
    EPOCH_DIR = TRAJ_DIR + epoch + '/'
    PRED_DIR = EPOCH_DIR + 'pred/' 
    LABEL_DIR = EPOCH_DIR + 'label/'

    img_files = sorted(glob.glob(PRED_DIR+'*.png'),key=lambda x: int(os.path.basename(x).split('_')[2]))
    convert2pcd(img_files, PRED_DIR)

    img_files = sorted(glob.glob(LABEL_DIR+'*.png'),key=lambda x: int(os.path.basename(x).split('_')[2]))
    convert2pcd(img_files, LABEL_DIR)