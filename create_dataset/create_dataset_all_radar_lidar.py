# This file creates the dataset required for training from timestamped radar and lidar

# This file assumes timestamp files indicating the global start and end time of radar and lidar 
# (timestamp_files_radar_lidar/)
# This file assumes that radar packets have been processed into into timestamped radar frames (radar/)
# This file assumes that lidar packets have been converted into timestamped 3D point clouds with 
# back half cut away (lidar_pcl/)

# After running this script, you will get a folder of corresponding lidar and radar images with the following structure
# dataset_xx/lidar/
# .................L_001_0.png
# .................L_001_1.png
# dataset_xx/radar/
# .................R_001_0.png
# .................R_001_1.png

# Prior to training or testing you should reorganize this into the following directory structure
# dataset_xx/train/
# .................lidar/***.png
# .................radar/***.png
# dataset_xx/test/
# .................lidar/***.png
# .................radar/***.png


import sys
import os
import pandas as pd
import numpy as np
from scipy.io import savemat
import pickle
from PIL import Image
import glob as glob

# Parameters
X_MAX = 10
Y_MAX = 10
Z_MIN = -0.3
Z_MAX = 0.3

R_MAX = 10.8
RBINS = 256
ABINS_LIDAR = 512
ABINS_RADAR = 64

azimuth_tx = [0, 2]
elev_tx = [1]
num_rx = [0, 1, 2, 3]
RANGE_FFT = 512
AZIM_FFT = 64
MAX_RANGE = 21.59
MAG_THRESHOLD = 0.05
RANGE_GUARD = 5
CFAR_THRESHOLD = 5
METHOD = 'mag'
folder = '../dataset_xx/'

sine_theta = -2*np.linspace(-0.5,0.5,AZIM_FFT)
cos_theta = np.sqrt(1-sine_theta**2)
theta_grid = np.flipud(np.arcsin(sine_theta)*180/np.pi)
range_d, sine_theta_mat = np.meshgrid(np.linspace(0,MAX_RANGE,RANGE_FFT),sine_theta)
_, cos_theta_mat = np.meshgrid(np.linspace(0,MAX_RANGE,RANGE_FFT),cos_theta)
x_axis = np.multiply(range_d, cos_theta_mat)
y_axis = np.multiply(range_d, sine_theta_mat)

# ########################################

def get_global_timestamps(index):
    timestamp = np.load('./timestamp_files_radar_lidar/' + str(index) + '_global_start_end_timestamp.npy')

    lf = open('./lidar_pcl/' + str(index) + '_fwd.csv', 'rb')
    lidar_time = pd.read_csv(lf,header=None,usecols=[4]).to_numpy()*1e6

    # Some time adjustment
    lidar_time -= 4*3600*1e6

    lidar_time = np.unique(lidar_time)

    if timestamp[0] == timestamp[2] and timestamp[1] == timestamp[3]:
        left_index = np.argmax(lidar_time >= timestamp[0])
        right_index = np.argmax(lidar_time >= timestamp[1])
        timestamps = lidar_time[left_index:right_index+1]
    
    return timestamps

def create_image(pcl_data, is_binary):
    image = np.zeros((XBINS, YBINS))

    # Filter pcl data based on x y z values
    mask = ((pcl_data[:,0] > 0) & ((pcl_data[:,0] <= X_MAX) & ((pcl_data[:,2] >= Z_MIN) & ((pcl_data[:,2] <= Z_MAX) & 
            ((pcl_data[:,1] >= -Y_MAX) & (pcl_data[:,1] <= Y_MAX))))))
    pcl_data = pcl_data[mask,:]
    x = pcl_data[:,0]
    y = pcl_data[:,1]
    intensity = pcl_data[:,3]

    # Find indices
    x_grid = np.linspace(0,X_MAX,XBINS)
    y_grid = np.linspace(-Y_MAX,Y_MAX,YBINS)

    for i in range(pcl_data.shape[0]):
        xi = np.argmax(x_grid>=x[i])
        yi = np.argmax(y_grid>=y[i])

        if image[xi,yi] == 0:
            if is_binary:
                image[xi,yi] = 1
            else:
                image[xi,yi] = intensity[i]

    if is_binary:
        image = image.astype(np.bool_)
    else:
        image = image.astype(np.uint16)

    return image

def pcl_to_polar(pcl_data):

    # Filter pcl data based on x y z values
    mask = ((pcl_data[:,0] > 0) & ((pcl_data[:,0] <= X_MAX) & ((pcl_data[:,2] >= Z_MIN) & ((pcl_data[:,2] <= Z_MAX) & 
            ((pcl_data[:,1] >= -Y_MAX) & (pcl_data[:,1] <= Y_MAX))))))
    pcl_data = pcl_data[mask,:]

    polar_data = np.zeros(pcl_data.shape)

    for i in range(pcl_data.shape[0]):
        xi = pcl_data[i,0]
        yi = pcl_data[i,1]
        zi = pcl_data[i,2]
        ri = np.sqrt(xi**2+yi**2+zi**2)
        ai = np.rad2deg(np.arctan2(yi,xi))
        ei = np.rad2deg(np.arcsin(zi/ri))

        polar_data[i,0] = ri
        polar_data[i,1] = ai
        polar_data[i,2] = ei
        polar_data[i,3] = pcl_data[i,3]
            
    return polar_data

def create_image_polar(polar_data, is_lidar, is_binary):

    mask = ((polar_data[:,0] > 0) & (polar_data[:,0] <= R_MAX))
    polar_data = polar_data[mask,:]

    if is_lidar == False:
        mask = ((polar_data[:,1] >= -70) & (polar_data[:,1] <= 70))
        polar_data = polar_data[mask,:]        

    r = polar_data[:,0]
    a = polar_data[:,1]
    intensity = polar_data[:,3]

    if is_lidar:
        image = np.zeros((RBINS, ABINS_LIDAR))
        r_grid = np.linspace(0,R_MAX,RBINS)
        a_grid = np.linspace(-90,90,ABINS_LIDAR)

    else:
        image = np.zeros((RBINS, ABINS_RADAR))
        r_grid = np.linspace(0,R_MAX,RBINS)
        a_grid = np.linspace(-90,90,ABINS_RADAR)
        intensity = 10*np.log10(intensity)

    min_intensity = np.min(intensity)
    max_intensity = np.max(intensity)
    intensity = (intensity - min_intensity) / (max_intensity-min_intensity)

    for i in range(polar_data.shape[0]):
        ri = np.argmax(r_grid>=r[i])
        ai = np.argmax(a_grid>=a[i])

        if (image[ri,ai] == 0) | ((intensity[i] > image[ri,ai]) & is_binary==False):
            if is_binary:
                image[ri,ai] = 1
            else:
                image[ri,ai] = intensity[i]

    if is_binary:
        image = image.astype(np.bool_)

    return image

def fft_2d(frame, input_tx, fft_out_dim):
    # shape (8/4 x 512)
    if len(input_tx) == 2:
        fft_array = np.concatenate((frame[0,:,:], frame[2,:,:]),axis=0)
    else:
        fft_array = frame[1,:,:]

    first_fft = np.fft.fft(fft_array, n=RANGE_FFT, axis=1)
    second_fft = np.fft.fft(first_fft, n=fft_out_dim, axis=0)
    second_fft = np.fft.fftshift(second_fft, axes=0)
    second_fft = np.abs(second_fft)

    return second_fft

#  This can be CFAR instead
def threshold(frame_fft):

    if METHOD == 'mag':
        # Magnitude only thresholding
        m = np.max(frame_fft[:,6:])
        idx = (frame_fft[:,6:] >= MAG_THRESHOLD*m)
        idx = np.concatenate((np.zeros((AZIM_FFT,6),dtype=bool),idx),axis=1)
    elif METHOD == 'cfar':
        # CFAR range only thresholding
        idx = np.zeros((AZIM_FFT,RANGE_FFT),dtype=bool)
        frame_fft = 10*np.log10(frame_fft)
        for i in range(10,RANGE_FFT-RANGE_GUARD):
            cut = frame_fft[:,i]
            guard = np.sum(frame_fft[:,i-RANGE_GUARD:i+RANGE_GUARD+1],axis=1)
            guard = (guard - cut) / (2*RANGE_GUARD)
            idx[:,i] = ((cut - guard) > CFAR_THRESHOLD)
    elif METHOD == 'no':
        idx = np.concatenate((np.zeros((AZIM_FFT,6),dtype=bool),np.ones((AZIM_FFT,RANGE_FFT-6),dtype=bool)),axis=1) 
        
    x = x_axis[idx].reshape(-1,1)
    y = y_axis[idx].reshape(-1,1)
    z = np.zeros(x.shape)
    intensity = frame_fft[idx].reshape(-1,1)

    frame_pcl = np.concatenate((x,y,z,intensity),axis=1)

    return frame_pcl

def convert_radar_to_pcl(frame):
    frame_fft = fft_2d(frame, azimuth_tx, AZIM_FFT)
    frame_pcl = threshold(frame_fft)

    return frame_pcl


def get_lidar_pcl_at_timestamps(index, timestamps):
    lf = open('./lidar_pcl/' + str(index) + '_fwd.csv', 'rb')
    lidar_data = pd.read_csv(lf,header=None)
    lidar_time = lidar_data.iloc[:,4].to_numpy()*1e6

    # Some time adjustment
    lidar_time -= 4*3600*1e6

    lidar_image = np.zeros((len(timestamps)-1,RBINS,ABINS_LIDAR))

    for i in range(len(timestamps)-1):
        start_idx = np.argmax(lidar_time >= timestamps[i])
        end_idx = np.argmax(lidar_time >= timestamps[i+1])

        curr_lidar_data = lidar_data.iloc[start_idx:end_idx,0:4].values
        if not(np.unique(lidar_data.iloc[start_idx:end_idx,4].values).shape[0] == 1):
            exit()

        polar_lidar_data = pcl_to_polar(curr_lidar_data)
        lidar_img = create_image_polar(polar_lidar_data, is_lidar=True, is_binary=False)
        lidar_image[i,:,:] = lidar_img

        file_name = 'L_' + str(index) + '_' + str(i) + '.png'
        curr_folder = folder + 'lidar/'
        if not os.path.exists(curr_folder):
            os.makedirs(curr_folder)
        file_name = curr_folder + file_name

        im = Image.fromarray((lidar_img*255).astype(np.uint8))
        im.save(file_name)

    return lidar_image

def get_radar_pcl_at_timestamps(index, timestamps):
    timestamp = np.load('./timestamp_files_radar_lidar/' + str(index) + '_global_start_end_timestamp.npy')

    with open('./radar/' + str(index) + '_read.pkl', 'rb') as f:
        data = pickle.load(f)
        frames = data['frames'].astype(np.complex64)
        frames = frames[:,:3,:,:]

    print('Radar frame rate collected: ', (frames.shape[0]*1e6) / (timestamp[3]-timestamp[2]))

    # find where to trim
    radar_time = np.arange(start=timestamp[2], stop=timestamp[3]+50000, step=50000)
    radar_left_index = np.argmax(radar_time >= (timestamp[0]))
    radar_right_index = np.argmax(radar_time >= (timestamp[1]))
    radar_right_index += 1

    radar_data = frames[radar_left_index:radar_right_index]

    print(radar_data.shape, radar_time.shape)

    radar_image = np.zeros((timestamps.shape[0]-1,RBINS,ABINS_RADAR))

    for i in range(timestamps.shape[0]-1):
        idx = np.argmax(radar_time >= timestamps[i])
        curr_radar_data = convert_radar_to_pcl(radar_data[idx,:])
        polar_radar_data = pcl_to_polar(curr_radar_data)
        radar_img = create_image_polar(polar_radar_data, is_lidar=False, is_binary=False)
        radar_image[i,:,:] = radar_img

        file_name = 'R_' + str(index) + '_' + str(i) + '.png'
        curr_folder = folder + 'radar/'
        if not os.path.exists(curr_folder):
            os.makedirs(curr_folder)
        file_name = curr_folder + file_name

        im = Image.fromarray((radar_img*255).astype(np.uint8))
        im.save(file_name)

    return radar_image

all_radar_files = sorted(glob.glob('./radar/*'))

for curr_file in all_radar_files:

    index = int(curr_file[8:11])
    print(index)

    # Find lidar timestamps that lie within the global start and end times
    timestamps = get_global_timestamps(index)
    print(timestamps.shape)

    # Find the lidar point cloud closest to the required timestamp
    lidar_image = get_lidar_pcl_at_timestamps(index, timestamps)
    print(lidar_image.shape)

    # Find the radar frame closest to the required timestamp
    radar_image = get_radar_pcl_at_timestamps(index, timestamps)
    print(radar_image.shape)

