# This file makes sure radar starts and stops before lidar starts and stops
# That is, global start and end time is same as radar start and end 
# This should be run before 'create_dataset_all_radar_lidar.py'

import numpy as np
import dateutil.parser as dp
import sys
import pickle
from datetime import datetime as dt
import pandas as pd
import os
import glob as glob


all_files = sorted(glob.glob('./radar/*'))

for curr_file in all_files:

    ii = int(curr_file[8:11])
    print(ii)

    start_time = 0
    end_time = 0
    with open('./radar/' + str(ii) + '_read.pkl', 'rb') as f:
        data = pickle.load(f)
        start_time = data['start_time']
        end_time = data['end_time']
        num_frames = data['num_frames']

    radar_start_time = start_time
    radar_end_time = end_time

    radar_unix_start_time = dt.timestamp(dt.fromisoformat(radar_start_time[:-1]))*1e6
    radar_unix_end_time = dt.timestamp(dt.fromisoformat(radar_end_time[:-1]))*1e6
    radar_unix_start_time = radar_unix_end_time-(num_frames-1)*0.05*1e6
    
    # Some time adjustment
    radar_unix_start_time -= 4*3600*1e6
    radar_unix_end_time -= 4*3600*1e6

    with open('./lidar_pcl/' + str(ii) + '_fwd.csv', 'rb') as f:
        d=pd.read_csv(f,header=None,usecols=[4]).to_numpy()
        start_time = d[0]
        end_time = d[-1]

    lidar_start_time = start_time*1e6
    lidar_end_time = end_time*1e6

    # Some time adjustment
    lidar_start_time -= 4*3600*1e6
    lidar_end_time -= 4*3600*1e6

    sensor_start_time = np.array([int(radar_unix_start_time), int(lidar_start_time)])
    sensor_end_time = np.array([int(radar_unix_end_time), int(lidar_end_time)])

    print('Start times - radar, lidar')
    print_start = []
    for i in range(len(sensor_start_time)):
        print_start.append(dt.fromtimestamp(sensor_start_time[i]/1e6))
    print(print_start)

    print('End times - radar, lidar')
    print_end = []
    for i in range(len(sensor_end_time)):
        print_end.append(dt.fromtimestamp(sensor_end_time[i]/1e6))
    print(print_end)

    # find start/end time
    global_start_time = np.max(sensor_start_time)
    global_end_time = np.min(sensor_end_time)

    print('Global Start Time ', dt.fromtimestamp(global_start_time/1e6))
    print('Global End Time ', dt.fromtimestamp(global_end_time/1e6))

    curr_folder = './timestamp_files_radar_lidar/'
    if not os.path.exists(curr_folder):
        os.makedirs(curr_folder)

    save_arr = np.array([global_start_time, global_end_time, int(radar_unix_start_time), int(radar_unix_end_time)])
    np.save(curr_folder + str(ii) + '_global_start_end_timestamp', save_arr)
