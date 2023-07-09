# Creates a dataloader in batches for training and testing

import os
from PIL import Image
import numpy as np
import torch
import glob

class Dataset(torch.utils.data.Dataset):

    def __init__(self, basepath, sub,
                RBINS=256, ABINS_RADAR=64, ABINS_LIDAR=512,
                RBINS_ORIG=256, ABINS_RADAR_ORIG=64, ABINS_LIDAR_ORIG=1024, M=0):

        self.basepath = basepath
        self.lidar_path = self.basepath + sub + '/lidar/*'
        self.radar_path = self.basepath + sub + '/radar/*'

        self.RBINS = RBINS
        self.ABINS_RADAR = ABINS_RADAR
        self.ABINS_LIDAR = ABINS_LIDAR
        self.RBINS_ORIG = RBINS_ORIG
        self.ABINS_RADAR_ORIG = ABINS_RADAR_ORIG
        self.ABINS_LIDAR_ORIG = ABINS_LIDAR_ORIG
        self.history = M

        lidar_files = sorted(glob.glob(self.lidar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        radar_files = sorted(glob.glob(self.radar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        
        if self.history == 0:
            self.labels = lidar_files
            self.input_data = radar_files
        else:
            traj = [int(os.path.basename(x).split('_')[1]) for x in lidar_files]
            time_st = [int(os.path.basename(x).split('_')[2].split('.')[0]) for x in lidar_files]
            self.labels = []
            self.input_data = []

            for i in np.unique(traj):
                start_idx = np.where(traj==i)[0][0]
                end_idx = np.where(traj==i)[0][-1]+1
                print("Traj ", i, "Time ", time_st[start_idx], " ", time_st[end_idx-1])
                radar_files_time = radar_files[start_idx:end_idx]
                lidar_files_time = lidar_files[start_idx:end_idx]

                x_local = []
                for j in range(self.history, len(radar_files_time)):
                    x_local.append(radar_files_time[j-self.history:j+1])
                y_local = lidar_files_time[self.history:]
                
                self.labels.extend(y_local)
                self.input_data.extend(x_local)

    def __len__(self):
        return len(self.input_data)

    def __filenames__(self):
        return [os.path.basename(x).split('.')[0].split('L_')[1] for x in self.labels]

    def get_lidar(self, label_filename):

        a = Image.open(label_filename)
        y = torch.Tensor(np.reshape(np.asarray(a,dtype=np.bool_), (1,self.RBINS_ORIG,self.ABINS_LIDAR_ORIG)))
        y = y[:,0::int(self.RBINS_ORIG/self.RBINS),0::int(self.ABINS_LIDAR_ORIG/self.ABINS_LIDAR)]

        return y

    def get_radar(self, input_filename):

        a = Image.open(input_filename)
        X = torch.Tensor(np.reshape(np.asarray(a)/255.0, (1,self.RBINS_ORIG,self.ABINS_RADAR_ORIG)))
        X = X[:,0::int(self.RBINS_ORIG/self.RBINS),0::int(self.ABINS_RADAR_ORIG/self.ABINS_RADAR)]

        return X

    def __getitem__(self, index):

        # Select sample
        if self.history == 0:
            input_filename = self.input_data[index]
            label_filename = self.labels[index]
            X, y = self.get_radar(input_filename), self.get_lidar(label_filename)
        
        else:
            X = torch.Tensor([])
            input_filenames = self.input_data[index]
            label_filename = self.labels[index]
            for i in input_filenames:
                xx = self.get_radar(i)
                X = torch.cat((X, xx), dim=0)        
            y = self.get_lidar(label_filename)

        return X, y
