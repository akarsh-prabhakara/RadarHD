#  File for testing RadarHD

import time
import os
import datetime
import json

import torch

import numpy as np
from torchsummary import summary

from train_test_utils.dataloader import *
from train_test_utils.model import *

"""
## Constants. Edit this to change the model to test on.
"""

params = {
    'model_name': '13',
    'expt': 1,
    'dt': '20230707-152337',
    'epoch_num': 120,
    'data': 5,
    'gpu': 1,
}

def dataloader(train_params):
    print('Loading data')
    basepath = './dataset_' + str(params['data']) + '/'

    orig_size = [256, 64, 512]
    reqd_size = [256, 64, 512]

    test_set = Dataset(basepath, 'test',
                        RBINS=reqd_size[0], ABINS_RADAR=reqd_size[1], ABINS_LIDAR=reqd_size[2],
                        RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1], ABINS_LIDAR_ORIG=orig_size[2], 
                        M=train_params['history'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    ordered_filename = test_set.__filenames__()
    print('# of points to test: ', len(test_loader))
    return (test_loader, ordered_filename)

def main():
    print(torch.__version__)
    torch.manual_seed(0)  

    # Can be set to cuda/cpu. Make sure model and data are moved to cuda if cuda is used
    if params['gpu'] == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    name_str = params['model_name'] + '_' + str(params['expt']) + '_' + params['dt']
    LOG_DIR = './logs/' + name_str + '/'
    with open(os.path.join(LOG_DIR, 'params.json'), 'r') as f:
        train_params = json.load(f)

    # Load data
    (test_loader, ordered_filename) = dataloader(train_params)

    # Define model
    gen = UNet1(train_params['history']+1, 1).to(device)
    summary(gen, (train_params['history']+1, 256, 64))

    epoch_num = '%03d' % params['epoch_num']
    model_file = LOG_DIR + epoch_num + '.pt_gen'
    checkpoint = torch.load(model_file, map_location=device)
    gen.load_state_dict(checkpoint['state_dict'])

    save_path = './logs/' + name_str + '/test_imgs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Testing
    gen.eval()

    t0 = time.time()
    for test_i, (test_data, test_label) in enumerate(test_loader):

        test_data, test_label = test_data.to(device), test_label.to(device)
        with torch.no_grad():
            pred = gen(test_data)
            
            pred = np.squeeze(pred.cpu().numpy())
            pred = (pred*255).astype(np.uint8)
            im1 = Image.fromarray(pred)

            im1_file_name = save_path + epoch_num + '_' + ordered_filename[test_i] + '_pred.png'
            im1.save(im1_file_name)
            
            label = np.squeeze(test_label.cpu().numpy())
            label = (label*255).astype(np.uint8)
            im1 = Image.fromarray(label)
            im1_file_name = save_path + epoch_num + '_' + ordered_filename[test_i] + '_label.png'
            im1.save(im1_file_name)
            
            print(ordered_filename[test_i])

    t1 = time.time()
    print('Time taken for inference: ' ,t1 - t0)

main()