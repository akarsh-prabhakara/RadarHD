# Create your own dataset

[`create_dataset/`](./create_dataset/) contains scripts that show our pre-ML radar and lidar processing on raw sensor data. Use this only for creating your own radar-lidar images dataset (similar to [`dataset_5`](./dataset_5/)) to train with our models. You can ignore files in this folder if you do not want to create your own dataset.

- First, move your raw radar and lidar data to this [folder](./) in a similar folder structure as [our raw dataset]().
- [`timestamp_check_radar_lidar.py`](./timestamp_check_radar_lidar.py) checks if the radar and lidar timestamps make sense. 

    - We assume that each trajectory was recorded with the lidar capture starting first, radar starting next, and radar finishing first followed by lidar.
    - The output of this is a global common start and end timestamp for both the sensors. This gets stored in [timestamp_files_radar_lidar/]('./timestamp_files_radar_lidar/').
    - Depending on the original timestamps of each sensor, one may need to account for timestamp format, time zone offsets etc.

- [`create_dataset_all_radar_lidar.py`](./create_dataset_all_radar_lidar.py) uses the timestamped raw lidar point cloud and raw radar frames, and the global time start and end computed by [`timestamp_check_radar_lidar.py`](./timestamp_check_radar_lidar.py).

    - Lidar frames are captured at 20 Hz. We use the lidar timestamps as is and find the closest radar timestamp and frame to each lidar frame. 
    - We then associate these two sensor frames and perform processing on each of them. 
    - We convert the lidar point cloud to a high resolution greyscale image that can be a label for training.
    - We perform range-doppler-azimuth processing to the raw radar frame. We then perform a magnitude based thresholding. We then convert to a polar grey-scale image that can be used as input for the network. Our paper discusses this pre-ML processing in detail in Section. 3A.

- We assume that the radar and lidar packets from the sensors have been processed to be in a structure that is similar to [our raw dataset]().