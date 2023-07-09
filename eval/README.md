# Evaluation

[`eval/`](./eval/) contains scripts for evaluating RadarHD's generated upsampled radar images.

- Executing [`test_radarhd.py`](./test_radarhd.py) will create generated upsampled radar and ground truth lidar images in polar format for all the test data in the corresponding log folder. (Default: [`logs/13_1_20220320-034822/`](./logs/13_1_20220320-034822/))
- Convert polar images to cartesian.

        cd ./eval/
        python3 pol_to_cart.py
    
    The output of this is stored in for example, [`processed_imgs_13_1_20220320-034822_test_imgs/`](./processed_imgs_13_1_20220320-034822_test_imgs/).

- Convert cartesian images to point cloud for point cloud error evaluation.

        python3 image_to_pcd.py

-  Visualize the generated point clouds for qualitative comparison in Matlab.

        pc_vizualize.m

- Generate quantitative point cloud comparison in Matlab (similar to [`eval/cdf.jpg`](./eval/cdf.jpg))

        pc_compare.m