# Train Test Utils

[`train_test_utils/`](./train_test_utils/) contains model, loss and dataloading definitions for training and testing. None of these files need to be executed independently. [`train_radarhd.py`](../train_radarhd.py) and [`test_radarhd.py`](../test_radarhd.py) invokes these definitions.

- [`unet_parts.py`](./unet_parts.py) defines the basic core blocks of the U-Net
- [`model.py`](./model.py) uses the U-Net parts to create the RadarHD model
- [`dice_score.py`](./dice_score.py) defines the dice loss for training
- [`dataloader.py`](./dataloader.py) defines how the radar-lidar dataset images (Example: [`dataset_5`](../dataset_5/)) are fed into for testing and training.