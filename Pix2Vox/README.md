# Pix2Vox
This is the implementation of Pix2Vox architecture used in 3d reconstruction of objects from 2d multi view images.This implementation is based on the paper ["Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images"](https://arxiv.org/abs/1901.11153)

## Dataset used

Used the [ShapeNet](https://www.shapenet.org/) which is available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

## Directory Structure
  ├── dataset                            # Contains ShapeNet data and configuration
    |   ├── ShapeNetRendering
    |   ├── ShapeNetVox32
    |   └── ShapeNet.json
    ├── models                           # Network architecture modules
    │   ├── decoder.py
    │   ├── discriminator.py
    │   ├── encoder.py
    │   ├── decoder.py
    │   ├── merger.py
    |   └── refiner.py
    ├── saved_model                      # Stores trained model checkpoints
    ├── src                              # Training and testing scripts
    │   ├── test.py
    |   └── train.py
    ├── utils                            #Utility scripts
    |    ├── binvox-rw.py
    |    ├── binvox-visualization.py
    |    ├── data_loaders.py
    |    ├── dataset_analyzer.py
    |    └── network_utils.py
    ├── app.py                          # Streamlit frontend to visualize model predictions
    │── config.py                       # Configuration settings for training/testing
    ├── helpers.py                      # Common helper functions used by `app.py
    ├── README.md
    ├── requirements.txt
    └── run.py                          #to run training/testing interactively

## Download the trained model
The trained model can be downloaded from the following link:https://drive.google.com/file/d/1bBA3_jC7BSLm0QXtc1NQv4spYVvDWXPB/view?usp=sharing


## Commands

To train Pix2Vox, you can simply use the following command:

```
python run.py
```

For testing Pix2Vox on test_dataset, you can use the following command:

```
python run.py --test --weights=/path/to/pretrained/model.pth
```

For testing Pix2Vox for constructing a single model from it multiview images

```
python run.py --test --weights=/path/to/pretrained/model.pth --img_dir=/path/to/where/multiview/images/are --n_views=3
```

specifying n_views is optional as if not specified all images in the directory will be used
