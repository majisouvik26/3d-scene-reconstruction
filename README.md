# 3D Scene Reconstruction

This repository contains our final project for the Computer Vision course, focusing on various techniques and approaches for **3D Scene Reconstruction**.

## Project Structure

The repository structure is organized as follows:

```
3d-scene-reconstruction/
├── exps/
│   ├── Gaussian_Splatting/
│   │   ├── semantic_gs/
│   │   ├── text_2_3d/
│   │   └── webgl_splat_viewer/
│   ├── Incremental-SfM/
│   ├── Multiview-3D-Reconstruction-SFM/
│   ├── NeRF/
│   ├── Pix2Vox/
│   └── SpaceCarving/
├── .gitignore
├── LICENSE
└── README.md
```

## Techniques Implemented

### 1. Gaussian Splatting
Implementation of 3D reconstruction using Gaussian Splatting methods.

#### Semantic Gaussian Splatting
Semantic-driven Gaussian splatting method for more contextual 3D reconstruction.

#### Text to 3D
Generating 3D structures directly from textual descriptions.

#### WebGL Splat Viewer
Interactive web-based viewer for visualizing Gaussian Splat reconstructions.

### 2. Incremental Structure-from-Motion (SfM)
Reconstruction approach using Incremental SfM.

### 3. Multiview 3D Reconstruction with SfM
Uses Structure-from-Motion for reconstructing 3D scenes from multiple views.

### 4. Neural Radiance Fields (NeRF)
Implementation of NeRF techniques for detailed scene reconstruction.

### 5. Pix2Vox
Exploration of reconstruction using Pix2Vox methodology.

### 6. Space Carving
3D shape reconstruction using the Space Carving technique.

## Usage

Each directory under `exps` contains specific instructions and code relevant to the respective reconstruction method.

## License
This project is licensed under the MIT License.

