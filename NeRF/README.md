# Neural Radiance Fields (NeRF) Implementation

This is an implementation of Neural Radiance Fields (NeRF) for 3D scene reconstruction from 2D images. The implementation is based on the original NeRF paper ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934).

## Project Structure

- `model.py`: Core NeRF neural network implementation with positional encoding and MLP architecture
- `utils.py`: Helper functions for ray rendering and transmittance computation
- `main.py`: Training and testing pipeline implementation
- `convert_to_mp4.py`: Utility to convert rendered images to MP4 video
- `test_model.py`: Script for testing the trained model

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd NeRF
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The implementation expects data in a specific format. The training and testing data should be provided as pickle files:

- `training_data.pkl`: Contains the training data
- `testing_data.pkl`: Contains the testing data

The data should be in the format of a numpy array with the following structure:

- Each row represents a ray
- Columns 0-2: Ray origin (x, y, z)
- Columns 3-5: Ray direction (normalized)
- Columns 6-8: RGB pixel values

## Training

To train the NeRF model, run:

```bash
python main.py
```

The training script will:

1. Load the training and testing data
2. Initialize the NeRF model
3. Train the model for the specified number of epochs
4. Save checkpoints at regular intervals
5. Generate test images during training
6. Plot the loss curve

### Training Parameters

You can modify the following parameters in `main.py`:

- `hidden_dim_pos`: Dimension of the positional encoding (default: 384)
- `hidden_dim_dir`: Dimension of the directional encoding (default: 128)
- `hn`: Near plane distance (default: 2.0)
- `hf`: Far plane distance (default: 6.0)
- `num_bins`: Number of sampling bins for ray rendering (default: 192)
- `num_epochs`: Number of training epochs (default: 20)
- `batch_size`: Batch size for training (default: 1024)
- `learning_rate`: Learning rate for the optimizer (default: 3e-4)
- `checkpoint_frequency`: Frequency of checkpoint saving (default: 5)

## Testing

After training, testing will start automatically

The test method will:

1. Load the trained model
2. Generate novel views from the testing dataset
3. Save the rendered images to the output directory

## Visualization

To visualize the results, you can use the `convert_to_mp4_gif.py` script to convert the rendered images to a GIF animation:

```bash
python convert_to_mp4_gif.py
```

This will create a GIF animation named `truck_reconstruction.gif` in the `output/` directory. The animation will show a smooth interpolation between different viewpoints of the reconstructed 3D scene.

### Result Animation

![NeRF Reconstruction Animation](output/truck_reconstruction.gif)

Animation specifications:
- Format: GIF
- Frame Rate: 10 FPS (100ms per frame)
- Resolution: Matches the training images (default: 400x400)
- Location: `output/truck_reconstruction.gif`

## Implementation Details

### Neural Network Architecture

- Positional encoding for both 3D coordinates and viewing directions
- MLP with skip connections for density and color prediction
- Ray rendering functionality
- Accumulated transmittance computation

### Training Pipeline

- Supervised training with MSE loss
- Adam optimizer with learning rate scheduling
- Batch processing with configurable batch size
- Checkpoint saving and loading
- Progress tracking with tqdm

### Testing Functionality

- Chunked inference for memory efficiency
- Automatic visualization of reconstructed views
- Progress tracking with tqdm

### Model Parameters

- Configurable embedding dimensions for position and direction
- Adjustable hidden layer dimensions
- Customizable number of sampling bins for ray rendering
- Adjustable near and far plane distances
- Configurable image dimensions (H×W)

## Output

The training and testing process will generate the following outputs:

- `checkpoints/`: Directory containing model checkpoints
  - `checkpoint_epoch_X.pth`: Checkpoint saved at epoch X
  - `latest_checkpoint.pth`: Latest checkpoint
- `output/`: Directory containing output files
  - `reconstructed_views_truck/`: Directory containing rendered images
  - `loss_curves/`: Directory containing loss curve plots

## References

1. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.
2. [Original NeRF Implementation](https://github.com/bmild/nerf)

## License and Contributing

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contributing

We welcome contributions to this project! If you'd like to contribute:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Submit a pull request

We're also open to issues! If you find a bug or have a feature request, please open an issue on the GitHub repository.
