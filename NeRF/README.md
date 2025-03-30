# Neural Radiance Fields (NeRF) Implementation

This is an implementation of Neural Radiance Fields (NeRF) for 3D scene reconstruction from 2D images. The implementation is based on the original NeRF paper ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934).

## Project Structure

- `model.py`: Core NeRF neural network implementation with positional encoding and MLP architecture
- `utils.py`: Helper functions for ray rendering and transmittance computation
- `main.py`: Training and testing pipeline implementation

## Implementation Details

### Current Features

- Neural network architecture with:
  - Positional encoding for both 3D coordinates and viewing directions
  - MLP with skip connections for density and color prediction
  - Ray rendering functionality
  - Accumulated transmittance computation
- Training pipeline:
  - Supervised training with MSE loss
  - Adam optimizer with learning rate scheduling
  - Batch processing with configurable batch size
  - Concurrent testing during training
- Testing functionality:
  - Chunked inference for memory efficiency
  - Automatic visualization of reconstructed views
  - Progress tracking with tqdm
- Model parameters:
  - Configurable embedding dimensions for position and direction
  - Adjustable hidden layer dimensions
  - Customizable number of sampling bins for ray rendering
  - Adjustable near and far plane distances
  - Configurable image dimensions (H×W)
  - Data loading and preprocessing pipeline

### Planned Features

- [ ] Extended validation metrics
- [ ] Interactive visualization tools
- [ ] Support for custom dataset integration
- [ ] Performance optimizations
- [ ] Comprehensive documentation and usage examples

## Usage

### Training

```python
model = NeRFModel(num_hidden=256)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

train(
    model,
    optimizer,
    scheduler,
    device,
    data_loader,
    hn=0,          # near plane
    hf=0.5,        # far plane
    num_bins=192,  # sampling bins
    H=400,         # image height
    W=400,         # image width
    num_epochs=1000
)
```

### Testing

```python
test(
    test_dataset,
    idx,           # image index
    hn=0,          # near plane
    hf=0.5,        # far plane
    num_bins=192,  # sampling bins
    H=400,         # image height
    W=400          # image width
)
```

## Requirements

- PyTorch
- NumPy
- Matplotlib
- tqdm

## References

1. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.
