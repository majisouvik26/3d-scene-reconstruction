# Neural Radiance Fields (NeRF) Implementation

This is a simplistic implementation of Neural Radiance Fields (NeRF) for 3D scene reconstruction from 2D images. The implementation is based on the original NeRF paper ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934).

## Project Structure

- `model.py`: Contains the core NeRF neural network implementation

- `utils.py`: Helper functions for ray rendering and transmittance computation

## Implementation Details

### Current Features- Neural network architecture with:

- Positional encoding for both 3D coordinates and viewing directions - MLP with skip connections for density and color prediction
- Ray rendering functionality - Accumulated transmittance computation
- Model parameters: - Configurable embedding dimensions for position and direction
  - Adjustable hidden layer dimensions - Customizable number of sampling bins for ray rendering

### Planned Features

- [ ] Training pipeline implementation- [ ] Data loading and preprocessing
- [ ] Validation and testing scripts- [ ] Visualization tools for rendered results
- [ ] Support for custom dataset integration- [ ] Performance optimizations
- [ ] Documentation and usage examples

## Usage (To be added as features are implemented)

## Requirements

- PyTorch- NumPy
- Matplotlib- tqdm

## References

1. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.
