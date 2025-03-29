import torch
import torch.nn as nn

def compute_accumulated_transmittance(alphas):
    """
    Computes the accumulated transmittance from alpha values.
    C_hat(r) on page 6 of the paper = cumo_prod of the alphas 
    """

    accumulated_transmittance = torch.cumprod(alphas, dim=1)
    first_few_acc_transmittance = accumulated_transmittance[:, :-1]
    return torch.cat(
        [torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device), first_few_acc_transmittance], dim=-1
    )

def ray_renderer(NeRFModel, ray_origings, ray_dirxn, hn=0, hf=0.5, num_bins=192):
    """
    Args:
        ray_origings: 3D coordinates of the ray origins
        ray_dirxn: direction vector of the rays
        hn: near plane distance
        hf: far plane distance
        num_bins: number of bins to sample along the ray
    Returns:
        RGB values and density values for the sampled points along the ray.
    """
    pass    