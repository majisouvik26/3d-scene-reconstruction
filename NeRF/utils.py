import torch
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def get_accumulated_transmittance(alphas):
    """
    Computes the accumulated transmittance from alpha values.
    C_hat(r) on page 6 of the paper = cumo_prod of the alphas
    """

    at = torch.cumprod(alphas, dim=1)   # accumulated transmittance
    at_last = at[:, :-1]
    return torch.cat(
        [torch.ones((at.shape[0], 1), device=alphas.device), at_last], dim=-1
    )

def ray_renderer(model, ray_origins, ray_dirxn, hn=0, hf=0.5, num_bins=192):
    """
    Fig 2 of the paper
    Args:
        ray_origings: 3D coordinates of the ray origins
        ray_dirxn: direction vector of the rays
        hn: near plane distance
        hf: far plane distance
        num_bins: number of bins to sample along the ray (since we're summing up and not integrating)
    Returns:
        RGB values and density values for the sampled points along the ray.
    """
    device = ray_origins.device

    # Sample t num of points along the ray
    t = torch.linspace(hn, hf, num_bins, device=device).expand(ray_origins.shape[0], num_bins)

    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    l = torch.cat([t[:, :1], mid], dim=-1) # lower
    u = torch.cat([mid, t[:, -1:]], dim=-1) # upper
    # perturbation sampling
    u_rand = torch.rand(t.shape, device=device)
    t = l + (u - l) * u_rand

    # since we want to do summation instead of integral, calc the deltas
    width = torch.cat(  # bw each of the successive pair of points amongst t
        [
            t[:, 1:] - t[:, :-1],
            torch.tensor(1e9, device=device).expand(ray_origins.shape[0], 1),
        ],
        dim=-1,
    )

    # Sample points along the ray
    points_along_ray = ray_origins.unsqueeze(1) + ray_dirxn.unsqueeze(1) * t.unsqueeze(2)   # o + d*t => [B, num_bins, 3] 
    ray_directions = ray_dirxn.expand(num_bins, ray_dirxn.shape[0], 3).transpose(0, 1)    # [num_bins, B, 3]

    main_shape = points_along_ray.shape
    points_along_ray = points_along_ray.reshape(-1, 3)   # [B*num_bins, 3]
    ray_directions = ray_directions.reshape(-1, 3)   # [B*num_bins, 3]

    colors, sigma = model(points_along_ray, ray_directions)
    # logging.debug(f"Colors: {colors}")
    # logging.debug(f"Sigma: {sigma}")

    colors = colors.reshape(main_shape)
    sigma = sigma.reshape(main_shape[:-1])

    # Compute weighted colors for each ray
    alpha = 1 - torch.exp(-sigma * width)
    weights = get_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # logging.debug(f"Alpha: {alpha}")
    # logging.debug(f"Weights: {weights}")

    weighted_colors = (weights * colors).sum(dim=1)
    sum_weights = weights.sum(dim=-1).sum(dim=-1)  # regularization term for scenes with white bg, else not needed
    sum_weights  = sum_weights.unsqueeze(-1)
    # logging.debug(f"Weighted Colors: {weighted_colors}")
    # logging.debug(f"Sum Weights: {sum_weights}")

    return weighted_colors + 1 - sum_weights