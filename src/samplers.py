import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "time_series": TimeSeriesSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class TimeSeriesSampler(DataSampler):
    """
    Sampler for time series data with variable spacing and number of points.
    
    Generates time points for time series tasks like predator-prey dynamics.
    Each batch can have different time spacing patterns.
    """
    def __init__(self, n_dims, min_spacing=0.01, max_spacing=0.1):
        super().__init__(n_dims)
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        Generate time series data points.
        
        Args:
            n_points: number of time points to generate
            b_size: batch size
            n_dims_truncated: not used for time series, kept for compatibility
            seeds: random seeds for reproducibility
            
        Returns:
            xs_b: tensor of shape [b_size, n_points, n_dims]
                  First dimension contains time points, rest are zeros
        """
        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        
        if seeds is None:
            for b in range(b_size):
                # Random spacing between min and max for each batch
                spacing = torch.rand(1).item() * (self.max_spacing - self.min_spacing) + self.min_spacing
                # Generate linearly spaced time points
                time_points = torch.linspace(0, spacing * (n_points - 1), n_points)
                xs_b[b, :, 0] = time_points
        else:
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                # Random spacing between min and max
                spacing = torch.rand(1, generator=generator).item() * (self.max_spacing - self.min_spacing) + self.min_spacing
                # Generate linearly spaced time points
                time_points = torch.linspace(0, spacing * (n_points - 1), n_points)
                xs_b[i, :, 0] = time_points
        
        return xs_b
