import math

import torch
import numpy as np


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "predator_prey": PredatorPrey,
        "blood_flow": BloodFlow,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(self.dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class PredatorPrey(Task):
    """
    Predator-Prey parameter estimation task using Lotka-Volterra equations.
    
    The task is to estimate parameters [alpha, beta, gamma, delta] from time series observations.
    
    Lotka-Volterra equations:
    dx/dt = alpha*x - beta*x*y    (prey growth - predation)
    dy/dt = delta*x*y - gamma*y   (predator growth - death)
    
    where x is prey population, y is predator population, and
    alpha, beta, gamma, delta are positive parameters.
    
    Input (xs): Time points [batch, n_points, n_dims] where first dim is time
    Output (ys): Population trajectories [batch, n_points, 2] for [prey, predator]
    Target: Parameters [alpha, beta, gamma, delta] at last position only
    """
    
    def __init__(
        self, 
        n_dims, 
        batch_size, 
        pool_dict=None, 
        seeds=None, 
        scale=1,
        param_range=(0.5, 2.0)
    ):
        """
        Args:
            n_dims: dimension of input time points (typically 1)
            batch_size: number of tasks in batch
            pool_dict: pregenerated parameter pool
            seeds: random seeds for reproducibility
            scale: scaling factor for outputs
            param_range: range for randomly sampling parameters (min, max)
        """
        super(PredatorPrey, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.param_range = param_range
        
        # Generate true parameters for each batch: [alpha, beta, gamma, delta]
        if pool_dict is None and seeds is None:
            param_min, param_max = param_range
            self.params_b = torch.rand(self.b_size, 4) * (param_max - param_min) + param_min
        elif seeds is not None:
            self.params_b = torch.zeros(self.b_size, 4)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            param_min, param_max = param_range
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.params_b[i] = torch.rand(4, generator=generator) * (param_max - param_min) + param_min
        else:
            assert "params" in pool_dict
            indices = torch.randperm(len(pool_dict["params"]))[:batch_size]
            self.params_b = pool_dict["params"][indices]
    
    def _simulate_lotka_volterra(self, params, time_points, device):
        """
        Simulate Lotka-Volterra equations using Euler method.
        
        Args:
            params: [alpha, beta, gamma, delta] parameters (tensor)
            time_points: [n_points] tensor of time values
            device: torch device
            
        Returns:
            trajectory: [n_points, 2] tensor of [prey, predator] populations
        """
        alpha, beta, gamma, delta = params
        n_points = len(time_points)
        
        # Initial conditions
        x = torch.tensor(10.0, device=device)  # initial prey population
        y = torch.tensor(5.0, device=device)   # initial predator population
        
        trajectory = torch.zeros(n_points, 2, device=device)
        trajectory[0, 0] = x
        trajectory[0, 1] = y
        
        for i in range(1, n_points):
            # Compute dt from time points
            dt = time_points[i] - time_points[i-1]
            
            # Lotka-Volterra equations
            dx_dt = alpha * x - beta * x * y
            dy_dt = delta * x * y - gamma * y
            
            # Euler integration
            x = x + dx_dt * dt
            y = y + dy_dt * dt
            
            # Prevent populations from going negative or exploding
            x = torch.clamp(x, 0.1, 1000.0)
            y = torch.clamp(y, 0.1, 1000.0)
            
            trajectory[i, 0] = x
            trajectory[i, 1] = y
        
        return trajectory

    def evaluate(self, xs_b):
        """
        Generate population trajectories from time points using Lotka-Volterra equations.
        
        Args:
            xs_b: Input time points of shape (batch_size, n_points, n_dims)
                  First dimension contains time values
            
        Returns:
            ys_b: Population trajectories of shape (batch_size, n_points, 2)
                  Contains [prey, predator] populations at each time point
        """
        batch_size = xs_b.shape[0]
        n_points = xs_b.shape[1]
        device = xs_b.device
        
        params_b = self.params_b.to(device)
        
        # Generate trajectories - ys_b will contain [prey, predator] populations
        ys_b = torch.zeros(batch_size, n_points, 2, device=device)
        
        for b in range(batch_size):
            # Extract time points from first dimension of xs_b
            time_points = xs_b[b, :, 0]
            
            # Simulate trajectory using the ODE with true parameters
            trajectory = self._simulate_lotka_volterra(params_b[b], time_points, device)
            
            # Store trajectory as output
            ys_b[b] = trajectory
        
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, param_range=(0.5, 2.0), **kwargs):
        """Generate a pool of predator-prey parameter estimation tasks."""
        param_min, param_max = param_range
        return {
            "params": torch.rand(num_tasks, 4) * (param_max - param_min) + param_min
        }

    def get_metric(self):
        """Return metric that evaluates parameter estimation at the last position."""
        params_b = self.params_b  # Capture true parameters
        
        def param_estimation_error(ys_pred, ys):
            # ys_pred: [batch, n_points, 4] - predicted parameters at each position
            # ys: [batch, n_points, 2] - observed populations (not used for loss)
            
            # Extract predictions at last position
            last_pred = ys_pred[:, -1, :]  # [batch, 4]
            
            # Compare to true parameters
            true_params = params_b.to(ys_pred.device)
            error = (last_pred - true_params).square()  # [batch, 4]
            
            return error.mean(dim=1)  # [batch] - mean error across 4 parameters
        
        return param_estimation_error

    def get_training_metric(self):
        """Return training metric for parameter estimation."""
        params_b = self.params_b  # Capture true parameters
        
        def param_estimation_loss(ys_pred, ys):
            # ys_pred: [batch, n_points, 4] - predicted parameters at each position
            # ys: [batch, n_points, 2] - observed populations
            
            # Only use the last position for loss
            last_pred = ys_pred[:, -1, :]  # [batch, 4]
            
            # Compare to true parameters
            true_params = params_b.to(ys_pred.device)
            loss = ((last_pred - true_params).square()).mean()
            
            return loss
        
        return param_estimation_loss

class BloodFlow:
    def __init__(self, n_dims=2, batch_size=4, param_ranges=None,
                 rk4_substeps=8, device=None, scale=1.0):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.rk4_substeps = rk4_substeps
        self.scale = scale
        self.device = device or torch.device('cpu')

        # physiologically plausible ranges for parameters
        if param_ranges is None:
            param_ranges = {
                'F': (0.5, 4.0),    # Flow in ml/g/min
                'vp': (0.02, 0.15), # Plasma volume fraction
                've': (0.18, 0.5),  # Extracellular volume fraction
                'PS': (0.1, 1.0),   # Permeability in ml/g/min
                'tau0': (0.2, 5.0)  # Transit delay (s)
            }
        self.param_ranges = param_ranges

        # generate random parameters for batch: [F, vp, ve, PS, tau0]
        self.params_b = torch.zeros(self.b_size, 5, device=self.device)
        for i, (_, (pmin, pmax)) in enumerate(param_ranges.items()):
            self.params_b[:, i] = torch.rand(self.b_size, device=self.device) * (pmax - pmin) + pmin

    def _generate_aif_batch(self, time_points):
        batch, _ = time_points.shape
        A = torch.rand(batch, 1, device=self.device) * 3.0 + 2.0
        alpha = torch.rand(batch, 1, device=self.device) * 2.0 + 2.0
        beta = torch.rand(batch, 1, device=self.device) * 2.0 + 1.0
        t0 = torch.rand(batch, 1, device=self.device) * 2.0 + 1.0
        t_shifted = (time_points - t0).clamp(min=0.0)
        return A * (t_shifted ** alpha) * torch.exp(-t_shifted / beta)

    def _apply_delay(self, time_points, aif, tau0):
        batch, _ = time_points.shape
        aif_delayed = torch.zeros_like(aif)
        for b in range(batch):
            t_shifted = (time_points[b] - tau0[b]).clamp(min=0.0)
            # linear interpolation over original AIF
            aif_delayed[b] = torch.from_numpy(
                np.interp(t_shifted.cpu().numpy(),
                          time_points[b].cpu().numpy(),
                          aif[b].cpu().numpy())).to(self.device)
        return aif_delayed

    def _rk4_step(self, Cp, Ce, Ca, dt, Fv, vp, ve, PS):
        
        # ODE derivatives
        def dCp(Cp, Ce, Ca): 
            return Fv*(Ca - Cp/vp) + PS*(Ce - Cp)
        def dCe(Cp, Ce): 
            return PS*(Cp - Ce)

        k1_Cp = dCp(Cp, Ce, Ca) * dt
        k1_Ce = dCe(Cp, Ce) * dt
        k2_Cp = dCp(Cp + 0.5 * k1_Cp, Ce + 0.5 * k1_Ce, Ca) * dt
        k2_Ce = dCe(Cp + 0.5 * k1_Cp, Ce + 0.5 * k1_Ce) * dt
        k3_Cp = dCp(Cp + 0.5 * k2_Cp, Ce + 0.5 * k2_Ce, Ca) * dt
        k3_Ce = dCe(Cp + 0.5 * k2_Cp, Ce + 0.5 * k2_Ce) * dt
        k4_Cp = dCp(Cp + k3_Cp, Ce + k3_Ce, Ca) * dt
        k4_Ce = dCe(Cp + k3_Cp, Ce + k3_Ce) * dt

        Cp_new = Cp + (k1_Cp + 2 * k2_Cp + 2 * k3_Cp + k4_Cp) / 6
        Ce_new = Ce + (k1_Ce + 2 * k2_Ce + 2 * k3_Ce + k4_Ce) / 6

        Cp_new = Cp_new.clamp(min=0.0)
        Ce_new = Ce_new.clamp(min=0.0)
        return Cp_new, Ce_new

    def _simulate_batch(self, time_points, aif_values, params):
        batch, n_points = time_points.shape
        tissue = torch.zeros(batch, n_points, device=self.device)
        Cp = torch.zeros(batch, device=self.device)
        Ce = torch.zeros(batch, device=self.device)

        Fv, vp, ve, PS, tau0 = params[:,0], params[:,1], params[:,2], params[:,3], params[:,4]

        # apply tau0 delay
        aif_delayed = self._apply_delay(time_points, aif_values, tau0)

        for i in range(n_points):
            tissue[:, i] = Cp * vp + Ce * ve
            if i < n_points - 1:
                dt = (time_points[:, i + 1] - time_points[:, i]).clamp(min=1e-6)
                dt_sub = dt / self.rk4_substeps
                for _ in range(self.rk4_substeps):
                    Cp, Ce = self._rk4_step(Cp, Ce, aif_delayed[:, i], dt_sub, Fv, vp, ve, PS)
        return tissue

    def evaluate(self, xs_b):
        _, _, n_dims = xs_b.shape
        time_points = xs_b[:, :, 0]
        aif_values = xs_b[:, :, 1] if n_dims >= 2 else self._generate_aif_batch(time_points)

        tissue = self._simulate_batch(time_points, aif_values, self.params_b)

        # add independent Gaussian measurement noise
        sigma = 0.03 * tissue.mean(dim=1, keepdim=True) # 3% of mean signal
        noise = torch.randn_like(tissue) * sigma

        return (tissue + noise).unsqueeze(-1) * self.scale
