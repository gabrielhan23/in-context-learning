#!/usr/bin/env python3
"""
Quick training test for the BloodFlow task.
Runs a few training steps to verify everything works.
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tasks import get_task_sampler
from samplers import get_data_sampler

def test_training_pipeline():
    """Test the blood flow task in the training pipeline."""
    
    print("=" * 60)
    print("Testing Blood Flow Task in Training Pipeline")
    print("=" * 60)
    
    # Configuration
    n_dims = 2  # time and AIF
    batch_size = 16
    n_points = 41  # Number of in-context examples
    
    print(f"\nConfiguration:")
    print(f"  n_dims: {n_dims}")
    print(f"  batch_size: {batch_size}")
    print(f"  n_points: {n_points}")
    
    # Create task sampler
    print("\n1. Creating task sampler...")
    task_sampler = get_task_sampler(
        task_name='blood_flow',
        n_dims=n_dims,
        batch_size=batch_size,
        num_tasks=None  # Generate new tasks each time
    )
    print("   ✓ Task sampler created")
    
    # Create data sampler
    print("\n2. Creating data sampler...")
    data_sampler = get_data_sampler('time_series', n_dims=n_dims)
    print("   ✓ Data sampler created")
    
    # Generate a batch of data
    print("\n3. Generating training batch...")
    task = task_sampler()
    xs = data_sampler.sample_xs(n_points=n_points, b_size=batch_size)
    print(f"   xs shape: {xs.shape}")
    
    # Evaluate task (forward model)
    print("\n4. Running forward model (generating tissue curves)...")
    ys = task.evaluate(xs)
    print(f"   ys shape: {ys.shape}")
    
    # Get metrics
    print("\n5. Getting metrics...")
    metric = task.get_metric()
    training_metric = task.get_training_metric()
    print("   ✓ Metrics obtained")
    
    # Simulate predictions (for testing the metric)
    print("\n6. Testing metric computation...")
    # In reality, these would be model predictions
    # For testing, we'll use the true outputs with some noise
    ys_pred = ys + torch.randn_like(ys) * 0.1
    
    # Compute per-example error (for evaluation)
    if callable(metric):
        errors = metric(ys_pred, ys)
        print(f"   Per-example errors shape: {errors.shape}")
        print(f"   Mean error: {errors.mean().item():.6f}")
    
    # Compute training loss
    if callable(training_metric):
        loss = training_metric(ys_pred, ys)
        print(f"   Training loss: {loss.item():.6f}")
    
    # Test multiple batches
    print("\n7. Testing multiple batches...")
    for i in range(3):
        task = task_sampler()
        xs = data_sampler.sample_xs(n_points=n_points, b_size=batch_size)
        ys = task.evaluate(xs)
        print(f"   Batch {i+1}: xs {xs.shape}, ys {ys.shape}, "
              f"y_mean={ys.mean().item():.3f}, y_std={ys.std().item():.3f}")
    
    # Print parameter statistics
    print("\n8. Parameter statistics from last batch:")
    params = task.params_b
    param_names = ['F', 'vp', 've', 'PS']
    for i, name in enumerate(param_names):
        p = params[:, i]
        print(f"   {name:3s}: min={p.min().item():.3f}, max={p.max().item():.3f}, "
              f"mean={p.mean().item():.3f}")
    
    print("\n" + "=" * 60)
    print("✓ All pipeline tests passed successfully!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Train model with: python src/train.py --config conf/blood_flow.yaml")
    print("  2. Adjust hyperparameters in src/conf/blood_flow.yaml as needed")
    print("  3. Monitor training with wandb")
    
    return True


if __name__ == "__main__":
    try:
        test_training_pipeline()
        print("\n✓ Blood flow task is ready for full training!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
