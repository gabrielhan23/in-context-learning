#!/usr/bin/env python3
"""
Test script for the BloodFlow task.
This script demonstrates the blood flow parameter estimation task.
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tasks import BloodFlow

def test_blood_flow():
    """Test the BloodFlow task with visualization."""
    
    print("=" * 60)
    print("Testing Blood Flow Parameter Estimation Task")
    print("=" * 60)
    
    # Task parameters
    n_dims = 2  # time and AIF
    batch_size = 4
    n_points = 100
    
    print("\nTask Configuration:")
    print(f"  Input dimensions: {n_dims} (time, AIF)")
    print(f"  Batch size: {batch_size}")
    print(f"  Time points: {n_points}")
    
    # Create blood flow task
    task = BloodFlow(n_dims=n_dims, batch_size=batch_size)
    
    print("\nTrue Parameters for each batch:")
    print("  Format: [F, vp, ve, PS]")
    for i in range(batch_size):
        params = task.params_b[i].numpy()
        print(f"  Batch {i}: F={params[0]:.3f}, vp={params[1]:.3f}, "
              f"ve={params[2]:.3f}, PS={params[3]:.3f}")
    
    # Generate batch time points
    time_points = torch.linspace(0, 60, n_points)
    time_points_batch = time_points.unsqueeze(0).repeat(batch_size, 1)  # [batch, n_points]

    # Generate AIFs for the whole batch
    aif_batch = task._generate_aif_batch(time_points_batch)  # [batch, n_points]

    # Fill xs_b
    xs_b = torch.zeros(batch_size, n_points, n_dims)
    xs_b[:, :, 0] = time_points_batch
    xs_b[:, :, 1] = aif_batch
    
    # Evaluate the task (generate tissue curves)
    print("\nGenerating tissue concentration curves...")
    ys_b = task.evaluate(xs_b)
    
    print(f"Output shape: {ys_b.shape}")
    print(f"  [batch_size={ys_b.shape[0]}, n_points={ys_b.shape[1]}, features={ys_b.shape[2]}]")
    
    # Visualize results
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    
    for i in range(batch_size):
        ax = axes[i]
        
        # Get data for this batch
        time = xs_b[i, :, 0].numpy()
        aif = xs_b[i, :, 1].numpy()
        tissue = ys_b[i, :, 0].numpy()
        params = task.params_b[i].numpy()
        
        # Plot AIF and tissue curves
        ax2 = ax.twinx()
        
        line1 = ax.plot(time, aif, 'b-', linewidth=2, label='AIF (Arterial Input)', alpha=0.7)
        line2 = ax2.plot(time, tissue, 'r-', linewidth=2, label='Tissue Concentration', alpha=0.7)
        
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('AIF Concentration', color='b', fontsize=10)
        ax2.set_ylabel('Tissue Concentration', color='r', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Title with true parameters
        title = f"Task {i+1}: F={params[0]:.2f}, vp={params[1]:.3f}, ve={params[2]:.2f}, PS={params[3]:.2f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure and show
    output_path = 'blood_flow_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()
    
    # Print statistics
    print("\nTissue Curve Statistics:")
    for i in range(batch_size):
        tissue = ys_b[i, :, 0]
        print(f"  Batch {i}: mean={tissue.mean():.3f}, std={tissue.std():.3f}, "
              f"max={tissue.max():.3f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return task, xs_b, ys_b


def test_parameter_ranges():
    """Test that parameters are within expected physiological ranges."""
    print("\n" + "=" * 60)
    print("Testing Parameter Ranges")
    print("=" * 60)
    
    # Generate many tasks to check distribution
    n_tasks = 1000
    task = BloodFlow(n_dims=2, batch_size=n_tasks)
    
    params = task.params_b.numpy()
    param_names = ['F (Flow)', 'vp (Plasma vol)', 've (Extra vol)', 'PS (Permeability)']
    
    print("\nParameter Statistics (n=1000):")
    for i, name in enumerate(param_names):
        p = params[:, i]
        print(f"  {name}:")
        print(f"    Range: [{p.min():.3f}, {p.max():.3f}]")
        print(f"    Mean: {p.mean():.3f}, Std: {p.std():.3f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run tests
    task, xs_b, ys_b = test_blood_flow()
    test_parameter_ranges()
    
    print("\nBlood Flow task is ready for training!")
    print("This task can be used with: task_name='blood_flow' in config files")
