import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import os

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from tasks import BloodFlow

def generate_and_save_dataset(
    n_samples=10000,
    n_points=100,
    batch_size=64,
    test_fraction=0.1,
    save_path="blood_flow_dataset.pt",
    device=None
):
    device = device or torch.device("cpu")
    print(f"Using device {device}")
    torch.manual_seed(48958479)

    n_dims = 2
    task = BloodFlow(n_dims=n_dims, batch_size=n_samples, device=device)

    all_params = torch.zeros(n_samples, task.params_b.shape[1])
    all_aif = torch.zeros(n_samples, n_points)
    all_tissue = torch.zeros(n_samples, n_points)

    time_points = torch.linspace(0, 60, n_points, device=device)
    
    pbar = tqdm(total=n_samples, desc="Generating dataset")
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        current_batch = end - start

        batch_params = task.params_b[start:end].to(device)
        aif_batch = task._generate_aif_batch(time_points.repeat(current_batch, 1))
        tissue_batch = task._simulate_batch(
            time_points.repeat(current_batch, 1), aif_batch, batch_params
        )

        # add realistic MRI Gaussian noise (~3% of mean tissue signal)
        sigma = 0.03 * tissue_batch.mean(dim=1, keepdim=True)
        noise = torch.randn_like(tissue_batch) * sigma
        tissue_batch_noisy = tissue_batch + noise

        all_params[start:end] = batch_params.cpu()
        all_aif[start:end] = aif_batch.cpu()
        all_tissue[start:end] = tissue_batch_noisy.cpu()

        pbar.update(current_batch)

    pbar.close()

    # split dataset into train and test portions
    params_train, params_test, aif_train, aif_test, tissue_train, tissue_test = train_test_split(
        all_params, all_aif, all_tissue, test_size=test_fraction, random_state=79532948
    )

    dataset = {
        "train": {"params": params_train, "aif": aif_train, "tissue": tissue_train},
        "test": {"params": params_test, "aif": aif_test, "tissue": tissue_test}
    }

    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Train size: {len(params_train)}, Test size: {len(params_test)}")

    return dataset

if __name__ == "__main__":
    dataset = generate_and_save_dataset(
        n_samples=1_000_000,
        n_points=100,
        batch_size=64,
        test_fraction=0.1,
        save_path="blood_flow_dataset.pt",
        device="cpu"
    )

data = torch.load("blood_flow_dataset.pt")
all_params = torch.cat([data["train"]["params"], data["test"]["params"]], dim=0)
all_aif = torch.cat([data["train"]["aif"], data["test"]["aif"]], dim=0)
all_tissue = torch.cat([data["train"]["tissue"], data["test"]["tissue"]], dim=0)


# check the first four samples of the newly created dataset
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

for i in range(4):
    ax = axes[i]
    
    # Get data for this batch
    time = torch.linspace(0, 60, all_aif.shape[1]).numpy()
    aif = all_aif[i].numpy()
    tissue = all_tissue[i].numpy()
    params = all_params[i].numpy()
    
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
plt.show()