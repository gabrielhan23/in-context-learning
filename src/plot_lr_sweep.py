import numpy as np
import matplotlib.pyplot as plt

curves = np.load("lr_sweep_curves.npy", allow_pickle=True).item()

plt.figure(figsize=(8, 5))

for lr, loss_curve in curves.items():
    plt.plot(loss_curve[:1000], label=f"LR = {lr}", linewidth=2)

plt.xlabel("Training Step")
plt.ylabel("MSE Loss")
plt.title("Learning Rate Sweep (1k steps)")
plt.grid(True)
plt.legend()

plt.savefig("lr_sweep.png", dpi=300, bbox_inches="tight")
plt.close()

print("[INFO] Saved LR sweep plot → lr_sweep.png")
