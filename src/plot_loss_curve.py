import numpy as np
import matplotlib.pyplot as plt
import os

loss_path = "models/blood_flow/906b023d-906d-420d-ac47-4eb4a237f3bb/loss.npy"
loss = np.load(loss_path)

plt.figure(figsize=(8, 5))
plt.plot(loss, linewidth=2)
plt.xlabel("Training step")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve (Blood Flow Task)")
plt.grid(True)

save_path = "loss_curve.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[INFO] Saved loss curve to: {os.path.abspath(save_path)}")
