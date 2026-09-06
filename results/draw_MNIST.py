import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_DIR = os.path.join(BASE_DIR, "h5")

files = {
    # "FedDyn": os.path.join(H5_DIR, "MNIST_FedDyn.h5"),
    # "FedAvg": os.path.join(H5_DIR, "MNIST_FedAvg.h5"),
    # "FedProx": os.path.join(H5_DIR, "MNIST_FedProx.h5"),
    # "MOON": os.path.join(H5_DIR, "MNIST_MOON.h5"), 
    
     "1.0": os.path.join(H5_DIR, "MNIST_1.0.h5"),
     "0.5": os.path.join(H5_DIR, "MNIST_0.5.h5"),
     "0.2": os.path.join(H5_DIR, "MNIST_0.2.h5"),
     "0.1": os.path.join(H5_DIR, "MNIST_0.1.h5"),
}

plt.figure(figsize=(7, 5))

for name, path in files.items():
    with h5py.File(path, "r") as f:
        acc = np.array(f["rs_test_acc"])
        
        max_round = 201
        acc = acc[:max_round]
        
        rounds = np.arange(len(acc))
        plt.plot(rounds, acc, linewidth=2, label=name)

plt.xlabel("Global Rounds")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy on Non-IID MNIST")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

