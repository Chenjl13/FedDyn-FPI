# Method and Implementation Notes

This repository implements a communication compression method for FedDyn under Non-IID federated learning.

## Pipeline

```text
global model
  -> FedDyn local training
  -> local model update
  -> FedDyn-aware parameter importance
  -> Top-rho parameter selection
  -> sparse update reconstruction
  -> server aggregation
```

## Paper-to-Code Mapping

| Paper concept | Paper equation | Code location | Notes |
| --- | --- | --- | --- |
| FedDyn local objective | local loss plus dynamic regularization | `system/flcore/clients/clientdyn.py`, `clientDyn.train()` | The current implementation adds a FedDyn regularization term and subtracts the historical gradient term during local training. |
| Local update | `Delta theta_k^t = theta_k^t - theta^{t-1}` | `system/flcore/clients/clientdyn.py`, `clientDyn.train()` | The sparse update is built as `current_model_vector - global_model_vector`. |
| Gradient change | `grad L_k(theta_k^t) - grad L_k(theta_k^{t-1})` | `system/flcore/clients/clientdyn.py`, `clientDyn.train()` | Current code uses `current_grad_vector - prev_grad_vector`, where `prev_grad_vector` is saved from the previous communication round. |
| FPI importance | `I = abs(gradient_change * parameter_update)` | `system/flcore/clients/clientdyn.py`, `clientDyn.train()` | Implemented as `torch.abs(delta_grad * delta_theta)`. |
| Top-rho selection | `S = Top_rho(I)` | `system/flcore/clients/clientdyn.py`, `clientDyn.train()` | Whole-model flattened Top-k selection. |
| Compressed update | keep selected update entries and zero the rest | `system/flcore/clients/clientdyn.py`, `clientDyn.train()` | Implemented by masking the update vector and reconstructing an uploaded model. |
| Server aggregation | aggregate uploaded compressed updates | `system/flcore/servers/serverdyn.py` | FedDyn uses uniform averaging plus the server dynamic state correction. |
| Communication cost | `C = R(rho) * rho` | Not found | The current code stores accuracy/loss/AUC only; normalized communication cost is not computed in code. |

## Known Implementation Differences

- The code stores the previous round gradient, not a gradient recomputed exactly at the current round's received global model.
- FPI ranking is whole-model global Top-k, not layer-wise Top-k.
- All trainable parameters are included in the flattened ranking.
- The server receives reconstructed model objects, not an actual sparse payload with transmitted indices.
- Communication cost is not measured at byte level.
- Existing result files have different curve lengths across datasets and methods.

## Data Partition

The checked dataset metadata uses Dirichlet label-skew Non-IID partitioning with `alpha = 0.1` and 20 clients.

The dataset generation scripts merge the original train and test splits, partition the merged data across clients, and then split each client locally with `train_ratio = 0.75`.
