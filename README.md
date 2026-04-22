
# Dynamic Self-Pruning Neural Network

## Overview

This project implements a neural network that learns to prune its own weights during training using learnable gate parameters.

---

## Method Summary

* Each weight is multiplied by a learnable sigmoid gate
* Gates close to 0 → weights effectively removed
* Sparsity is encouraged using L1 regularization on gate values

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
| ------ | ------------ | ------------ |
| 1e-4   | 71.71        | 0.31         |
| 5e-4   | 71.12        | 40.53        |
| 1e-2   | 71.95        | 46.25        |

---

## Key Observations

* Increasing λ significantly increases sparsity
* At λ = 5e-4 → ~40.5% sparsity with minimal accuracy drop
* At λ = 1e-2 → ~46.2% sparsity with ~71.95% accuracy
* Very low λ (1e-4) results in almost no pruning (~0.3%)

---

## Key Insight

The model successfully prunes nearly half of its weights (~46%) while maintaining ~72% accuracy, showing effective self-pruning without performance degradation.

---

## Gate Value Distribution

![Gate Distribution](results.png)

* Strong concentration of gate values near 0 → many weights pruned
* Smaller group of higher values → important connections retained
* Clear separation indicates effective learning of important vs unnecessary weights

---

## How to Run (Google Colab)

1. Open Google Colab
2. Upload `main.py`
3. Run all cells

Or run directly:

```bash
pip install torch torchvision matplotlib numpy
python main.py
```

---

## Tech Stack

* Python
* PyTorch
* torchvision
* matplotlib
* Google Colab

---

## Conclusion

This implementation demonstrates that a neural network can dynamically prune itself during training, achieving high sparsity while maintaining strong performance.
