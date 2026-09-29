# CNN for Pneumonia Detection in Chest X-Rays

A from-scratch Convolutional Neural Network implemented in pure **NumPy** with
**Numba JIT** acceleration and CPU **multiprocessing** — no TensorFlow / PyTorch /Keras. 

Built as part of a BSc Computer Science dissertation to demonstrate the
underlying mathematics of every layer, gradient and optimisation step.

The task is binary classification of paediatric chest X-rays as **PNEUMONIA** or **NORMAL**.

---

## Features

- **Optimiser:** Mini-Batch Gradient Descent (MBGD)
  - Behaves as **SGD** if `BATCH_SIZE = 1`
  - Behaves as **full BGD** if `BATCH_SIZE = total training samples` (5216 by default)

- **Initialisation:** He initialisation for all conv filters and dense weights

- **Activations:** ReLU in conv layers, Leaky ReLU in dense hidden layers, Sigmoid on output

- **Loss:** Weighted Binary Cross-Entropy with class weights derived from dataset imbalance

- **Normalisation:** custom 4D Batch Norm (conv) and 2D Batch Norm (dense), with running stats for val/test

- **Pooling:** 2x2 Max Pool with stride 2 after each conv layer

- **Regularisation:** L2 on conv + dense layers, Dropout in dense hidden layers

- **Schedule:** step-wise learning-rate decay

- **Data:** runtime image augmentation (rotation, translation, zoom, brightness, contrast, shear)

- **Training control:** early stopping on validation BCE, resumable training from any saved epoch

- **Performance:** per-image conv / pool kernels dispatched across CPU cores via `multiprocessing.Pool`,
  inner loops compiled with Numba `@njit`

---

## Architecture

```
Input  : (B, 222, 222, 1) grayscale X-ray
Conv1  : 16  filters 3x3   → BatchNorm → ReLU → MaxPool 2x2
Conv2  : 32  filters 3x3   → BatchNorm → ReLU → MaxPool 2x2
Conv3  : 64  filters 3x3   → BatchNorm → ReLU → MaxPool 2x2
Conv4  : 128 filters 3x3   → BatchNorm → ReLU → MaxPool 2x2
Flatten
Dense1 : Dense → BatchNorm → LeakyReLU → Dropout(0.5)
Dense2 : Dense → Sigmoid                    (output)
```

Hidden dense layer width is computed from the flattened conv output and capped
to keep RAM usage reasonable (`get_Dense_Shape` in `main.py`).

---

## Default hyperparameters (`main.py`)

| Parameter | Value | Notes |
|---|---|---|
| `IMG_SIZE` | `(222, 222)` | grayscale, normalised to `[0, 1]` |
| `FILTER_SHAPE` | `(3, 3, 1)` | sensitive — do not change |
| `NUM_FILTERS` | `[16, 32, 64, 128]` | one entry per conv layer |
| `DENSE_LAYERS` | `2` | including the sigmoid output layer |
| `EPOCHS` | `100` | upper bound; early stopping usually triggers first |
| `BATCH_SIZE` | `32` | MBGD; switch to 1 for SGD or 5216 for BGD |
| `LR` | `0.001` | initial learning rate |
| `DECAY_RATE` / `DECAY_EPOCHS` | `0.9` / `5` | step decay |
| `L2_LAMBDA_DENSE` / `L2_LAMBDA_CONV` | `1e-3` / `1e-4` | weight decay strength |
| `EARLY_STOP_EPOCHS` | `10` | epochs without val improvement before stopping |
| `DROPOUT_RATE` | `0.5` | dense hidden layers only |
| `SEED` | `20030701` | reproducible shuffling and init |
| `CPU_CORES` | `16` | size of the multiprocessing pool |

---

## Project layout

```
CNN/
├── main.py              # Single-file implementation (layers, train/val/test, IO)
├── README.txt           # Original short summary
├── README_NEW.md        # (this file)
├── DESCRIPTION_NEW.md   # One-page project description
├── Aug_img_xray/        # Output for augmented training images (created at runtime)
└── Weights/
    └── RUN_<n>__<MM-DD>/
        ├── Epoch_<k>__<HH-MM-SS>/
        │   ├── conv1..conv4.npz
        │   ├── c_batch_norm1..c_batch_norm4.npz
        │   ├── dense1..dense2.npz
        │   ├── d_batch_norm1.npz
        │   └── val.txt          # per-epoch validation metrics + confusion matrix
        ├── train_log.txt
        └── val_log.txt
```

A weights directory uses `.npz` per layer so a run can be resumed from any saved
epoch — pass the run folder name to `train(run_file=...)` and it picks up from
the last completed epoch (state, BN running stats, error logs, early-stop wait
counter, all restored).

---

## Expected dataset

Place the chest X-ray dataset under a sibling `IMG_xray/` directory (not
included in the repo):

```
IMG_xray/
├── train/
│   ├── PNEUMONIA/   *.jpeg
│   └── NORMAL/      *.jpeg
├── val/
│   ├── PNEUMONIA/   *.jpeg
│   └── NORMAL/      *.jpeg
└── test/
    ├── PNEUMONIA/   *.jpeg
    └── NORMAL/      *.jpeg
```

Built around the Kermany et al. paediatric chest X-ray dataset
(5,216 train / 16 val / 624 test images).

---

## Requirements

- Python 3.9+
- `numpy`
- `numba`
- `pillow`
- `matplotlib`

Install:

```bash
pip install numpy numba pillow matplotlib
```

A multi-core CPU is recommended — the conv / pool layers parallelise per-image
across `CPU_CORES` workers (default 16; lower this in `main.py` to match your
machine).

---

## Usage

Edit the bottom of `main.py` (or import from a REPL) to run one of:

```python
# Train from scratch — creates a new RUN_* folder under Weights/
train()

# Resume training from a saved run (state + BN stats + logs all restored)
train(run_file="RUN_18__05-20")

# Evaluate a specific epoch on the test set
test(run_file="RUN_18__05-20", epoch=24)
```

Each epoch:
1. Shuffles the training set with a per-epoch seed offset (so resumed training
   produces the same shuffle as continuous training).
2. Streams batches: half preprocessed normally, half augmented on the fly.
3. Forward pass → weighted BCE + L2 → backward pass → SGD step on weights, BN
   `gamma` / `beta`, biases.
4. Saves per-layer `.npz` checkpoints for the epoch.
5. Runs validation, writes `val.txt` (avg BCE, accuracy, TP/FP/FN/TN), updates
   the early-stopping counter.

---

## Notes

- `WARN = True` enables runtime warnings for dead ReLUs, saturated sigmoids,
  exploding/vanishing gradients, and extreme weight norms — useful when tuning.
- The conv backward pass is the main bottleneck; lowering `BATCH_SIZE` or
  `NUM_FILTERS` is the easiest way to speed up experimentation.
- Hyperparameters marked *"Sensitive — do not change"* (image size, filter
  shape) require corresponding changes elsewhere in the network.

## REFERENCES

1.Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled Optical Coherence
> Tomography (OCT) and Chest X-Ray Images for Classification.* Mendeley Data,
> V2. https://doi.org/10.17632/rscbjbr9sj.2
