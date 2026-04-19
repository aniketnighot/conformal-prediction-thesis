
**Master's Thesis Implementation**  

**Thesis Title**: "Conformal Uncertainty for Operator Deep Smoothing of Implied-Volatility Surfaces"

---

## Overview

This repository contains the implementation of conformal prediction methods for uncertainty quantification in deep learning-based implied volatility surface construction. The implementation demonstrates distribution-free coverage guarantees on real SPX options data.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook volatility_smoothing/conformal/notebooks/conformal_prediction.ipynb
```

Or open in VS Code / JupyterLab.

---

## Implementation Overview

### Core Methodology

**1. Rolling Window Calibration** 
- Uses most recent trading surfaces for temporal adaptation
- Computes nonconformity scores 

**2. Prediction with Uncertainty Bands** 
- Applies conformal quantiles to new test surfaces
- Provides distribution-free uncertainty quantification

**3. Validation on Held-Out Data**
- Temporal train/test split 
- Demonstrates robustness across market conditions

---
## Dataset

- **Index**: SPX (S&P 500)
- **Time Period**: December 2025 - April 2026

---

## Code Structure

### Core Implementation (`volatility_smoothing/conformal/`)
- `nonconformity_scores.py` - Score implementation 
- `rolling_window.py` - sliding window calibration
- `conformal_prediction.ipynb` - Demonstrate complete flow

### Model Architecture (`op_ds/`)
- `gno/gno.py` - Graph Neural Operator for IV surface prediction
- `gno/kernel.py` - Kernel transformations for GNO layers
- `utils/fnn.py` - Feedforward network components

### Utilities (`volatility_smoothing/utils/`)
- `black_scholes.py` - Vega calculation, Black-Scholes pricing
- `options_data.py` - SPX data loading and preprocessing
- `train/dataset.py` - GNO data wrapper
- `train/loss.py` - Training loss function
- `train/misc.py` - Model checkpoint loading

### Data & Model
- `data/openbb/spx/` - CSV files with SPX options quotes
- `train/store/9448705/checkpoints/` - Trained GNO model

---
