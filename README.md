# CryptoPredict (Mamba2 + TBPTT)

**Author:** Jiheng Li  
**Copyright:** This project and all its code, documentation, and related content are owned by Jiheng Li. Unauthorized reproduction or commercial use is prohibited.

Long‑horizon Bitcoin forecasting with a custom Mamba2 state‑space backbone, optional sparse Multi‑Head Self‑Attention (MHSA), and Truncated Backpropagation Through Time (TBPTT). The model jointly predicts: (1) a 3‑class 90‑day direction label and (2) continuous targets p90 / p10 / σ (30‑day realized vol). We emphasize **no label leakage**, **GPU‑friendly training**, and **stable optimization on a single GPU**.

## Multitask Learning Frame

We jointly predict a 3-class 90-day direction label and three continuous targets (r90’s 0.9 / 0.1 quantiles: p90 / p10, plus 30-day realized volatility σ). Labels are computed from future 90-day windows (forward log return r90, its quantiles, and σ), and direction classes are defined by thresholds on r90 (e.g., > +5% = up, < −5% = down). To prevent leakage, any feature that directly or indirectly uses future information (shifted returns, forward rolling stats, etc.) is removed. The remaining leak-free feature groups include:

- **Market / on-chain activity:** volume, transaction count, fees, active addresses, UTXO stats, etc.
- **Technical indicators:** EMA, MACD (and hist/signal), RSI, difficulty ribbon MAs, etc.
- **Macro signals:** DXY, 10Y/2Y yields, yield spread.
- **Cyclical time features:** hour/day-of-week encoded with sin/cos.

## Model Architecture

- **Block**: Mamba2 SSM + GatedMLP. Optionally plug in a lightweight MHSA branch every _N_ layers to inject global context while keeping memory in check.
- **Heads**: the pooled hidden state feeds (a) a 3‑class classifier and (b) a 3‑dim regression head (p90, p10, σ).
- **Loss**: uncertainty‑weighted multi‑task loss with learnable log variances (log_var_cls / log_var_reg).

### Example Sizes & Footprint (current codebase)

| Config             | Layers | MHSA layers | Heads | Params (≈) | TBPTT_len | Batch | Peak VRAM\* |
| ------------------ | ------ | ----------- | ----- | ---------- | --------- | ----- | ----------- |
| Baseline (no MHSA) | 4      | 0           | 3 / 3 | ~6.9M      | 1024–2048 | 16    | ~11–13 GB   |
| Sparse MHSA        | 6      | 2 (every 3) | 3 / 3 | ~9.3M      | 256–1024  | 8–16  | ~18–22 GB   |

\*Measured on a 48 GB GPU with AMP; numbers vary with sequence length and allocator state.

## Training & Memory Management (TBPTT)

- **TBPTT**: backprop every `tbptt_len` steps, then detach SSM / KV states to cap graph size and free old activations.
- **Stability tricks**: AMP, gradient clipping, conservative warm‑up, and (if needed) a first mini‑epoch using a tiny `tbptt_len` to let the CUDA allocator settle.
- **LR schedule**: short warm‑up (2–5%), then cosine decay to `lr_min = base_lr × 1e‑4`.

## Hyperparameters

- **No MHSA**: `d_model ≈ 256`, 4 layers, `tbptt_len = 1k–2k`, dropout ≈ 0.1, weight decay ≈ 1e‑2, LR ≈ 1e‑4 (AdamW).
- **Sparse MHSA**: keep `d_model` similar, insert MHSA every 3–4 layers, `num_heads = 4`, smaller `tbptt_len` (256–1k), weight decay ≈ 3e‑2, LR ≈ 3e‑5–8e‑5.

## Repository Structure

```
├── models
│   ├── blocks/
│   │   ├── mamba2_block.py
│   │   └── mamba2_enhanced_block.py
│   ├── modules/
│   │   └── multihead_attention.py
│   └── mamba2_multitask.py
├── datasets/
│   └── window_dataset.py
├── utils/
│   ├── train_val.py
│   └── misc.py
├── preprocess/
│   ├── make_labels.py
│   └── make_features.py
├── configs/
│   ├── config_baseline.py
│   └── config_mhsa.py
└── train.py
```

## Notes

- Always verify no future‑looking columns sneak into features.
- CUDA OOM can still happen if `tbptt_len` or KV cache grows unchecked—monitor and tune.
- If loss spikes to NaN: lower LR, tighten grad clip, or disable MHSA first.
