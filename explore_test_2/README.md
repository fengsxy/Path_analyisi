# Explore Test 2: EMA Training

This experiment adds Exponential Moving Average (EMA) to both Baseline and LIFT models.

## Motivation

From our previous experiments:
- Baseline achieves best FID at 200ep (33.11), then degrades to 59.40 at 2000ep
- LIFT DP-Total achieves best FID at 2000ep (36.55)
- Training loss keeps decreasing but FID increases → possible overfitting

EMA is expected to:
1. Stabilize training by averaging weights over time
2. Reduce overfitting by smoothing weight updates
3. Potentially improve FID at later epochs

## EMA Implementation

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {name: param.clone() for name, param in model.named_parameters()}

    def update(self):
        # Called after optimizer.step()
        for name, param in model.named_parameters():
            self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param
```

**Decay rate**: 0.9999 (standard for diffusion models)
- Higher decay = slower averaging = more stable but slower to adapt
- Lower decay = faster averaging = more responsive but less stable

## Usage

```bash
# Train both models with EMA (2 GPUs)
./explore_test_2/train_ema.sh

# Or train individually
python explore_test_2/train_baseline_ema.py --epochs 2000 --device 0
python explore_test_2/train_lift_ema.py --epochs 2000 --device 1
```

## Checkpoints

Checkpoints include both regular and EMA weights:
```python
checkpoint = {
    'model_state': model.state_dict(),      # Regular weights
    'ema_state': ema.state_dict(),          # EMA weights
    'ema_decay': 0.9999,
    ...
}
```

To use EMA weights for evaluation:
```python
# Load checkpoint
checkpoint = torch.load('checkpoints/baseline_ema_2000ep.pth')
model.load_state_dict(checkpoint['model_state'])

# Create EMA and load state
ema = EMA(model)
ema.load_state_dict(checkpoint['ema_state'])

# Apply EMA weights for evaluation
ema.apply()
# ... generate images ...
ema.restore()  # Optional: restore original weights
```

## Expected Results

| Model | Without EMA | With EMA (expected) |
|-------|-------------|---------------------|
| Baseline best | 33.11 @ 200ep | ? @ ?ep |
| Baseline @ 2000ep | 59.40 | ? (should be better) |
| LIFT DP-Total best | 36.55 @ 2000ep | ? @ ?ep |

## References

- [DDPM paper](https://arxiv.org/abs/2006.11239): Uses EMA with decay=0.9999
- [EDM paper](https://arxiv.org/abs/2206.00364): Post-hoc EMA tuning
- [EMA in Deep Learning](https://arxiv.org/abs/2411.18704): Analysis of EMA benefits
