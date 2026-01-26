# multi_period_optimization

Multi-period portfolio optimization with
- regime-switching GBM
- transaction costs
- PPO-based no-trade band policies

## Structure
- src/        : core library (env, PPO, regime models)
- scripts/    : experiment entry points
- configs/    : regime / experiment configs
- asset_data/ : input data (not tracked)

## How to run
```bash
python -m scripts.train_regime_gbm
python -m scripts.regime_hmm_estimation
