import numpy as np
from datetime import timedelta

# ────────────────────────────────────────────────────────
# Regime-Based Walk-Forward Testing
# ────────────────────────────────────────────────────────
def regime_walk_forward(
    df_features: 'pd.DataFrame',
    df_price: 'pd.DataFrame',
    target: 'pd.Series',
    model_builder,          # function that returns an unfitted pipeline
    n_windows: int = 5,
    initial_equity: float = 10_000.0,
    pip_size: float = 0.01
):
    """
    Perform dynamic regime-based walk-forward:
      - Split full date span into n_windows+1 segments
      - For each window i:
          Train on data ≤ split_i
          Test on data between split_i and split_{i+1}
          Fit, signal, backtest
          Log metrics per window
    """
    import pandas as pd
    print(f"\n🧭 STEP: Regime Walk-Forward ({n_windows} windows)")
    dates = df_features.index.sort_values()
    start, end = dates.min(), dates.max()
    total_days = (end - start).days
    delta = total_days / (n_windows + 1)
    split_points = [start + timedelta(days=delta * i) for i in range(n_windows + 2)]

    from trading.backtester import backtest_signals
    for i in range(1, n_windows + 1):
        train_end = split_points[i]
        test_start = train_end
        test_end = split_points[i+1]
        print(f"\n🔄 Window {i}: Train ≤ {train_end.date()}, Test {test_start.date()}→{test_end.date()}")

        train_mask = df_features.index <= train_end
        test_mask = (df_features.index > train_end) & (df_features.index <= test_end)
        X_tr, y_tr = df_features.loc[train_mask], target.loc[train_mask]
        X_te, y_te = df_features.loc[test_mask], target.loc[test_mask]
        if X_te.empty:
            print("⚠️ No test data for this window, skipping.")
            continue

        # Build & train
        pipe = model_builder()
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:,1]
        sigs = pd.DataFrame({'signal': np.where(probs>=0.6,1,np.where(probs<=0.4,-1,0))}, index=X_te.index)

        # Backtest
        price_sub = df_price.loc[X_te.index]
        eq = backtest_signals(sigs, price_sub, initial_equity=initial_equity, pip_size=pip_size)
        final = eq.iloc[-1]
        ret = (final - initial_equity) / initial_equity * 100
        dd  = ((eq.cummax() - eq) / eq.cummax()).max() * 100
        print(f"📊 Window {i} → Return: {ret:.2f}%, Max DD: {dd:.2f}%")


# ────────────────────────────────────────────────────────
# Monte Carlo Trade-Sequence Bootstrap
# ────────────────────────────────────────────────────────
def monte_carlo_bootstrap(
    equity: 'pd.Series',
    n_sims: int = 1000,
    initial_equity: float = 10_000.0
):
    """
    Bootstrap bar returns n_sims times,
    report mean±std of final return & max drawdown.
    """
    print(f"\n🎲 STEP: Monte Carlo Bootstrap ({n_sims} sims)")
    # bar returns
    rets = equity.pct_change().dropna().values
    final_rets = []
    max_dds = []
    for _ in range(n_sims):
        samp = np.random.choice(rets, size=len(rets), replace=True)
        eq_sim = initial_equity * np.cumprod(1 + samp)
        final_rets.append((eq_sim[-1] - initial_equity) / initial_equity * 100)
        dd = ((np.maximum.accumulate(eq_sim) - eq_sim) / np.maximum.accumulate(eq_sim)).max() * 100
        max_dds.append(dd)
    print(f"Final Return: {np.mean(final_rets):.2f}% ± {np.std(final_rets):.2f}%")
    print(f"Max Drawdown: {np.mean(max_dds):.2f}% ± {np.std(max_dds):.2f}%")
