# src/trading/strategy.py

import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin

class TradingStrategy:
    """
    Generate bar-by-bar trade signals (1 = long, -1 = short, 0 = flat)
    from a fitted binary classification model.
    """

    def __init__(self, model: ClassifierMixin, long_threshold: float = 0.6, short_threshold: float = 0.4):
        """
        Args:
            model: any sklearn-like classifier with predict_proba()
            long_threshold: prob ≥ this ⇒ long signal
            short_threshold: prob ≤ this ⇒ short signal
        """
        self.model = model
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_signals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Given feature DataFrame X, return a DataFrame of signals.

        Returns:
            DataFrame with columns:
              - 'prob_long': model.predict_proba()[:,1]
              - 'signal':  1, 0, or -1
        """
        # get probability of class “1” (long)
        probs = self.model.predict_proba(X)[:, 1]
        df = pd.DataFrame(index=X.index)
        df['prob_long'] = probs

        # signal logic
        df['signal'] = 0
        df.loc[df['prob_long'] >= self.long_threshold, 'signal'] = 1
        df.loc[df['prob_long'] <= self.short_threshold, 'signal'] = -1

        return df
