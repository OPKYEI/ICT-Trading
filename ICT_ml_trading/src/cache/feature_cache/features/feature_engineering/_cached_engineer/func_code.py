# first line: 25
@memory.cache
def _cached_engineer(
    X: pd.DataFrame,
    symbol: str,
    lookback_periods: tuple,
    feature_selection_threshold: float,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    additional_data: dict
) -> pd.DataFrame:
    """
    Standalone cached feature-engineering function (no self).
    """
    # 1) Build the engineer with the same params
    fe = ICTFeatureEngineer(
        lookback_periods=list(lookback_periods),
        feature_selection_threshold=feature_selection_threshold
    )
    # 2) Engineer features
    fs = fe.engineer_features(
        data=X,
        symbol=symbol,
        additional_data=additional_data
    )
    feats = fs.features.copy()
    # 3) Drop all look-ahead columns
    drop_cols = [c for c in feats.columns if c.startswith("future_")]
    feats.drop(columns=drop_cols, inplace=True, errors="ignore")
    # 4) Re-index to match X exactly
    return feats.reindex(X.index)
