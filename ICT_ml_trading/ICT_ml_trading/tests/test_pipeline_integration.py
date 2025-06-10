# tests/test_pipeline_integration.py

def test_pipeline_integration(tmp_path):
    import os
    from data_processing.data_loader      import DataLoader
    from features.feature_engineering     import ICTFeatureEngineer
    from ml_models.model_builder          import ModelBuilder
    from ml_models.trainer                import grid_search_with_checkpoint
    from trading.strategy                 import TradingStrategy
    from trading.backtester               import backtest_signals

    # 1) load
    df = DataLoader(data_path="data/raw")\
         .load_data("EURUSD", "2023-01-01", "2023-01-10", "1h")

    # 2) features
    fe = ICTFeatureEngineer([5,10,20], 0.01)
    fs = fe.engineer_features(df, symbol="EURUSD", additional_data={})
    X, y = fs.features, fs.features.pop("future_direction_5")

    # 3) train
    model, _ = grid_search_with_checkpoint(
        build_random_forest(50, None),
        {"clf__n_estimators": [50,100]},
        X, y,
        cv=3, scoring="accuracy",
        checkpoint_path=str(tmp_path/"rf.pkl")
    )

    # 4) signals & backtest
    signals = TradingStrategy(model).generate_signals(X)
    equity  = backtest_signals(signals, df[["close"]], 10_000)

    # 5) sanity check
    assert equity.iloc[-1] >= 0
