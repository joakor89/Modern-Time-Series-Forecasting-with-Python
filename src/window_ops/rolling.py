import pandas as pd
import numpy as np

def seasonal_rolling_max(series: pd.Series, period: int, window: int) -> pd.Series:
    """
    Rolling max over seasonal lags.
    """
    return series.shift(period).rolling(window=window, min_periods=1).max()

def seasonal_rolling_mean(series: pd.Series, period: int, window: int) -> pd.Series:
    """
    Rolling mean over seasonal lags.
    """
    return series.shift(period).rolling(window=window, min_periods=1).mean()

def seasonal_rolling_min(series: pd.Series, period: int, window: int) -> pd.Series:
    """
    Rolling min over seasonal lags.
    """
    return series.shift(period).rolling(window=window, min_periods=1).min()

def seasonal_rolling_std(series: pd.Series, period: int, window: int) -> pd.Series:
    """
    Rolling std deviation over seasonal lags.
    """
    return series.shift(period).rolling(window=window, min_periods=1).std()