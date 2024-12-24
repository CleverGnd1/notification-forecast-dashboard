from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import numpy as np
import pandas as pd

def train_arima_model(data):
    """
    Treina um modelo ARIMA otimizado.
    """
    if len(data) < 6:
        return None

    # Converter para série temporal com frequência mensal
    ts_data = pd.Series(
        data['notification_count'].astype(float),
        index=pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq='ME'
        )
    )

    try:
        model = auto_arima(
            ts_data,
            start_p=0, start_q=0,
            max_p=2, max_q=2,
            m=1,
            seasonal=False,
            d=1,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        return model
    except Exception as e:
        print(f"Erro ao treinar ARIMA: {e}")
        return None

def train_sarima_model(data):
    """
    Treina um modelo SARIMA otimizado.
    """
    if len(data) < 24:
        return None

    # Converter para série temporal com frequência mensal
    ts_data = pd.Series(
        data['notification_count'].astype(float),
        index=pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq='ME'
        )
    )

    try:
        model = SARIMAX(
            ts_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            initialization='approximate_diffuse',
            enforce_stationarity=False
        )
        return model.fit(disp=False)
    except Exception as e:
        print(f"Erro ao treinar SARIMA: {e}")
        return None

def train_ets_model(data):
    """
    Treina um modelo ETS otimizado.
    """
    if len(data) < 12:
        return None

    # Converter para série temporal com frequência mensal
    ts_data = pd.Series(
        data['notification_count'].astype(float),
        index=pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq='ME'
        )
    )

    try:
        # Verificar se temos dados suficientes para sazonalidade
        if len(ts_data) >= 24:  # 2 ciclos completos
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal='add',
                seasonal_periods=12,
                initialization_method='estimated'
            )
        else:
            # Modelo mais simples sem sazonalidade
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )

        return model.fit(
            optimized=True,
            remove_bias=True,
            use_brute=False
        )
    except Exception as e:
        print(f"Erro ao treinar ETS: {e}")
        return None
