import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_arima_model(data):
    """
    Treina um modelo ARIMA.
    """
    try:
        print("ARIMA: Iniciando treinamento...")
        model = ARIMA(data, order=(1, 1, 1))
        fitted_model = model.fit()
        print("ARIMA: Treinamento concluído com sucesso")
        return fitted_model
    except Exception as e:
        print(f"ARIMA: Erro durante o treinamento - {str(e)}")
        return None

def train_sarima_model(data):
    """
    Treina um modelo SARIMA.
    """
    try:
        print("SARIMA: Iniciando treinamento...")
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fitted_model = model.fit(disp=False)
        print("SARIMA: Treinamento concluído com sucesso")
        return fitted_model
    except Exception as e:
        print(f"SARIMA: Erro durante o treinamento - {str(e)}")
        return None

def train_ets_model(data):
    """
    Treina um modelo ETS (Exponential Smoothing).
    """
    try:
        print("ETS: Iniciando treinamento...")
        model = ExponentialSmoothing(
            data,
            seasonal_periods=12,
            trend='add',
            seasonal='add',
            damped_trend=True
        )
        fitted_model = model.fit()
        print("ETS: Treinamento concluído com sucesso")
        return fitted_model
    except Exception as e:
        print(f"ETS: Erro durante o treinamento - {str(e)}")
        return None
