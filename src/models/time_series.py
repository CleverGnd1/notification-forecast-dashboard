import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_statistical_models(data, forecast_steps=12, seasonal_period=12, target_year=2025):
    """
    Treina modelos estatísticos de séries temporais.

    Args:
        data (pd.DataFrame): DataFrame com os dados
        forecast_steps (int): Número de passos para previsão
        seasonal_period (int): Período de sazonalidade
        target_year (int): Ano alvo para as previsões
    """
    predictions = {}
    y = data['notification_count']

    # Validar dados
    if y.empty:
        print("Erro: Não há dados suficientes para treinar os modelos")
        return predictions

    if (y <= 0).any():
        print("Aviso: Detectados valores não positivos nos dados. Ajustando...")
        y = y.clip(lower=1)  # Ajusta valores menores que 1 para 1

    # Verificar se há variação nos dados
    if y.std() == 0:
        print("Erro: Dados constantes, não é possível treinar os modelos")
        return predictions

    # Criar índices futuros
    future_dates = pd.date_range(
        start=y.index[-1].to_timestamp() + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='M'
    )

    # ARIMA
    try:
        print("ARIMA: Iniciando treinamento...")
        model = ARIMA(y.values, order=(1, 1, 1))
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=forecast_steps)
        # Garantir que as previsões sejam positivas
        forecast = pd.Series(
            np.maximum(forecast_values, 1),  # Garantir valores positivos
            index=future_dates
        )
        predictions["Estatísticos - ARIMA"] = forecast
        print("ARIMA: Treinamento concluído com sucesso")
        print("✓ ARIMA: Previsão gerada com sucesso")
    except Exception as e:
        print(f"ARIMA: Erro durante o treinamento - {str(e)}")

    # SARIMA
    try:
        print("SARIMA: Iniciando treinamento...")
        model = SARIMAX(y.values, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
        model_fit = model.fit(disp=False)  # Desabilitar output detalhado
        forecast_values = model_fit.forecast(steps=forecast_steps)
        # Garantir que as previsões sejam positivas
        forecast = pd.Series(
            np.maximum(forecast_values, 1),  # Garantir valores positivos
            index=future_dates
        )
        predictions["Estatísticos - SARIMA"] = forecast
        print("SARIMA: Treinamento concluído com sucesso")
        print("✓ SARIMA: Previsão gerada com sucesso")
    except Exception as e:
        print(f"SARIMA: Erro durante o treinamento - {str(e)}")

    # ETS (Holt-Winters)
    try:
        print("ETS: Iniciando treinamento...")
        # Adicionar pequena variação aos dados se necessário
        y_ets = y.copy()
        if y_ets.std() / y_ets.mean() < 0.01:  # Se a variação for muito pequena
            print("Aviso: Adicionando pequena variação aos dados para o ETS")
            y_ets = y_ets * (1 + np.random.normal(0, 0.001, len(y_ets)))

        y_ets = y_ets.clip(lower=1)  # Garantir valores positivos

        model = ExponentialSmoothing(
            y_ets.values,
            seasonal_periods=seasonal_period,
            trend='add',
            seasonal='add',
            use_boxcox=True
        )
        model_fit = model.fit()
        forecast_values = model_fit.forecast(forecast_steps)
        # Garantir que as previsões sejam positivas
        forecast = pd.Series(
            np.maximum(forecast_values, 1),  # Garantir valores positivos
            index=future_dates
        )
        predictions["Estatísticos - ETS"] = forecast
        print("ETS: Treinamento concluído com sucesso")
        print("✓ ETS: Previsão gerada com sucesso")
    except Exception as e:
        print(f"ETS: Erro durante o treinamento - {str(e)}")

    return predictions
