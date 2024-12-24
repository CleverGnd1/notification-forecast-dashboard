import pandas as pd
import numpy as np
from .time_series import (
    train_arima_model,
    train_sarima_model,
    train_ets_model
)
from .ml_models import (
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    make_ml_predictions
)
from .deep_learning import (
    train_lstm_model,
    train_nbeats_model,
    make_lstm_predictions,
    make_nbeats_predictions
)

def train_all_models(data):
    """
    Treina todos os modelos disponíveis.
    """
    models = {}

    # Modelos estatísticos clássicos
    print("Treinando modelos estatísticos...")
    models['arima'] = train_arima_model(data)
    models['sarima'] = train_sarima_model(data)
    models['ets'] = train_ets_model(data)

    # Modelos de Machine Learning
    print("Treinando modelos de Machine Learning...")
    models['random_forest'], models['rf_features'] = train_random_forest(data)
    models['xgboost'], models['xgb_features'] = train_xgboost(data)
    models['lightgbm'], models['lgb_features'] = train_lightgbm(data)

    # Modelos de Deep Learning
    print("Treinando modelos de Deep Learning...")
    models['lstm'], models['lstm_scaler'] = train_lstm_model(data)
    models['nbeats'], models['nbeats_scaler'] = train_nbeats_model(data)

    return models

def make_future_predictions(models, channel, last_date, months_ahead=12, data=None):
    """
    Gera previsões para meses futuros com todos os modelos disponíveis.
    """
    future_dates = pd.date_range(start=last_date, periods=months_ahead + 1, freq='MS')[1:]
    predictions = {'dates': future_dates}

    # Previsões dos modelos estatísticos clássicos
    for model_name in ['arima', 'sarima', 'ets']:
        model = models.get(model_name)
        if model is not None:
            try:
                if model_name in ['arima', 'sarima']:
                    forecast = model.predict(n_periods=months_ahead)
                    predictions[model_name] = pd.Series(forecast).values
                elif model_name == 'ets':
                    forecast = model.forecast(steps=months_ahead)
                    predictions[model_name] = pd.Series(forecast).values
            except Exception as e:
                print(f"Erro na previsão {model_name.upper()} para {channel}: {e}")

    if data is not None:
        # Previsões dos modelos de Machine Learning
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            model = models.get(model_name)
            features = models.get(f'{model_name[:2] if model_name == "random_forest" else model_name[:3]}_features')
            if model is not None and features is not None:
                try:
                    forecast = make_ml_predictions(model, features, data, months_ahead)
                    if forecast is not None:
                        predictions[model_name] = forecast
                except Exception as e:
                    print(f"Erro na previsão {model_name.upper()} para {channel}: {e}")

        # Previsões dos modelos de Deep Learning
        for model_name, scaler_suffix in [('lstm', 'lstm_scaler'), ('nbeats', 'nbeats_scaler')]:
            model = models.get(model_name)
            scaler = models.get(scaler_suffix)
            if model is not None and scaler is not None:
                try:
                    if model_name == 'lstm':
                        forecast = make_lstm_predictions(model, scaler, data, months_ahead)
                    else:  # nbeats
                        forecast = make_nbeats_predictions(model, scaler, data, months_ahead)
                    if forecast is not None:
                        predictions[model_name] = forecast
                except Exception as e:
                    print(f"Erro na previsão {model_name.upper()} para {channel}: {e}")

    return predictions

def prepare_data_for_prediction(df):
    """
    Prepara dados para modelagem preditiva.
    """
    df = df.copy()

    # Converter para datetime
    df['month'] = pd.to_datetime(df['month'])

    # Remover valores nulos
    df = df.dropna()

    # Agrupar por mês e canal, somando as contagens e removendo duplicatas
    df = (df.groupby(['month', 'channels'])['notification_count']
          .sum()
          .reset_index())

    # Ordenar por data
    df = df.sort_values('month')

    # Criar pivot table para separar os canais
    df_pivot = df.pivot(
        index='month',
        columns='channels',
        values='notification_count'
    ).fillna(0)

    # Garantir frequência mensal sem duplicatas
    new_index = pd.date_range(
        start=df_pivot.index.min(),
        end=df_pivot.index.max(),
        freq='ME'
    )

    # Reindexar sem duplicatas
    df_pivot = df_pivot.reindex(new_index, method='ffill')

    # Voltar para o formato original
    df_final = df_pivot.stack().reset_index()
    df_final.columns = ['month', 'channels', 'notification_count']
    df_final = df_final.set_index('month')

    return df_final
