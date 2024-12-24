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

def make_future_predictions(data, channel, months_ahead=12):
    """
    Gera previsões para meses futuros com todos os modelos disponíveis.
    """
    print(f"\nGerando previsões para o canal {channel}...")

    # Filtrar dados do canal específico
    channel_data = data[data['channels'] == channel].copy()
    if channel_data.empty:
        print(f"Erro: Não foram encontrados dados para o canal {channel}")
        return None

    # Preparar dados para previsão
    channel_data = channel_data.set_index('month')
    channel_data = channel_data.sort_index()

    # Gerar previsões usando todos os modelos
    predictions = generate_predictions(channel_data, months_ahead)

    if not predictions:
        print("Erro: Nenhuma previsão foi gerada")
        return None

    print(f"Previsões geradas com sucesso para o canal {channel}")
    return predictions

def generate_predictions(data, months_ahead=12):
    """
    Gera previsões usando todos os modelos disponíveis.
    """
    print("\nIniciando geração de previsões...")
    predictions = {}

    # Preparar série temporal
    time_series = data['notification_count'].astype(float)

    # Modelos estatísticos
    try:
        print("\n=== Modelos Estatísticos ===")
        # ARIMA
        arima_model = train_arima_model(time_series)
        if arima_model is not None:
            try:
                arima_pred = arima_model.forecast(months_ahead)
                predictions['Estatísticos - ARIMA'] = arima_pred
                print("✓ ARIMA: Previsão gerada com sucesso")
            except Exception as e:
                print(f"✗ Erro na previsão ARIMA: {str(e)}")

        # SARIMA
        sarima_model = train_sarima_model(time_series)
        if sarima_model is not None:
            try:
                sarima_pred = sarima_model.forecast(months_ahead)
                predictions['Estatísticos - SARIMA'] = sarima_pred
                print("✓ SARIMA: Previsão gerada com sucesso")
            except Exception as e:
                print(f"✗ Erro na previsão SARIMA: {str(e)}")

        # ETS
        ets_model = train_ets_model(time_series)
        if ets_model is not None:
            try:
                ets_pred = ets_model.forecast(months_ahead)
                predictions['Estatísticos - ETS'] = ets_pred
                print("✓ ETS: Previsão gerada com sucesso")
            except Exception as e:
                print(f"✗ Erro na previsão ETS: {str(e)}")
    except Exception as e:
        print(f"Erro nos modelos estatísticos: {str(e)}")

    # Modelos de Machine Learning
    try:
        print("\n=== Modelos de Machine Learning ===")
        # Preparar features
        features_df = pd.DataFrame(index=data.index)
        features_df['year'] = features_df.index.year
        features_df['month'] = features_df.index.month
        features_df['day'] = features_df.index.day
        features_df['dayofweek'] = features_df.index.dayofweek
        features_df['quarter'] = features_df.index.quarter

        feature_columns = ['year', 'month', 'day', 'dayofweek', 'quarter']
        target = time_series

        # Random Forest
        rf_model = train_random_forest(features_df[feature_columns], target)
        if rf_model:
            rf_pred = make_ml_predictions(rf_model, feature_columns, data, months_ahead)
            if rf_pred is not None:
                predictions['Machine Learning - Random Forest'] = rf_pred
                print("✓ Random Forest: Previsão gerada com sucesso")

        # XGBoost
        xgb_model = train_xgboost(features_df[feature_columns], target)
        if xgb_model:
            xgb_pred = make_ml_predictions(xgb_model, feature_columns, data, months_ahead)
            if xgb_pred is not None:
                predictions['Machine Learning - XGBoost'] = xgb_pred
                print("✓ XGBoost: Previsão gerada com sucesso")

        # LightGBM
        lgb_model = train_lightgbm(features_df[feature_columns], target)
        if lgb_model:
            lgb_pred = make_ml_predictions(lgb_model, feature_columns, data, months_ahead)
            if lgb_pred is not None:
                predictions['Machine Learning - LightGBM'] = lgb_pred
                print("✓ LightGBM: Previsão gerada com sucesso")
    except Exception as e:
        print(f"Erro nos modelos de Machine Learning: {str(e)}")

    # Modelos de Deep Learning
    try:
        print("\n=== Modelos de Deep Learning ===")
        # LSTM
        lstm_model, lstm_scaler = train_lstm_model(data)
        if lstm_model and lstm_scaler:
            lstm_pred = make_lstm_predictions(lstm_model, lstm_scaler, data, months_ahead)
            if lstm_pred is not None:
                predictions['Deep Learning - LSTM'] = lstm_pred
                print("✓ LSTM: Previsão gerada com sucesso")

        # N-BEATS
        nbeats_model, nbeats_scaler = train_nbeats_model(data)
        if nbeats_model and nbeats_scaler:
            nbeats_pred = make_nbeats_predictions(nbeats_model, nbeats_scaler, data, months_ahead)
            if nbeats_pred is not None:
                predictions['Deep Learning - N-BEATS'] = nbeats_pred
                print("✓ N-BEATS: Previsão gerada com sucesso")
    except Exception as e:
        print(f"Erro nos modelos de Deep Learning: {str(e)}")

    print("\nGeração de previsões concluída!")
    print(f"Total de modelos com previsões: {len(predictions)}")
    for model_name in predictions.keys():
        print(f"- {model_name}")

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

    return df
