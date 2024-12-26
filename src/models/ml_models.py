import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def create_features(data, frequency='monthly'):
    """
    Cria features para os modelos de ML.

    Args:
        data (pd.DataFrame): DataFrame com os dados
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
    """
    df = data.copy()
    df['year'] = df.index.year

    if frequency == 'weekly':
        df['week'] = df.index.isocalendar().week
        df['day_of_year'] = df.index.dayofyear
    else:
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

    df['trend'] = np.arange(len(df))
    return df

def make_future_features(last_date, steps, frequency='monthly'):
    """
    Cria features para previsões futuras.

    Args:
        last_date (pd.Timestamp): Última data dos dados
        steps (int): Número de passos para previsão
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
    """
    if frequency == 'weekly':
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=steps,
            freq='W'
        )
    else:
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq='M'
        )

    future_df = pd.DataFrame(index=future_dates)
    future_df['year'] = future_df.index.year

    if frequency == 'weekly':
        future_df['week'] = future_df.index.isocalendar().week
        future_df['day_of_year'] = future_df.index.dayofyear
    else:
        future_df['month'] = future_df.index.month
        future_df['quarter'] = future_df.index.quarter

    future_df['trend'] = np.arange(len(future_df)) + len(future_df)
    return future_df

def train_ml_models(data, forecast_steps=12, frequency='monthly', target_year=2025):
    """
    Treina modelos de machine learning.

    Args:
        data (pd.DataFrame): DataFrame com os dados
        forecast_steps (int): Número de passos para previsão
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
        target_year (int): Ano alvo para as previsões
    """
    predictions = {}

    # Preparar features
    df = create_features(data, frequency)
    y = df['notification_count']

    if frequency == 'weekly':
        X = df[['year', 'week', 'day_of_year', 'trend']]
    else:
        X = df[['year', 'month', 'quarter', 'trend']]

    # Criar features futuras
    future_features = make_future_features(data.index[-1], forecast_steps, frequency)

    # Random Forest
    try:
        print("Random Forest: Iniciando treinamento...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        print("Random Forest: Treinamento concluído com sucesso")

        print("ML Predictions: Iniciando geração de previsões...")
        rf_predictions = rf_model.predict(future_features)
        predictions["Machine Learning - Random Forest"] = pd.Series(
            rf_predictions,
            index=future_features.index
        )
        print("ML Predictions: Previsões geradas com sucesso")
        print("✓ Random Forest: Previsão gerada com sucesso")
    except Exception as e:
        print(f"Random Forest: Erro durante o treinamento - {str(e)}")

    # XGBoost
    try:
        print("XGBoost: Iniciando treinamento...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X, y)
        print("XGBoost: Treinamento concluído com sucesso")

        print("ML Predictions: Iniciando geração de previsões...")
        xgb_predictions = xgb_model.predict(future_features)
        predictions["Machine Learning - XGBoost"] = pd.Series(
            xgb_predictions,
            index=future_features.index
        )
        print("ML Predictions: Previsões geradas com sucesso")
        print("✓ XGBoost: Previsão gerada com sucesso")
    except Exception as e:
        print(f"XGBoost: Erro durante o treinamento - {str(e)}")

    # LightGBM
    try:
        print("LightGBM: Iniciando treinamento...")
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        lgb_model.fit(X, y)
        print("LightGBM: Treinamento concluído com sucesso")

        print("ML Predictions: Iniciando geração de previsões...")
        lgb_predictions = lgb_model.predict(future_features)
        predictions["Machine Learning - LightGBM"] = pd.Series(
            lgb_predictions,
            index=future_features.index
        )
        print("ML Predictions: Previsões geradas com sucesso")
        print("✓ LightGBM: Previsão gerada com sucesso")
    except Exception as e:
        print(f"LightGBM: Erro durante o treinamento - {str(e)}")

    return predictions
