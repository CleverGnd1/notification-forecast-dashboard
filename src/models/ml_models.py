import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def train_random_forest(X, y):
    """
    Treina um modelo Random Forest.
    """
    try:
        print("Random Forest: Iniciando treinamento...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        print("Random Forest: Treinamento concluído com sucesso")
        return model
    except Exception as e:
        print(f"Random Forest: Erro durante o treinamento - {str(e)}")
        return None

def train_xgboost(X, y):
    """
    Treina um modelo XGBoost.
    """
    try:
        print("XGBoost: Iniciando treinamento...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        print("XGBoost: Treinamento concluído com sucesso")
        return model
    except Exception as e:
        print(f"XGBoost: Erro durante o treinamento - {str(e)}")
        return None

def train_lightgbm(X, y):
    """
    Treina um modelo LightGBM.
    """
    try:
        print("LightGBM: Iniciando treinamento...")
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        print("LightGBM: Treinamento concluído com sucesso")
        return model
    except Exception as e:
        print(f"LightGBM: Erro durante o treinamento - {str(e)}")
        return None

def make_ml_predictions(model, feature_columns, last_data, months_ahead=12):
    """
    Gera previsões usando modelos de ML.
    """
    if model is None:
        print("ML Predictions: Modelo não disponível")
        return None

    try:
        print("ML Predictions: Iniciando geração de previsões...")
        # Criar dataframe para previsões futuras
        future_dates = pd.date_range(
            start=last_data.index[-1],
            periods=months_ahead + 1,
            freq='MS'
        )[1:]

        # Criar features para previsão
        future_features = []
        for date in future_dates:
            future_features.append({
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'dayofweek': date.dayofweek,
                'quarter': date.quarter
            })
        future_features = pd.DataFrame(future_features)

        # Fazer previsões
        predictions = model.predict(future_features[feature_columns])
        predictions = np.maximum(predictions, 0)  # Garantir valores não negativos

        print("ML Predictions: Previsões geradas com sucesso")
        return predictions
    except Exception as e:
        print(f"ML Predictions: Erro ao gerar previsões - {str(e)}")
        return None
