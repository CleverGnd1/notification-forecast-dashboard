import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def create_features(df):
    """
    Cria features para modelos de ML.
    """
    df = df.copy()

    # Remover a coluna channels que não é necessária para previsão
    if 'channels' in df.columns:
        df = df.drop('channels', axis=1)

    # Criar features temporais
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear

    # Adicionar lag features (valores anteriores)
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['notification_count'].shift(lag)

    # Adicionar médias móveis
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df['notification_count'].rolling(window=window).mean()

    # Adicionar features de tendência e sazonalidade
    df['trend'] = np.arange(len(df))
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # Preencher valores nulos com a média da coluna
    for col in df.columns:
        if col != 'notification_count':
            df[col] = df[col].fillna(df[col].mean())

    return df

def train_random_forest(data, **kwargs):
    """
    Treina um modelo Random Forest para previsão de séries temporais.
    """
    if len(data) < 12:
        print("Dados insuficientes para treinar Random Forest (mínimo 12 meses)")
        return None, None

    try:
        # Preparar dados
        df = create_features(data)

        # Separar features
        feature_columns = [col for col in df.columns if col != 'notification_count']
        X = df[feature_columns]
        y = df['notification_count']

        # Treinar modelo
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,  # Reduzido para evitar overfitting
            min_samples_leaf=2,
            min_samples_split=5,
            random_state=42,
            **kwargs
        )
        model.fit(X, y)

        return model, feature_columns
    except Exception as e:
        print(f"Erro ao treinar Random Forest: {e}")
        return None, None

def train_xgboost(data, **kwargs):
    """
    Treina um modelo XGBoost para previsão de séries temporais.
    """
    if len(data) < 12:
        print("Dados insuficientes para treinar XGBoost (mínimo 12 meses)")
        return None, None

    try:
        # Preparar dados
        df = create_features(data)

        # Separar features
        feature_columns = [col for col in df.columns if col != 'notification_count']
        X = df[feature_columns]
        y = df['notification_count']

        # Treinar modelo
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,  # Reduzido para evitar overfitting
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **kwargs
        )
        model.fit(X, y)

        return model, feature_columns
    except Exception as e:
        print(f"Erro ao treinar XGBoost: {e}")
        return None, None

def train_lightgbm(data, **kwargs):
    """
    Treina um modelo LightGBM para previsão de séries temporais.
    """
    if len(data) < 12:
        print("Dados insuficientes para treinar LightGBM (mínimo 12 meses)")
        return None, None

    try:
        # Preparar dados
        df = create_features(data)

        # Separar features
        feature_columns = [col for col in df.columns if col != 'notification_count']
        X = df[feature_columns]
        y = df['notification_count']

        # Treinar modelo
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=4,  # Reduzido para evitar overfitting
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,  # Reduzir mensagens de log
            **kwargs
        )
        model.fit(X, y)

        return model, feature_columns
    except Exception as e:
        print(f"Erro ao treinar LightGBM: {e}")
        return None, None

def make_ml_predictions(model, feature_columns, last_data, months_ahead=12):
    """
    Gera previsões usando modelos de ML.
    """
    if model is None or feature_columns is None:
        return None

    try:
        # Criar dataframe para previsões futuras
        future_dates = pd.date_range(
            start=last_data.index[-1],
            periods=months_ahead + 1,
            freq='MS'
        )[1:]

        future_df = pd.DataFrame(index=future_dates)
        future_df['notification_count'] = np.nan

        # Combinar dados históricos e futuros
        combined_df = pd.concat([last_data[['notification_count']], future_df])

        # Criar features para previsão
        pred_df = create_features(combined_df)

        # Fazer previsões mês a mês
        predictions = []
        current_features = pred_df.copy()

        for i in range(months_ahead):
            # Atualizar features
            if i > 0:
                current_features.loc[future_dates[i], 'notification_count'] = predictions[-1]
                current_features = create_features(current_features)

            # Selecionar features para previsão
            X_pred = current_features.loc[future_dates[i:i+1], feature_columns]

            # Fazer previsão
            pred = max(0, float(model.predict(X_pred)[0]))  # Garantir previsões não-negativas
            predictions.append(pred)

        return np.array(predictions)
    except Exception as e:
        print(f"Erro ao gerar previsões ML: {e}")
        return None
