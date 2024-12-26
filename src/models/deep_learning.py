import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences(data, n_steps=6, frequency='monthly'):
    """
    Prepara sequências para modelos de deep learning.

    Args:
        data (pd.Series): Série temporal
        n_steps (int): Tamanho da sequência
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
    """
    if isinstance(data, pd.Series):
        values = data.to_numpy().reshape(-1, 1)
    else:
        values = np.array(data).reshape(-1, 1)

    # Normalizar dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Criar sequências
    X, y = [], []
    for i in range(len(scaled) - n_steps):
        X.append(scaled[i:(i + n_steps), 0])
        y.append(scaled[i + n_steps, 0])

    return np.array(X), np.array(y), scaler

def train_deep_learning_models(data, forecast_steps=12, frequency='monthly', target_year=2025):
    """
    Treina modelos de deep learning.

    Args:
        data (pd.DataFrame): DataFrame com os dados
        forecast_steps (int): Número de passos para previsão
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
        target_year (int): Ano alvo para as previsões
    """
    predictions = {}
    y = data['notification_count']
    last_date = y.index[-1]

    # Definir parâmetros com base na frequência
    if frequency == 'weekly':
        n_steps = 12  # 3 meses em semanas
        epochs = 50
    else:
        n_steps = 6   # 6 meses
        epochs = 100

    # Preparar dados
    X, y_train, scaler = prepare_sequences(y, n_steps, frequency)

    # LSTM
    try:
        print("LSTM: Iniciando preparação dos dados...")
        X_lstm = X.reshape((X.shape[0], X.shape[1], 1))

        print("LSTM: Criando e configurando modelo...")
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(n_steps, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        print("LSTM: Iniciando treinamento...")
        history = model.fit(X_lstm, y_train, epochs=epochs, verbose=0)
        print(f"LSTM: Treinamento concluído com sucesso (Loss: {history.history['loss'][-1]:.6f})")

        print("LSTM Predictions: Iniciando geração de previsões...")
        last_sequence = np.array(y[-n_steps:]).reshape(-1, 1)
        last_sequence = scaler.transform(last_sequence)
        future_predictions = []

        current_sequence = last_sequence.reshape(1, n_steps, 1)
        for _ in range(forecast_steps):
            next_pred = model.predict(current_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1, 0] = next_pred

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()

        # Criar índices futuros
        if frequency == 'weekly':
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=forecast_steps,
                freq='W'
            )
        else:
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_steps,
                freq='M'
            )

        predictions["Deep Learning - LSTM"] = pd.Series(
            future_predictions,
            index=future_dates
        )
        print("LSTM Predictions: Previsões geradas com sucesso")
        print("✓ LSTM: Previsão gerada com sucesso")
    except Exception as e:
        print(f"LSTM: Erro durante o treinamento - {str(e)}")

    # N-BEATS (simplificado)
    try:
        print("N-BEATS: Iniciando preparação dos dados...")
        X_nbeats = X.reshape((X.shape[0], X.shape[1]))

        print("N-BEATS: Criando e configurando modelo...")
        model = Sequential([
            Dense(32, activation='relu', input_shape=(n_steps,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        print("N-BEATS: Iniciando treinamento...")
        history = model.fit(X_nbeats, y_train, epochs=epochs, verbose=0)

        last_sequence = np.array(y[-n_steps:]).reshape(-1, 1)
        last_sequence = scaler.transform(last_sequence)
        future_predictions = []

        current_sequence = last_sequence.reshape(1, n_steps)
        for _ in range(forecast_steps):
            next_pred = model.predict(current_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1] = next_pred

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()

        # Criar índices futuros
        if frequency == 'weekly':
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=forecast_steps,
                freq='W'
            )
        else:
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_steps,
                freq='M'
            )

        predictions["Deep Learning - N-BEATS"] = pd.Series(
            future_predictions,
            index=future_dates
        )
        print("✓ N-BEATS: Previsão gerada com sucesso")
    except Exception as e:
        print(f"N-BEATS: Erro durante o treinamento - {str(e)}")

    return predictions
