import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.data[idx:idx+self.seq_length],
            self.data[idx+self.seq_length]
        )

class SimpleNBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class NBeatsModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleNBeatsBlock(input_size, hidden_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        residuals = x
        for block in self.blocks:
            block_out = block(residuals)
            residuals = residuals - block_out
        return residuals

def prepare_sequences(data, seq_length):
    """
    Prepara sequências para modelos de deep learning.
    """
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def train_lstm_model(data, seq_length=12, epochs=100):
    """
    Treina um modelo LSTM para previsão de séries temporais.
    """
    if len(data) < seq_length * 2:
        return None, None

    try:
        # Normalizar dados
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['notification_count'].values.reshape(-1, 1))

        # Preparar sequências
        X, y = prepare_sequences(scaled_data, seq_length)

        # Criar modelo
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(30, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Reshape dados para LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Treinar modelo
        model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )

        return model, scaler
    except Exception as e:
        print(f"Erro ao treinar LSTM: {e}")
        return None, None

def train_nbeats_model(data, seq_length=12, epochs=100):
    """
    Treina um modelo N-BEATS simplificado para previsão de séries temporais.
    """
    if len(data) < seq_length * 2:
        return None, None

    try:
        # Normalizar dados
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['notification_count'].values.reshape(-1, 1))

        # Criar dataset
        dataset = TimeSeriesDataset(scaled_data, seq_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Criar modelo
        model = NBeatsModel(seq_length, hidden_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Treinar modelo
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        return model, scaler
    except Exception as e:
        print(f"Erro ao treinar N-BEATS: {e}")
        return None, None

def make_lstm_predictions(model, scaler, last_data, months_ahead=12, seq_length=12):
    """
    Gera previsões usando modelo LSTM.
    """
    if model is None or scaler is None:
        return None

    try:
        # Preparar dados de entrada
        input_data = scaler.transform(last_data[-seq_length:]['notification_count'].values.reshape(-1, 1))
        input_seq = input_data.reshape((1, seq_length, 1))

        # Fazer previsões
        predictions = []
        current_seq = input_seq.copy()

        for _ in range(months_ahead):
            pred = model.predict(current_seq, verbose=0)[0][0]
            predictions.append(pred)

            # Atualizar sequência para próxima previsão
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1, 0] = pred

        # Reverter normalização
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    except Exception as e:
        print(f"Erro ao gerar previsões LSTM: {e}")
        return None

def make_nbeats_predictions(model, scaler, last_data, months_ahead=12, seq_length=12):
    """
    Gera previsões usando modelo N-BEATS.
    """
    if model is None or scaler is None:
        return None

    try:
        # Preparar dados de entrada
        input_data = scaler.transform(last_data[-seq_length:]['notification_count'].values.reshape(-1, 1))
        input_seq = torch.FloatTensor(input_data)

        # Fazer previsões
        predictions = []
        current_seq = input_seq.clone()

        model.eval()
        with torch.no_grad():
            for _ in range(months_ahead):
                pred = model(current_seq).numpy()[-1]
                predictions.append(pred)

                # Atualizar sequência para próxima previsão
                current_seq = torch.roll(current_seq, -1)
                current_seq[-1] = torch.tensor(pred)

        # Reverter normalização
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    except Exception as e:
        print(f"Erro ao gerar previsões N-BEATS: {e}")
        return None
