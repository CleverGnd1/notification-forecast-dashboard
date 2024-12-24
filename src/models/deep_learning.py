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

def train_lstm_model(data, seq_length=6, epochs=50):
    """
    Treina um modelo LSTM para previsão de séries temporais.
    """
    if len(data) < seq_length * 2:
        print("LSTM: Dados insuficientes (mínimo 12 pontos)")
        return None, None

    try:
        print("LSTM: Iniciando preparação dos dados...")
        # Normalizar dados
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['notification_count'].values.reshape(-1, 1))

        # Preparar sequências
        X, y = prepare_sequences(scaled_data, seq_length)

        print("LSTM: Criando e configurando modelo...")
        # Criar modelo
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            Dropout(0.1),
            LSTM(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Reshape dados para LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        print("LSTM: Iniciando treinamento...")
        # Treinar modelo
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=16,
            verbose=0,
            validation_split=0.1
        )

        # Imprimir métricas finais
        final_loss = history.history['loss'][-1]
        print(f"LSTM: Treinamento concluído com sucesso (Loss: {final_loss:.6f})")

        return model, scaler
    except Exception as e:
        print(f"LSTM: Erro durante o treinamento - {str(e)}")
        return None, None

def train_nbeats_model(data, seq_length=6, epochs=50):
    """
    Treina um modelo N-BEATS simplificado para previsão de séries temporais.
    """
    if len(data) < seq_length * 2:
        print("N-BEATS: Dados insuficientes (mínimo 12 pontos)")
        return None, None

    try:
        print("N-BEATS: Iniciando preparação dos dados...")
        # Normalizar dados
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['notification_count'].values.reshape(-1, 1))

        # Criar dataset
        dataset = TimeSeriesDataset(scaled_data, seq_length)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        print("N-BEATS: Criando e configurando modelo...")
        # Criar modelo
        model = NBeatsModel(seq_length, hidden_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        print("N-BEATS: Iniciando treinamento...")
        # Treinar modelo
        model.train()
        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 10 == 0:
                print(f"N-BEATS: Época {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        print(f"N-BEATS: Treinamento concluído com sucesso (Melhor Loss: {best_loss:.6f})")
        return model, scaler
    except Exception as e:
        print(f"N-BEATS: Erro durante o treinamento - {str(e)}")
        return None, None

def make_lstm_predictions(model, scaler, last_data, months_ahead=12, seq_length=6):
    """
    Gera previsões usando modelo LSTM.
    """
    if model is None or scaler is None:
        print("LSTM Predictions: Modelo ou scaler não disponíveis")
        return None

    try:
        print("LSTM Predictions: Iniciando geração de previsões...")
        # Preparar dados de entrada
        input_data = scaler.transform(last_data[-seq_length:]['notification_count'].values.reshape(-1, 1))
        input_seq = input_data.reshape((1, seq_length, 1))

        # Fazer previsões
        predictions = []
        current_seq = input_seq.copy()

        for i in range(months_ahead):
            # Fazer previsão
            pred = model.predict(current_seq, verbose=0)[0][0]
            pred = max(0, pred)  # Garantir valores não negativos
            predictions.append(pred)

            # Atualizar sequência para próxima previsão
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred

        # Reverter normalização
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        print("LSTM Predictions: Previsões geradas com sucesso")
        return predictions.flatten()
    except Exception as e:
        print(f"LSTM Predictions: Erro ao gerar previsões - {str(e)}")
        return None

def make_nbeats_predictions(model, scaler, last_data, months_ahead=12, seq_length=6):
    """
    Gera previsões usando modelo N-BEATS.
    """
    if model is None or scaler is None:
        print("N-BEATS Predictions: Modelo ou scaler não disponíveis")
        return None

    try:
        print("N-BEATS Predictions: Iniciando geração de previsões...")
        # Preparar dados de entrada
        input_data = scaler.transform(last_data[-seq_length:]['notification_count'].values.reshape(-1, 1))
        input_seq = torch.FloatTensor(input_data)

        # Fazer previsões
        predictions = []
        current_seq = input_seq.clone()

        model.eval()
        with torch.no_grad():
            for i in range(months_ahead):
                # Fazer previsão
                output = model(current_seq)
                pred = output[-1].item()
                pred = max(0, pred)  # Garantir valores não negativos
                predictions.append(pred)

                # Atualizar sequência para próxima previsão
                current_seq = torch.roll(current_seq, -1, dims=0)
                current_seq[-1] = torch.tensor(pred)

        # Reverter normalização
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        print("N-BEATS Predictions: Previsões geradas com sucesso")
        return predictions.flatten()
    except Exception as e:
        print(f"N-BEATS Predictions: Erro ao gerar previsões - {str(e)}")
        return None
