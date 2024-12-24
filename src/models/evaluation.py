import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models(data, predictions):
    """
    Avalia o desempenho dos modelos de previsão.
    """
    evaluations = {}

    # Preparar dados históricos
    historical_data = data['notification_count'].values[-12:]  # Últimos 12 meses

    for model_name, forecast in predictions.items():
        try:
            # Calcular métricas
            mse = mean_squared_error(historical_data, forecast[:12])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(historical_data, forecast[:12])
            r2 = r2_score(historical_data, forecast[:12])

            evaluations[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

            print(f"\nMétricas para {model_name}:")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.2f}")

        except Exception as e:
            print(f"Erro ao avaliar modelo {model_name}: {str(e)}")
            evaluations[model_name] = None

    return evaluations
