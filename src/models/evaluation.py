import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de avaliação para as previsões.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape
    }

def evaluate_models(data, models, predictions):
    """
    Avalia o desempenho dos modelos usando diferentes métricas.
    """
    evaluation = {}

    # Separar dados de teste (últimos 3 meses)
    test_data = data.iloc[-3:]
    y_true = test_data['notification_count'].values

    # Avaliar cada modelo
    for model_name in predictions:
        if model_name != 'dates' and isinstance(predictions[model_name], (np.ndarray, pd.Series)):
            # Usar apenas os primeiros 3 meses das previsões
            y_pred = predictions[model_name][:3]
            if len(y_pred) == len(y_true):
                evaluation[model_name] = calculate_metrics(y_true, y_pred)

    return evaluation

def create_evaluation_summary(evaluations):
    """
    Cria um resumo das avaliações de todos os modelos.
    """
    summary = []

    # Agrupar modelos por categoria
    model_groups = {
        'Modelos Estatísticos': ['arima', 'sarima', 'ets'],
        'Machine Learning': ['random_forest', 'xgboost', 'lightgbm'],
        'Deep Learning': ['lstm', 'nbeats']
    }

    for channel, eval_dict in evaluations.items():
        channel_summary = f"\n=== Avaliação do Canal: {channel.upper()} ===\n"

        for group_name, models in model_groups.items():
            channel_summary += f"\n{group_name}:\n"
            channel_summary += "-" * 80 + "\n"
            channel_summary += f"{'Modelo':<15} {'MAE':>12} {'RMSE':>12} {'R²':>12} {'MAPE (%)':>12}\n"
            channel_summary += "-" * 80 + "\n"

            for model in models:
                if model in eval_dict:
                    metrics = eval_dict[model]
                    channel_summary += (
                        f"{model.upper():<15} "
                        f"{metrics['MAE']:>12.2f} "
                        f"{metrics['RMSE']:>12.2f} "
                        f"{metrics['R²']:>12.2f} "
                        f"{metrics['MAPE (%)']:>12.2f}\n"
                    )

            channel_summary += "-" * 80 + "\n"

        summary.append(channel_summary)

    return "\n".join(summary)

def find_best_models(evaluations):
    """
    Identifica os melhores modelos para cada canal com base nas métricas.
    """
    best_models = {}

    for channel, eval_dict in evaluations.items():
        best_models[channel] = {
            'RMSE': {'model': None, 'value': float('inf')},
            'MAPE': {'model': None, 'value': float('inf')},
            'R²': {'model': None, 'value': float('-inf')}
        }

        for model, metrics in eval_dict.items():
            # Melhor modelo por RMSE (menor é melhor)
            if metrics['RMSE'] < best_models[channel]['RMSE']['value']:
                best_models[channel]['RMSE'] = {
                    'model': model,
                    'value': metrics['RMSE']
                }

            # Melhor modelo por MAPE (menor é melhor)
            if metrics['MAPE (%)'] < best_models[channel]['MAPE']['value']:
                best_models[channel]['MAPE'] = {
                    'model': model,
                    'value': metrics['MAPE (%)']
                }

            # Melhor modelo por R² (maior é melhor)
            if metrics['R²'] > best_models[channel]['R²']['value']:
                best_models[channel]['R²'] = {
                    'model': model,
                    'value': metrics['R²']
                }

    return best_models

def create_best_models_summary(best_models):
    """
    Cria um resumo dos melhores modelos para cada canal.
    """
    summary = "\n=== Melhores Modelos por Canal ===\n"

    for channel, metrics in best_models.items():
        summary += f"\n{channel.upper()}:\n"
        summary += "-" * 50 + "\n"

        for metric, info in metrics.items():
            if info['model']:
                summary += (
                    f"Melhor modelo por {metric}: {info['model'].upper()} "
                    f"(valor: {info['value']:.2f})\n"
                )

        summary += "-" * 50 + "\n"

    return summary
