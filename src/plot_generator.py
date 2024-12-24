import matplotlib.pyplot as plt
import pandas as pd
import os

# Configurar o backend do matplotlib para não usar display
plt.switch_backend('Agg')

def plot_predictions(data, predictions, channel, output_path, start_date=None):
    """
    Gera gráficos com as previsões de todos os modelos.
    """
    # Usar estilo padrão do matplotlib
    plt.style.use('default')
    plt.figure(figsize=(15, 8))

    # Plotar dados históricos
    plt.plot(data.index, data['notification_count'],
             label='Dados Históricos', color='black', linewidth=2)

    # Definir datas futuras
    future_dates = pd.date_range(
        start=data.index[-1],
        periods=len(next(iter(predictions.values()))) + 1,
        freq='MS'
    )[1:]

    # Cores para cada tipo de modelo
    colors = {
        'Estatísticos': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # Azul, Laranja, Verde
        'Machine Learning': ['#d62728', '#9467bd', '#8c564b'],  # Vermelho, Roxo, Marrom
        'Deep Learning': ['#e377c2', '#7f7f7f']  # Rosa, Cinza
    }

    # Plotar previsões de cada modelo
    for model_name, forecast in predictions.items():
        if 'Estatísticos' in model_name:
            color = colors['Estatísticos'].pop(0)
        elif 'Machine Learning' in model_name:
            color = colors['Machine Learning'].pop(0)
        else:  # Deep Learning
            color = colors['Deep Learning'].pop(0)

        plt.plot(future_dates, forecast,
                label=model_name, linestyle='--',
                color=color, linewidth=2)

    # Configurar o gráfico
    plt.title(f'Previsões para o Canal {channel}', fontsize=14, pad=20)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Número de Notificações', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ajustar a legenda
    plt.legend(bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.,
              fontsize=10)

    # Ajustar o layout para evitar cortes
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig(output_path,
                bbox_inches='tight',
                dpi=300,
                format='png')
    plt.close()

def plot_predictions_2024(data, predictions, channel, output_path):
    """
    Gera gráficos com as previsões de todos os modelos focando em 2024.
    """
    # Usar estilo padrão do matplotlib
    plt.style.use('default')
    plt.figure(figsize=(15, 8))

    # Filtrar dados de 2024
    start_date = '2024-01-01'
    data_2024 = data[data.index >= start_date].copy()

    # Plotar dados históricos de 2024
    plt.plot(data_2024.index, data_2024['notification_count'],
             label='Dados Históricos', color='black', linewidth=2)

    # Definir datas futuras
    future_dates = pd.date_range(
        start=data.index[-1],
        periods=len(next(iter(predictions.values()))) + 1,
        freq='MS'
    )[1:]

    # Cores para cada tipo de modelo
    colors = {
        'Estatísticos': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # Azul, Laranja, Verde
        'Machine Learning': ['#d62728', '#9467bd', '#8c564b'],  # Vermelho, Roxo, Marrom
        'Deep Learning': ['#e377c2', '#7f7f7f']  # Rosa, Cinza
    }

    # Plotar previsões de cada modelo
    for model_name, forecast in predictions.items():
        if 'Estatísticos' in model_name:
            color = colors['Estatísticos'].pop(0)
        elif 'Machine Learning' in model_name:
            color = colors['Machine Learning'].pop(0)
        else:  # Deep Learning
            color = colors['Deep Learning'].pop(0)

        plt.plot(future_dates, forecast,
                label=model_name, linestyle='--',
                color=color, linewidth=2)

    # Configurar o gráfico
    plt.title(f'Previsões 2024 para o Canal {channel}', fontsize=14, pad=20)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Número de Notificações', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ajustar a legenda
    plt.legend(bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.,
              fontsize=10)

    # Ajustar o layout para evitar cortes
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig(output_path,
                bbox_inches='tight',
                dpi=300,
                format='png')
    plt.close()
