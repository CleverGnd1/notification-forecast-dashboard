import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_predictions(data, predictions, channel, output_path, frequency='monthly'):
    """
    Gera gráfico com dados históricos e previsões.

    Args:
        data (pd.DataFrame): DataFrame com os dados históricos
        predictions (dict): Dicionário com as previsões
        channel (str): Nome do canal
        output_path (str): Caminho para salvar o gráfico
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
    """
    try:
        print(f"\nIniciando geração do gráfico para o canal {channel}...")
        print(f"Caminho do arquivo de saída: {output_path}")

        if data.empty or not predictions:
            print(f"Aviso: Não há dados suficientes para gerar o gráfico para o canal {channel}")
            if data.empty:
                print("DataFrame está vazio")
            if not predictions:
                print("Dicionário de previsões está vazio")
            return

        print(f"Dados disponíveis: {len(data)} registros")
        print(f"Colunas disponíveis: {', '.join(data.columns)}")
        print(f"Índice: {data.index}")
        print(f"Modelos com previsões: {list(predictions.keys())}")

        plt.figure(figsize=(15, 8))

        # Configurar estilo
        plt.style.use('default')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Converter índices Period para Timestamp para plotagem
        plot_data = data.copy()
        if isinstance(plot_data.index, pd.PeriodIndex):
            print("Convertendo índice Period para Timestamp...")
            plot_data.index = plot_data.index.to_timestamp()

        # Plotar dados históricos
        print("Plotando dados históricos...")
        plt.plot(plot_data.index, plot_data['notification_count'],
                label='Dados Históricos',
                marker='o',
                linewidth=2,
                markersize=6,
                color='black')

        # Plotar previsões de cada modelo
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for (model_name, forecast), color in zip(predictions.items(), colors):
            print(f"\nProcessando previsões do modelo {model_name}...")
            if isinstance(forecast, pd.Series) and not forecast.empty:
                plot_forecast = forecast.copy()
                print(f"Dados da previsão: {len(plot_forecast)} registros")
                print(f"Índice da previsão: {plot_forecast.index}")

                if isinstance(plot_forecast.index, pd.PeriodIndex):
                    print("Convertendo índice Period para Timestamp...")
                    plot_forecast.index = plot_forecast.index.to_timestamp()

                print(f"Plotando previsões do modelo {model_name}")
                print(f"Período de previsão: {plot_forecast.index[0]} até {plot_forecast.index[-1]}")
                print(f"Valores: min={plot_forecast.min():.2f}, max={plot_forecast.max():.2f}")

                plt.plot(plot_forecast.index, plot_forecast.values,
                        label=model_name,
                        marker='s',
                        linewidth=2,
                        markersize=6,
                        linestyle='--',
                        color=color)
            else:
                print(f"Previsões inválidas para o modelo {model_name}")
                if not isinstance(forecast, pd.Series):
                    print(f"Tipo incorreto: {type(forecast)}")
                if isinstance(forecast, pd.Series) and forecast.empty:
                    print("Série está vazia")

        # Configurar título e labels
        period = "Semanas" if frequency == "weekly" else "Meses"
        plt.title(f'Notificações por {period} - Canal: {channel}', fontsize=14, pad=20)
        plt.xlabel(period, fontsize=12)
        plt.ylabel('Número de Notificações', fontsize=12)

        # Configurar legenda
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Rotacionar labels do eixo x
        plt.xticks(rotation=45)

        # Ajustar layout
        plt.tight_layout()

        # Criar diretório se não existir
        print(f"\nCriando diretório: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Salvar gráfico
        print(f"Salvando gráfico em: {output_path}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        if os.path.exists(output_path):
            print(f"Gráfico salvo com sucesso em: {output_path}")
            print(f"Tamanho do arquivo: {os.path.getsize(output_path)} bytes")
        else:
            print(f"Erro: Arquivo não foi criado em {output_path}")

    except Exception as e:
        print(f"Erro ao gerar gráfico para o canal {channel}: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close()

def plot_predictions_2024(data, predictions, channel, output_path, frequency='monthly'):
    """
    Gera gráfico focado nas previsões para 2024.

    Args:
        data (pd.DataFrame): DataFrame com os dados históricos
        predictions (dict): Dicionário com as previsões
        channel (str): Nome do canal
        output_path (str): Caminho para salvar o gráfico
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
    """
    try:
        print(f"\nIniciando geração do gráfico 2024 para o canal {channel}...")
        print(f"Caminho do arquivo de saída: {output_path}")

        if data.empty or not predictions:
            print(f"Aviso: Não há dados suficientes para gerar o gráfico para o canal {channel}")
            if data.empty:
                print("DataFrame está vazio")
            if not predictions:
                print("Dicionário de previsões está vazio")
            return

        print(f"Dados disponíveis: {len(data)} registros")
        print(f"Colunas disponíveis: {', '.join(data.columns)}")
        print(f"Índice: {data.index}")
        print(f"Modelos com previsões: {list(predictions.keys())}")

        plt.figure(figsize=(15, 8))

        # Configurar estilo
        plt.style.use('default')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Converter índices Period para Timestamp para plotagem
        plot_data = data.copy()
        if isinstance(plot_data.index, pd.PeriodIndex):
            print("Convertendo índice Period para Timestamp...")
            plot_data.index = plot_data.index.to_timestamp()

        # Filtrar dados de 2024
        print("Filtrando dados de 2024...")
        data_2024 = plot_data[plot_data.index.year >= 2024]

        if not data_2024.empty:
            print(f"Dados históricos de 2024: {len(data_2024)} registros")
            print(f"Período: {data_2024.index[0]} até {data_2024.index[-1]}")
            # Plotar dados históricos de 2024
            plt.plot(data_2024.index, data_2024['notification_count'],
                    label='Dados Históricos',
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    color='black')
        else:
            print("Nenhum dado histórico encontrado para 2024")

        # Plotar previsões de cada modelo
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for (model_name, forecast), color in zip(predictions.items(), colors):
            print(f"\nProcessando previsões do modelo {model_name}...")
            if isinstance(forecast, pd.Series) and not forecast.empty:
                plot_forecast = forecast.copy()
                print(f"Dados da previsão: {len(plot_forecast)} registros")
                print(f"Índice da previsão: {plot_forecast.index}")

                if isinstance(plot_forecast.index, pd.PeriodIndex):
                    print("Convertendo índice Period para Timestamp...")
                    plot_forecast.index = plot_forecast.index.to_timestamp()

                forecast_2024 = plot_forecast[plot_forecast.index.year >= 2024]
                if not forecast_2024.empty:
                    print(f"Plotando previsões do modelo {model_name} para 2024")
                    print(f"Período de previsão: {forecast_2024.index[0]} até {forecast_2024.index[-1]}")
                    print(f"Valores: min={forecast_2024.min():.2f}, max={forecast_2024.max():.2f}")

                    plt.plot(forecast_2024.index, forecast_2024.values,
                            label=model_name,
                            marker='s',
                            linewidth=2,
                            markersize=6,
                            linestyle='--',
                            color=color)
                else:
                    print(f"Nenhuma previsão para 2024 encontrada no modelo {model_name}")
            else:
                print(f"Previsões inválidas para o modelo {model_name}")
                if not isinstance(forecast, pd.Series):
                    print(f"Tipo incorreto: {type(forecast)}")
                if isinstance(forecast, pd.Series) and forecast.empty:
                    print("Série está vazia")

        # Configurar título e labels
        period = "Semanas" if frequency == "weekly" else "Meses"
        plt.title(f'Notificações por {period} em 2024 - Canal: {channel}', fontsize=14, pad=20)
        plt.xlabel(period, fontsize=12)
        plt.ylabel('Número de Notificações', fontsize=12)

        # Configurar legenda
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Rotacionar labels do eixo x
        plt.xticks(rotation=45)

        # Ajustar layout
        plt.tight_layout()

        # Criar diretório se não existir
        print(f"\nCriando diretório: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Salvar gráfico
        print(f"Salvando gráfico em: {output_path}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        if os.path.exists(output_path):
            print(f"Gráfico salvo com sucesso em: {output_path}")
            print(f"Tamanho do arquivo: {os.path.getsize(output_path)} bytes")
        else:
            print(f"Erro: Arquivo não foi criado em {output_path}")

    except Exception as e:
        print(f"Erro ao gerar gráfico 2024 para o canal {channel}: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close()
