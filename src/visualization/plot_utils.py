import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_predictions(data, predictions, channel, output_path, frequency='monthly'):
    """
    Gera gráfico com dados históricos e previsões usando Plotly.

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

        # Criar figura Plotly
        fig = go.Figure()

        # Converter índices Period para Timestamp para plotagem
        plot_data = data.copy()
        if isinstance(plot_data.index, pd.PeriodIndex):
            print("Convertendo índice Period para Timestamp...")
            plot_data.index = plot_data.index.to_timestamp()

        # Plotar dados históricos
        print("Plotando dados históricos...")
        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=plot_data['notification_count'],
            name='Dados Históricos',
            mode='lines+markers',
            line=dict(color='#2c3e50', width=2),
            marker=dict(size=6)
        ))

        # Plotar previsões de cada modelo
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for (model_name, forecast), color in zip(predictions.items(), colors):
            print(f"\nProcessando previsões do modelo {model_name}...")
            if isinstance(forecast, pd.Series) and not forecast.empty:
                plot_forecast = forecast.copy()
                if isinstance(plot_forecast.index, pd.PeriodIndex):
                    plot_forecast.index = plot_forecast.index.to_timestamp()

                print(f"Plotando previsões do modelo {model_name}")
                print(f"Período de previsão: {plot_forecast.index[0]} até {plot_forecast.index[-1]}")
                print(f"Valores: min={plot_forecast.min():.2f}, max={plot_forecast.max():.2f}")

                fig.add_trace(go.Scatter(
                    x=plot_forecast.index,
                    y=plot_forecast.values,
                    name=f'Previsão {model_name}',
                    mode='lines+markers',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6, symbol='square')
                ))

        # Configurar layout
        period = "Semanas" if frequency == "weekly" else "Meses"
        fig.update_layout(
            title=dict(
                text=f'Análise de Notificações - Canal {channel}',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title=f'Período ({period})',
            yaxis_title='Número de Notificações',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            width=1200,
            height=600,
            margin=dict(l=50, r=200, t=50, b=50),
            template='plotly_white'
        )

        # Criar diretório se não existir
        print(f"\nCriando diretório: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Salvar gráfico como HTML
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"Gráfico HTML salvo em: {html_path}")

        # Salvar também como imagem estática para compatibilidade
        fig.write_image(output_path)
        print(f"Gráfico estático salvo em: {output_path}")

    except Exception as e:
        print(f"Erro ao gerar gráfico para o canal {channel}: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_predictions_2024(data, predictions, channel, output_path, frequency='monthly'):
    """
    Gera gráfico focado nas previsões para 2024 usando Plotly.

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

        # Criar figura Plotly
        fig = go.Figure()

        # Converter índices Period para Timestamp para plotagem
        plot_data = data.copy()
        if isinstance(plot_data.index, pd.PeriodIndex):
            plot_data.index = plot_data.index.to_timestamp()

        # Filtrar dados de 2024
        data_2024 = plot_data[plot_data.index.year >= 2024]

        if not data_2024.empty:
            print(f"Dados históricos de 2024: {len(data_2024)} registros")
            print(f"Período: {data_2024.index[0]} até {data_2024.index[-1]}")

            fig.add_trace(go.Scatter(
                x=data_2024.index,
                y=data_2024['notification_count'],
                name='Dados Históricos 2024',
                mode='lines+markers',
                line=dict(color='#2c3e50', width=2),
                marker=dict(size=6)
            ))

        # Plotar previsões de cada modelo
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for (model_name, forecast), color in zip(predictions.items(), colors):
            if isinstance(forecast, pd.Series) and not forecast.empty:
                plot_forecast = forecast.copy()
                if isinstance(plot_forecast.index, pd.PeriodIndex):
                    plot_forecast.index = plot_forecast.index.to_timestamp()

                forecast_2024 = plot_forecast[plot_forecast.index.year >= 2024]
                if not forecast_2024.empty:
                    print(f"Plotando previsões do modelo {model_name} para 2024")
                    print(f"Período de previsão: {forecast_2024.index[0]} até {forecast_2024.index[-1]}")
                    print(f"Valores: min={forecast_2024.min():.2f}, max={forecast_2024.max():.2f}")

                    fig.add_trace(go.Scatter(
                        x=forecast_2024.index,
                        y=forecast_2024.values,
                        name=f'Previsão {model_name}',
                        mode='lines+markers',
                        line=dict(color=color, width=2, dash='dash'),
                        marker=dict(size=6, symbol='square')
                    ))

        # Configurar layout
        period = "Semanas" if frequency == "weekly" else "Meses"
        fig.update_layout(
            title=dict(
                text=f'Previsões 2024 - Canal {channel}',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title=f'Período ({period})',
            yaxis_title='Número de Notificações',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            width=1200,
            height=600,
            margin=dict(l=50, r=200, t=50, b=50),
            template='plotly_white'
        )

        # Criar diretório se não existir
        print(f"\nCriando diretório: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Salvar gráfico como HTML
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"Gráfico HTML salvo em: {html_path}")

        # Salvar também como imagem estática para compatibilidade
        fig.write_image(output_path)
        print(f"Gráfico estático salvo em: {output_path}")

    except Exception as e:
        print(f"Erro ao gerar gráfico 2024 para o canal {channel}: {str(e)}")
        import traceback
        traceback.print_exc()
