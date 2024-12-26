import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_max_value(predictions):
    """Helper function to get maximum value from predictions dict"""
    max_val = 0
    for pred in predictions.values():
        if isinstance(pred, pd.Series):
            max_val = max(max_val, pred.max())
        elif isinstance(pred, (int, float)):
            max_val = max(max_val, pred)
    return max_val

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

        # Separar dados por ano
        data_2023 = plot_data[plot_data.index.astype(str).str.startswith('2023')]
        data_2024 = plot_data[plot_data.index.astype(str).str.startswith('2024')]

        # Função auxiliar para extrair mês do índice
        def get_month(idx):
            return pd.to_datetime(str(idx)).month

        # Preparar dados mensais para 2023
        y_2023 = []
        for m in range(1, 13):
            month_data = data_2023[data_2023.index.map(get_month) == m]
            y_2023.append(month_data['notification_count'].iloc[0] if not month_data.empty else None)

        # Preparar dados mensais para 2024
        y_2024 = []
        for m in range(1, 13):
            month_data = data_2024[data_2024.index.map(get_month) == m]
            y_2024.append(month_data['notification_count'].iloc[0] if not month_data.empty else None)

        # Adicionar barras para dados históricos
        months = [f"{m:02d}" for m in range(1, 13)]
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

        # Barras para 2023
        fig.add_trace(go.Bar(
            x=months,
            y=y_2023,
            name='2023',
            width=0.35,
            offset=-0.2,
            marker_color='rgba(52, 152, 219, 0.7)',
            hovertemplate="Mês: %{text}<br>Valor: %{y:,.0f}<extra>2023</extra>",
            text=month_names
        ))

        # Barras para 2024
        fig.add_trace(go.Bar(
            x=months,
            y=y_2024,
            name='2024',
            width=0.35,
            offset=0.2,
            marker_color='rgba(46, 204, 113, 0.7)',
            hovertemplate="Mês: %{text}<br>Valor: %{y:,.0f}<extra>2024</extra>",
            text=month_names
        ))

        # Calcular valor máximo para o eixo y
        max_value = max(
            max(filter(None, y_2023)) if y_2023 else 0,
            max(filter(None, y_2024)) if y_2024 else 0,
            get_max_value(predictions)
        )

        # Adicionar linha vertical separando dados históricos de previsões
        fig.add_shape(
            type="line",
            x0="12",
            x1="12",
            y0=0,
            y1=max_value * 1.1,
            line=dict(
                color="rgba(128, 128, 128, 0.3)",
                width=2,
                dash="dash",
            ),
            layer="below"
        )

        # Plotar previsões de cada modelo
        colors = ['#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']
        for (model_name, forecast), color in zip(predictions.items(), colors):
            if isinstance(forecast, pd.Series) and not forecast.empty:
                plot_forecast = forecast.copy()
                if isinstance(plot_forecast.index, pd.PeriodIndex):
                    plot_forecast.index = plot_forecast.index.to_timestamp()

                # Filtrar apenas previsões para 2025
                forecast_2025 = plot_forecast[plot_forecast.index.astype(str).str.startswith('2025')]

                if not forecast_2025.empty:
                    print(f"\nProcessando previsões do modelo {model_name}")
                    print(f"Período de previsão: {forecast_2025.index[0]} até {forecast_2025.index[-1]}")
                    print(f"Valores: min={forecast_2025.min():.2f}, max={forecast_2025.max():.2f}")

                    # Converter para formato mensal
                    monthly_data = []
                    for month in range(1, 13):
                        month_data = forecast_2025[forecast_2025.index.map(get_month) == month]
                        if not month_data.empty:
                            monthly_data.append(month_data.iloc[0])
                        else:
                            monthly_data.append(None)

                    fig.add_trace(go.Scatter(
                        x=months,
                        y=monthly_data,
                        name=f'Previsão 2025 - {model_name}',
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(size=8, symbol='diamond'),
                        hovertemplate="Mês: %{text}<br>Previsão: %{y:,.0f}<extra>" + model_name + "</extra>",
                        text=month_names
                    ))

        # Configurar layout
        fig.update_layout(
            title=dict(
                text=f'Análise de Notificações - Canal {channel}',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title='Mês',
            yaxis_title='Número de Notificações',
            xaxis=dict(
                ticktext=month_names,
                tickvals=months,
                tickmode='array',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            width=1200,
            height=600,
            margin=dict(l=50, r=200, t=50, b=50),
            template='plotly_white',
            barmode='group',
            plot_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            annotations=[
                dict(
                    x="12",
                    y=max_value * 1.05,
                    text="Início das Previsões",
                    showarrow=False,
                    font=dict(size=10, color="rgba(128, 128, 128, 0.8)")
                )
            ]
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

        # Função auxiliar para extrair mês do índice
        def get_month(idx):
            return pd.to_datetime(str(idx)).month

        # Filtrar dados de 2024
        data_2024 = plot_data[plot_data.index.astype(str).str.startswith('2024')]

        # Preparar dados mensais para 2024
        months = [f"{m:02d}" for m in range(1, 13)]
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        y_2024 = []

        if not data_2024.empty:
            print(f"Dados históricos de 2024: {len(data_2024)} registros")
            print(f"Período: {data_2024.index[0]} até {data_2024.index[-1]}")

            for m in range(1, 13):
                month_data = data_2024[data_2024.index.map(get_month) == m]
                y_2024.append(month_data['notification_count'].iloc[0] if not month_data.empty else None)

            fig.add_trace(go.Bar(
                x=months,
                y=y_2024,
                name='2024',
                marker_color='rgba(46, 204, 113, 0.7)',
                hovertemplate="Mês: %{text}<br>Valor: %{y:,.0f}<extra>2024</extra>",
                text=month_names
            ))

        # Calcular valor máximo para o eixo y
        max_value = max(
            max(filter(None, y_2024)) if y_2024 else 0,
            get_max_value(predictions)
        )

        # Adicionar linha vertical separando dados históricos de previsões
        fig.add_shape(
            type="line",
            x0="12",
            x1="12",
            y0=0,
            y1=max_value * 1.1,
            line=dict(
                color="rgba(128, 128, 128, 0.3)",
                width=2,
                dash="dash",
            ),
            layer="below"
        )

        # Plotar previsões de cada modelo
        colors = ['#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']
        for (model_name, forecast), color in zip(predictions.items(), colors):
            if isinstance(forecast, pd.Series) and not forecast.empty:
                plot_forecast = forecast.copy()
                if isinstance(plot_forecast.index, pd.PeriodIndex):
                    plot_forecast.index = plot_forecast.index.to_timestamp()

                # Filtrar previsões para 2024
                forecast_2024 = plot_forecast[plot_forecast.index.astype(str).str.startswith('2024')]

                if not forecast_2024.empty:
                    print(f"Plotando previsões do modelo {model_name} para 2024")
                    print(f"Período de previsão: {forecast_2024.index[0]} até {forecast_2024.index[-1]}")
                    print(f"Valores: min={forecast_2024.min():.2f}, max={forecast_2024.max():.2f}")

                    # Converter para formato mensal
                    monthly_data = []
                    for month in range(1, 13):
                        month_data = forecast_2024[forecast_2024.index.map(get_month) == month]
                        if not month_data.empty:
                            monthly_data.append(month_data.iloc[0])
                        else:
                            monthly_data.append(None)

                    fig.add_trace(go.Scatter(
                        x=months,
                        y=monthly_data,
                        name=f'Previsão - {model_name}',
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(size=8, symbol='diamond'),
                        hovertemplate="Mês: %{text}<br>Previsão: %{y:,.0f}<extra>" + model_name + "</extra>",
                        text=month_names
                    ))

        # Configurar layout
        fig.update_layout(
            title=dict(
                text=f'Previsões 2024 - Canal {channel}',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title='Mês',
            yaxis_title='Número de Notificações',
            xaxis=dict(
                ticktext=month_names,
                tickvals=months,
                tickmode='array',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            width=1200,
            height=600,
            margin=dict(l=50, r=200, t=50, b=50),
            template='plotly_white',
            plot_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            annotations=[
                dict(
                    x="12",
                    y=max_value * 1.05,
                    text="Início das Previsões",
                    showarrow=False,
                    font=dict(size=10, color="rgba(128, 128, 128, 0.8)")
                )
            ]
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
