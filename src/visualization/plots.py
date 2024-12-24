import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Definições globais
model_groups = {
    'Estatísticos': ['arima', 'sarima', 'ets'],
    'Machine Learning': ['random_forest', 'xgboost', 'lightgbm'],
    'Deep Learning': ['lstm', 'nbeats']
}

colors_prediction = {
    'arima': '#FF6B6B',      # Vermelho
    'sarima': '#4ECDC4',     # Turquesa
    'ets': '#45B7D1',        # Azul claro
    'random_forest': '#FFB347',  # Laranja
    'xgboost': '#98FB98',    # Verde claro
    'lightgbm': '#DDA0DD',   # Roxo claro
    'lstm': '#87CEEB',       # Azul céu
    'nbeats': '#F08080'      # Coral
}

def create_channel_visualization(df_notifications, channel, models, predictions, row=None, col=None):
    """
    Cria visualização combinada de dados históricos e previsões para cada canal.
    """
    available_models = {model_name: model for model_name, model in models.items() if model is not None}
    if not available_models:
        print(f"Sem modelos disponíveis para visualização no canal {channel}.")
        return None

    # Reset index e filtrar dados do canal
    df_channel = df_notifications.reset_index().copy()
    df_channel = df_channel[df_channel['channels'] == channel].copy()

    # Preparar dados por ano
    df_channel['year'] = df_channel['month'].dt.year
    df_channel['month_name'] = df_channel['month'].dt.strftime('%B')

    # Agrupar por mês e ano
    monthly_data = df_channel.groupby(['month', 'month_name', 'year'])['notification_count'].sum().reset_index()

    traces = []

    # Adicionar dados históricos por ano
    colors = {2023: 'rgba(53, 119, 233, 0.8)', 2024: 'rgba(44, 160, 44, 0.8)'}
    for year in sorted(monthly_data['year'].unique()):
        year_data = monthly_data[monthly_data['year'] == year]
        traces.append(go.Bar(
            name=str(year),
            x=year_data['month_name'],
            y=year_data['notification_count'],
            marker_color=colors.get(year, 'gray'),
            text=year_data['notification_count'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto',
            showlegend=True  # Mostrar legenda para cada gráfico
        ))

    # Adicionar previsões por grupo
    for group_name, model_names in model_groups.items():
        for model_name in model_names:
            if model_name in predictions and model_name != 'dates':
                traces.append(go.Scatter(
                    x=[d.strftime('%B') for d in predictions['dates']],
                    y=predictions[model_name],
                    name=f'{group_name} - {model_name.upper()}',
                    line=dict(
                        color=colors_prediction.get(model_name, 'black'),
                        dash='dash',
                        width=2
                    ),
                    mode='lines',
                    showlegend=True,  # Mostrar legenda para cada gráfico
                    legendgroup=f"{channel}_{group_name}"  # Grupo único por canal
                ))

    return traces

def create_total_notifications_visualization(df_notifications, prediction_models, future_predictions):
    """
    Cria visualização dos totais de todas as notificações combinadas.
    """
    df_all = df_notifications.reset_index().copy()
    df_all['year'] = df_all['month'].dt.year
    df_all['month_name'] = df_all['month'].dt.strftime('%B')

    # Calcular totais mensais
    monthly_totals = df_all.groupby(['month', 'month_name', 'year'])['notification_count'].sum().reset_index()

    traces = []

    # Adicionar dados históricos por ano
    colors = {2023: 'rgba(53, 119, 233, 0.8)', 2024: 'rgba(44, 160, 44, 0.8)'}
    for year in sorted(monthly_totals['year'].unique()):
        year_data = monthly_totals[monthly_totals['year'] == year]
        traces.append(go.Bar(
            name=str(year),
            x=year_data['month_name'],
            y=year_data['notification_count'],
            marker_color=colors.get(year, 'gray'),
            text=year_data['notification_count'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto',
        ))

    # Adicionar previsões totais
    if future_predictions:
        total_predictions = _calculate_total_predictions(future_predictions)
        traces.extend(_create_total_predictions_traces(total_predictions))

    return traces

def create_combined_visualization(df_notifications, prediction_models, future_predictions, year_filter=None):
    """
    Cria uma visualização combinada com todos os gráficos em um único HTML.
    """
    if year_filter:
        df_notifications = df_notifications[df_notifications.index.year == year_filter].copy()

    channels = sorted(df_notifications['channels'].unique())
    n_channels = len(channels)
    n_cols = 1
    n_rows = n_channels + 1

    # Criar subplots com grid_pattern
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Canal: {channel.upper()}" for channel in channels] + ["Total de Notificações"],
        specs=[[{"secondary_y": True}] for _ in range(n_rows)],  # Habilitar eixo y secundário
        vertical_spacing=0.15/n_rows  # Ajuste dinâmico baseado no número de gráficos
    )

    # Adicionar visualizações por canal
    for i, channel in enumerate(channels):
        row = i + 1
        col = 1

        channel_traces = create_channel_visualization(
            df_notifications,
            channel,
            prediction_models.get(channel, {}),
            future_predictions.get(channel, {}),
            row=row,
            col=col
        )

        if channel_traces:
            for trace in channel_traces:
                fig.add_trace(trace, row=row, col=col)

    # Adicionar visualização total
    total_traces = create_total_notifications_visualization(
        df_notifications,
        prediction_models,
        future_predictions
    )

    for trace in total_traces:
        fig.add_trace(trace, row=n_rows, col=1)

    # Atualizar layout
    title_suffix = f" - {year_filter}" if year_filter else ""
    fig.update_layout(
        title={
            'text': f"Análise de Notificações{title_suffix}",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        width=1200,
        height=400 * n_rows,  # Altura proporcional ao número de gráficos
        template="plotly_white",
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # Configurar cada subplot individualmente
    for i in range(1, n_rows + 1):
        # Configurar eixo X
        fig.update_xaxes(
            title_text="Mês",
            row=i,
            col=1,
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor='rgba(255, 255, 255, 0.8)',
                y=1.1
            )
        )

        # Configurar eixo Y
        fig.update_yaxes(
            title_text="Número de Notificações",
            row=i,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            rangemode='tozero'
        )

        # Adicionar legenda para cada subplot
        fig.update_layout({
            f'xaxis{i}_rangeslider_visible': False,
            f'legend{i}': dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ) if i < n_rows else dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            )
        })

    return fig

def _calculate_total_predictions(future_predictions):
    """
    Calcula as previsões totais somando todos os canais.
    """
    # Pegar o primeiro canal para inicializar as datas
    first_channel = next(iter(future_predictions.values()))

    # Inicializar dicionário com as datas
    total_predictions = {
        'dates': first_channel['dates']
    }

    # Lista de todos os modelos possíveis
    model_names = ['arima', 'sarima', 'ets', 'random_forest', 'xgboost',
                  'lightgbm', 'lstm', 'nbeats']

    # Inicializar arrays para cada modelo
    for model in model_names:
        if model in first_channel and isinstance(first_channel[model], (list, np.ndarray)):
            total_predictions[model] = np.zeros_like(first_channel[model])

    # Somar previsões de todos os canais
    for channel_predictions in future_predictions.values():
        for model in model_names:
            if model in channel_predictions and model in total_predictions:
                try:
                    total_predictions[model] += channel_predictions[model]
                except Exception as e:
                    print(f"Erro ao somar previsões do modelo {model}: {e}")

    return total_predictions

def _create_total_predictions_traces(total_predictions):
    """
    Cria as traces de previsão total.
    """
    traces = []
    colors = {
        'arima': '#FF6B6B',      # Vermelho
        'sarima': '#4ECDC4',     # Turquesa
        'ets': '#45B7D1',        # Azul claro
        'random_forest': '#FFB347',  # Laranja
        'xgboost': '#98FB98',    # Verde claro
        'lightgbm': '#DDA0DD',   # Roxo claro
        'lstm': '#87CEEB',       # Azul céu
        'nbeats': '#F08080'      # Coral
    }

    # Agrupar modelos por categoria
    model_groups = {
        'Estatísticos': ['arima', 'sarima', 'ets'],
        'Machine Learning': ['random_forest', 'xgboost', 'lightgbm'],
        'Deep Learning': ['lstm', 'nbeats']
    }

    # Criar traces para cada modelo por grupo
    for group_name, models in model_groups.items():
        for model_name in models:
            if (model_name in total_predictions and
                isinstance(total_predictions[model_name], (list, np.ndarray)) and
                len(total_predictions[model_name]) > 0):

                months = [d.strftime('%B') for d in total_predictions['dates']]
                traces.append(go.Scatter(
                    x=months,
                    y=total_predictions[model_name],
                    name=f'{group_name} - {model_name.upper()}',
                    line=dict(
                        color=colors.get(model_name, 'gray'),
                        dash='dash',
                        width=2
                    ),
                    mode='lines',
                    legendgroup=group_name
                ))

    return traces
