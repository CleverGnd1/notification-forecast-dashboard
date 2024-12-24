import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

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
            showlegend=row == 1 and col == 1  # Mostrar legenda apenas para o primeiro gráfico
        ))

    # Adicionar previsões
    colors_prediction = {
        # Modelos Estatísticos Clássicos
        'arima': '#FF6B6B',      # Vermelho
        'sarima': '#4ECDC4',     # Turquesa
        'ets': '#45B7D1',        # Azul claro

        # Modelos de Machine Learning
        'random_forest': '#FFB347',  # Laranja
        'xgboost': '#98FB98',     # Verde claro
        'lightgbm': '#DDA0DD',    # Roxo claro

        # Modelos de Deep Learning
        'lstm': '#87CEEB',       # Azul céu
        'nbeats': '#F08080'      # Coral
    }

    # Agrupar modelos por categoria
    model_groups = {
        'Estatísticos': ['arima', 'sarima', 'ets'],
        'Machine Learning': ['random_forest', 'xgboost', 'lightgbm'],
        'Deep Learning': ['lstm', 'nbeats']
    }

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
                    showlegend=row == 1 and col == 1,  # Mostrar legenda apenas para o primeiro gráfico
                    legendgroup=group_name  # Agrupar na legenda por categoria
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
    # Filtrar dados por ano se especificado
    if year_filter:
        df_notifications = df_notifications[df_notifications.index.year == year_filter].copy()

    channels = df_notifications['channels'].unique()
    n_channels = len(channels)

    # Calcular número de linhas e colunas para o layout
    n_cols = min(2, n_channels)
    n_rows = (n_channels + 1) // 2 + 1  # +1 para o gráfico total

    # Criar subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[*channels, "Total de Notificações"],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Adicionar visualizações por canal
    for i, channel in enumerate(channels):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

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

    # Adicionar visualização total na última linha
    total_row = n_rows
    total_col = 1 if n_channels % 2 == 0 else (n_channels % 2)

    total_traces = create_total_notifications_visualization(
        df_notifications,
        prediction_models,
        future_predictions
    )

    for trace in total_traces:
        fig.add_trace(trace, row=total_row, col=total_col)

    # Atualizar layout
    title_suffix = f" - {year_filter}" if year_filter else ""
    fig.update_layout(
        title={
            'text': f"Análise de Notificações{title_suffix}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        width=1500,
        height=400 * n_rows,
        barmode='group',
        legend=dict(
            groupclick="toggleitem",  # Permite clicar no grupo para mostrar/esconder todos os itens
            tracegroupgap=10,         # Espaço entre grupos na legenda
        ),
        template="plotly_white",      # Tema mais limpo
        font=dict(size=12)            # Tamanho da fonte
    )

    # Atualizar eixos
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_xaxes(title_text="Mês", row=i, col=j)
            fig.update_yaxes(title_text="Número de Notificações", row=i, col=j)

    return fig

def _calculate_total_predictions(future_predictions):
    """Calcula as previsões totais somando todos os canais."""
    first_predictions = next(iter(future_predictions.values()))
    total_predictions = {
        'dates': first_predictions['dates'],
        'arima': np.zeros(len(first_predictions.get('arima', []))),
        'sarima': np.zeros(len(first_predictions.get('sarima', []))),
        'ets': np.zeros(len(first_predictions.get('ets', [])))
    }

    for channel_predictions in future_predictions.values():
        for model_name in ['arima', 'sarima', 'ets']:
            if model_name in channel_predictions:
                total_predictions[model_name] += channel_predictions[model_name]

    return total_predictions

def _create_total_predictions_traces(total_predictions):
    """Cria as traces de previsão total."""
    traces = []
    colors = {
        'arima': '#FF6B6B',
        'sarima': '#4ECDC4',
        'ets': '#45B7D1'
    }

    for model_name in ['arima', 'sarima', 'ets']:
        if model_name in total_predictions and len(total_predictions[model_name]) > 0:
            months = [d.strftime('%B') for d in total_predictions['dates']]
            traces.append(go.Scatter(
                x=months,
                y=total_predictions[model_name],
                name=f'Previsão - {model_name.upper()}',
                line=dict(color=colors[model_name], dash='dash', width=2),
                mode='lines'
            ))

    return traces
