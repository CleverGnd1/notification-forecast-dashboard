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

def create_individual_channel_plot(df_notifications, channel, models, predictions):
    """
    Cria um gráfico individual para um canal específico.
    """
    # Reset index e filtrar dados do canal
    df_channel = df_notifications.reset_index().copy()
    df_channel = df_channel[df_channel['channels'] == channel].copy()

    # Preparar dados por ano
    df_channel['year'] = df_channel['month'].dt.year
    df_channel['month_name'] = df_channel['month'].dt.strftime('%B')

    # Agrupar por mês e ano
    monthly_data = df_channel.groupby(['month', 'month_name', 'year'])['notification_count'].sum().reset_index()

    # Criar figura
    fig = go.Figure()

    # Adicionar dados históricos por ano
    colors = {2023: 'rgba(53, 119, 233, 0.8)', 2024: 'rgba(44, 160, 44, 0.8)'}
    for year in sorted(monthly_data['year'].unique()):
        year_data = monthly_data[monthly_data['year'] == year]
        fig.add_trace(go.Bar(
            name=str(year),
            x=year_data['month_name'],
            y=year_data['notification_count'],
            marker_color=colors.get(year, 'gray'),
            text=year_data['notification_count'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto'
        ))

    # Adicionar previsões por grupo
    if predictions:
        for group_name, model_names in model_groups.items():
            for model_name in model_names:
                if model_name in predictions and model_name != 'dates':
                    fig.add_trace(go.Scatter(
                        x=[d.strftime('%B') for d in predictions['dates']],
                        y=predictions[model_name],
                        name=f'{group_name} - {model_name.upper()}',
                        line=dict(
                            color=colors_prediction.get(model_name, 'black'),
                            dash='dash',
                            width=2
                        ),
                        mode='lines'
                    ))

    # Atualizar layout
    fig.update_layout(
        title={
            'text': f"Canal: {channel.upper()}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='black', family='Arial Bold')
        },
        width=1400,
        height=500,
        template="plotly_white",
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(t=100)  # Aumentar margem superior para acomodar a legenda
    )

    # Configurar eixos
    fig.update_xaxes(
        title_text="Mês",
        tickangle=45,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )

    fig.update_yaxes(
        title_text="Número de Notificações",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        rangemode='tozero'
    )

    return fig

def create_total_plot(df_notifications, prediction_models, future_predictions):
    """
    Cria o gráfico do total de notificações.
    """
    df_all = df_notifications.reset_index().copy()
    df_all['year'] = df_all['month'].dt.year
    df_all['month_name'] = df_all['month'].dt.strftime('%B')

    # Calcular totais mensais
    monthly_totals = df_all.groupby(['month', 'month_name', 'year'])['notification_count'].sum().reset_index()

    # Criar figura
    fig = go.Figure()

    # Adicionar dados históricos por ano
    colors = {2023: 'rgba(53, 119, 233, 0.8)', 2024: 'rgba(44, 160, 44, 0.8)'}
    for year in sorted(monthly_totals['year'].unique()):
        year_data = monthly_totals[monthly_totals['year'] == year]
        fig.add_trace(go.Bar(
            name=str(year),
            x=year_data['month_name'],
            y=year_data['notification_count'],
            marker_color=colors.get(year, 'gray'),
            text=year_data['notification_count'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto'
        ))

    # Adicionar previsões totais
    if future_predictions:
        total_predictions = _calculate_total_predictions(future_predictions)
        for group_name, model_names in model_groups.items():
            for model_name in model_names:
                if (model_name in total_predictions and
                    isinstance(total_predictions[model_name], (list, np.ndarray)) and
                    len(total_predictions[model_name]) > 0):

                    months = [d.strftime('%B') for d in total_predictions['dates']]
                    fig.add_trace(go.Scatter(
                        x=months,
                        y=total_predictions[model_name],
                        name=f'{group_name} - {model_name.upper()}',
                        line=dict(
                            color=colors_prediction.get(model_name, 'gray'),
                            dash='dash',
                            width=2
                        ),
                        mode='lines'
                    ))

    # Atualizar layout
    fig.update_layout(
        title={
            'text': "Total de Notificações",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='black', family='Arial Bold')
        },
        width=1400,
        height=500,
        template="plotly_white",
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(t=100)  # Aumentar margem superior para acomodar a legenda
    )

    # Configurar eixos
    fig.update_xaxes(
        title_text="Mês",
        tickangle=45,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )

    fig.update_yaxes(
        title_text="Número de Notificações",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        rangemode='tozero'
    )

    return fig

def create_combined_visualization(df_notifications, prediction_models, future_predictions, year_filter=None):
    """
    Cria visualizações individuais para cada canal e total.
    Retorna uma lista de figuras que podem ser organizadas no HTML.
    """
    if year_filter:
        df_notifications = df_notifications[df_notifications.index.year == year_filter].copy()

    channels = sorted(df_notifications['channels'].unique())
    figures = []

    # Criar gráficos individuais para cada canal
    for channel in channels:
        fig = create_individual_channel_plot(
            df_notifications,
            channel,
            prediction_models.get(channel, {}),
            future_predictions.get(channel, {})
        )
        figures.append(fig)

    # Criar gráfico total
    total_fig = create_total_plot(df_notifications, prediction_models, future_predictions)
    figures.append(total_fig)

    return figures

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
