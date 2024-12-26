import plotly.graph_objects as go
import pandas as pd

def create_combined_visualization(data, predictions, year_filter=None):
    """
    Cria visualizações combinadas para todos os canais.
    """
    figures = []
    channels = data['channels'].unique()

    for channel in channels:
        try:
            # Filtrar dados do canal
            channel_data = data[data['channels'] == channel].copy()
            channel_data = channel_data.set_index('month')
            channel_data = channel_data.sort_index()

            # Filtrar por ano se necessário
            if year_filter:
                channel_data = channel_data[channel_data.index.year >= year_filter]
                if channel_data.empty:
                    print(f"Sem dados para o canal {channel} no ano {year_filter}")
                    continue

            # Criar figura
            fig = go.Figure()

            # Adicionar dados históricos
            fig.add_trace(go.Scatter(
                x=channel_data.index,
                y=channel_data['notification_count'],
                name='Dados Históricos',
                line=dict(color='black', width=2)
            ))

            # Adicionar previsões
            if channel in predictions and predictions[channel]:
                # Verificar se há alguma previsão válida
                valid_predictions = {k: v for k, v in predictions[channel].items()
                                  if isinstance(v, pd.Series) and not v.empty}

                if valid_predictions:
                    future_dates = pd.date_range(
                        start=channel_data.index[-1],
                        periods=len(next(iter(valid_predictions.values()))) + 1,
                        freq='MS'
                    )[1:]

                    colors = {
                        'Estatísticos': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # Azul, Laranja, Verde
                        'Machine Learning': ['#d62728', '#9467bd', '#8c564b'],  # Vermelho, Roxo, Marrom
                        'Deep Learning': ['#e377c2', '#7f7f7f']  # Rosa, Cinza
                    }

                    for model_name, forecast in valid_predictions.items():
                        try:
                            if 'Estatísticos' in model_name and colors['Estatísticos']:
                                color = colors['Estatísticos'].pop(0)
                            elif 'Machine Learning' in model_name and colors['Machine Learning']:
                                color = colors['Machine Learning'].pop(0)
                            elif 'Deep Learning' in model_name and colors['Deep Learning']:
                                color = colors['Deep Learning'].pop(0)
                            else:
                                continue

                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=forecast,
                                name=model_name,
                                line=dict(color=color, width=2, dash='dash')
                            ))
                            print(f"Adicionada previsão do modelo {model_name} para o canal {channel}")
                        except Exception as e:
                            print(f"Erro ao adicionar previsão do modelo {model_name}: {str(e)}")
                            continue
                else:
                    print(f"Nenhuma previsão válida encontrada para o canal {channel}")

            # Configurar layout
            title = f'Previsões para o Canal {channel}'
            if year_filter:
                title += f' ({year_filter})'

            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16)
                ),
                xaxis_title='Data',
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
                margin=dict(l=50, r=200, t=50, b=50)
            )

            figures.append(fig)
            print(f"Visualização criada com sucesso para o canal {channel}")

        except Exception as e:
            print(f"Erro ao criar visualização para o canal {channel}: {str(e)}")
            continue

    return figures
