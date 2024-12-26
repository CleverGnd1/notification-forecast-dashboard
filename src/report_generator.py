import os
from visualization.plot_utils import plot_predictions, plot_predictions_2024

def generate_html_report(data, predictions, frequency='monthly'):
    """
    Gera relatório HTML com os resultados da análise.

    Args:
        data (dict): Dicionário com os dados históricos por canal
        predictions (dict): Dicionário com as previsões por canal
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
    """
    try:
        print("\nIniciando geração do relatório HTML...")
        print(f"Frequência dos dados: {frequency}")
        print(f"Canais disponíveis: {', '.join(data.keys())}")
        print(f"Canais com previsões: {', '.join(predictions.keys())}")

        # Criar diretório de saída se não existir
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/plots", exist_ok=True)

        # Gerar gráficos para cada canal e total
        print("\nGerando gráficos...")
        for channel in data.keys():
            if channel in predictions and predictions[channel]:
                print(f"\nProcessando gráficos para o canal {channel}...")

                # Gerar apenas o gráfico completo
                plot_path = f"output/plots/plot_{channel}.png"
                print(f"Gerando gráfico completo em {plot_path}")
                plot_predictions(data[channel], predictions[channel], channel, plot_path, frequency)
            else:
                print(f"Aviso: Canal {channel} não possui previsões")

        # Gerar HTML
        print("\nGerando arquivo HTML...")
        import pandas as pd
        current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análise de Previsões de Notificações</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f0f2f5;
                    color: #1a1a1a;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 40px auto;
                    padding: 30px;
                }}
                .header {{
                    background-color: #ffffff;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                h1 {{
                    color: #1a73e8;
                    margin: 0;
                    font-size: 2.5em;
                    text-align: center;
                }}
                .summary {{
                    background-color: #ffffff;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                .summary h2 {{
                    color: #1a73e8;
                    margin-top: 0;
                }}
                .channel-section {{
                    background-color: #ffffff;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                .channel-section h2 {{
                    color: #1a73e8;
                    margin-top: 0;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #e8eaed;
                }}
                .plot-container {{
                    margin: 25px 0;
                    width: 100%;
                    height: 600px;
                }}
                .plot-container h3 {{
                    color: #5f6368;
                    margin: 15px 0;
                }}
                .stats-section {{
                    margin: 30px 0;
                }}
                .stats-section h3 {{
                    color: #1a73e8;
                    margin: 25px 0 15px 0;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #e8eaed;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s;
                }}
                .stat-card:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .stat-card h4 {{
                    color: #5f6368;
                    margin: 0 0 15px 0;
                    font-size: 1.1em;
                }}
                .stat-value {{
                    font-size: 1.8em;
                    color: #1a73e8;
                    font-weight: bold;
                    margin-bottom: 8px;
                }}
                .stat-period {{
                    color: #5f6368;
                    font-size: 0.9em;
                    margin-top: 5px;
                }}
                .stat-model {{
                    color: #34a853;
                    font-size: 0.9em;
                    margin-top: 5px;
                    font-style: italic;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #5f6368;
                    font-size: 0.9em;
                    background-color: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                iframe {{
                    width: 100%;
                    height: 100%;
                    border: none;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Relatório de Previsões de Notificações</h1>
                </div>
        """

        # Adicionar seção de total primeiro
        if 'total' in data and 'total' in predictions:
            total_data = data['total']
            # Separar dados por ano
            total_data_2023 = total_data[total_data.index.astype(str).str.startswith('2023')]
            total_data_2024 = total_data[total_data.index.astype(str).str.startswith('2024')]

            # Estatísticas por ano
            total_stats = {
                '2023': {
                    'total': total_data_2023['notification_count'].sum(),
                    'mean': total_data_2023['notification_count'].mean(),
                    'max_value': total_data_2023['notification_count'].max(),
                    'max_period': total_data_2023['notification_count'].idxmax(),
                    'min_value': total_data_2023['notification_count'].min(),
                    'min_period': total_data_2023['notification_count'].idxmin(),
                } if not total_data_2023.empty else {},
                '2024': {
                    'total': total_data_2024['notification_count'].sum(),
                    'mean': total_data_2024['notification_count'].mean(),
                    'max_value': total_data_2024['notification_count'].max(),
                    'max_period': total_data_2024['notification_count'].idxmax(),
                    'min_value': total_data_2024['notification_count'].min(),
                    'min_period': total_data_2024['notification_count'].idxmin(),
                } if not total_data_2024.empty else {}
            }

            # Calcular estatísticas das previsões para 2025
            if 'total' in predictions:
                pred_2025 = {}
                max_pred = float('-inf')
                min_pred = float('inf')
                max_model = ''
                min_model = ''
                total_pred = 0
                count_models = 0

                for model, pred_data in predictions['total'].items():
                    pred_2025_data = pred_data[pred_data.index.astype(str).str.startswith('2025')]
                    if not pred_2025_data.empty:
                        model_sum = pred_2025_data.sum()
                        total_pred += model_sum
                        count_models += 1

                        model_max = pred_2025_data.max()
                        if model_max > max_pred:
                            max_pred = model_max
                            max_model = model
                            max_period = pred_2025_data.idxmax()

                        model_min = pred_2025_data.min()
                        if model_min < min_pred:
                            min_pred = model_min
                            min_model = model
                            min_period = pred_2025_data.idxmin()

                if count_models > 0:
                    pred_2025 = {
                        'total_predicted': total_pred / count_models,
                        'max_value': max_pred,
                        'max_period': max_period,
                        'max_model': max_model,
                        'min_value': min_pred,
                        'min_period': min_period,
                        'min_model': min_model
                    }

            template += f"""
            <div class="channel-section">
                <h2>Total de Notificações (Todos os Canais)</h2>
                <div class="stats-section">
                    <h3>Estatísticas 2023</h3>
                    <div class="stats">
                        <div class="stat-card">
                            <h4>Total de Notificações</h4>
                            <div class="stat-value">{f"{total_stats['2023'].get('total'):,.0f}" if total_stats['2023'].get('total') is not None else 'N/A'}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Média de Notificações</h4>
                            <div class="stat-value">{f"{total_stats['2023'].get('mean'):,.0f}" if total_stats['2023'].get('mean') is not None else 'N/A'}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Máximo e Período</h4>
                            <div class="stat-value">{f"{total_stats['2023'].get('max_value'):,.0f}" if total_stats['2023'].get('max_value') is not None else 'N/A'}</div>
                            <div class="stat-period">{total_stats['2023'].get('max_period', 'N/A')}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Mínimo e Período</h4>
                            <div class="stat-value">{f"{total_stats['2023'].get('min_value'):,.0f}" if total_stats['2023'].get('min_value') is not None else 'N/A'}</div>
                            <div class="stat-period">{total_stats['2023'].get('min_period', 'N/A')}</div>
                        </div>
                    </div>

                    <h3>Estatísticas 2024</h3>
                    <div class="stats">
                        <div class="stat-card">
                            <h4>Total de Notificações</h4>
                            <div class="stat-value">{f"{total_stats['2024'].get('total'):,.0f}" if total_stats['2024'].get('total') is not None else 'N/A'}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Média de Notificações</h4>
                            <div class="stat-value">{f"{total_stats['2024'].get('mean'):,.0f}" if total_stats['2024'].get('mean') is not None else 'N/A'}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Máximo e Período</h4>
                            <div class="stat-value">{f"{total_stats['2024'].get('max_value'):,.0f}" if total_stats['2024'].get('max_value') is not None else 'N/A'}</div>
                            <div class="stat-period">{total_stats['2024'].get('max_period', 'N/A')}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Mínimo e Período</h4>
                            <div class="stat-value">{f"{total_stats['2024'].get('min_value'):,.0f}" if total_stats['2024'].get('min_value') is not None else 'N/A'}</div>
                            <div class="stat-period">{total_stats['2024'].get('min_period', 'N/A')}</div>
                        </div>
                    </div>

                    <h3>Previsões 2025</h3>
                    <div class="stats">
                        <div class="stat-card">
                            <h4>Total Previsto (Média dos Modelos)</h4>
                            <div class="stat-value">{f"{pred_2025.get('total_predicted'):,.0f}" if pred_2025.get('total_predicted') is not None else 'N/A'}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Máximo Previsto</h4>
                            <div class="stat-value">{f"{pred_2025.get('max_value'):,.0f}" if pred_2025.get('max_value') is not None else 'N/A'}</div>
                            <div class="stat-period">{pred_2025.get('max_period', 'N/A')}</div>
                            <div class="stat-model">Modelo: {pred_2025.get('max_model', 'N/A')}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Mínimo Previsto</h4>
                            <div class="stat-value">{f"{pred_2025.get('min_value'):,.0f}" if pred_2025.get('min_value') is not None else 'N/A'}</div>
                            <div class="stat-period">{pred_2025.get('min_period', 'N/A')}</div>
                            <div class="stat-model">Modelo: {pred_2025.get('min_model', 'N/A')}</div>
                        </div>
                    </div>
                </div>
                <div class="plot-container">
                    <h3>Histórico e Previsões Completas - Total</h3>
                    <iframe src="plots/plot_total.html"></iframe>
                </div>
            </div>
            """

        # Adicionar seções para cada canal (exceto total)
        for channel in data.keys():
            if channel != 'total' and channel in predictions and predictions[channel]:
                channel_data = data[channel]
                # Separar dados por ano
                channel_data_2023 = channel_data[channel_data.index.astype(str).str.startswith('2023')]
                channel_data_2024 = channel_data[channel_data.index.astype(str).str.startswith('2024')]

                # Estatísticas por ano
                channel_stats = {
                    '2023': {
                        'total': channel_data_2023['notification_count'].sum(),
                        'mean': channel_data_2023['notification_count'].mean(),
                        'max_value': channel_data_2023['notification_count'].max(),
                        'max_period': channel_data_2023['notification_count'].idxmax(),
                        'min_value': channel_data_2023['notification_count'].min(),
                        'min_period': channel_data_2023['notification_count'].idxmin(),
                    } if not channel_data_2023.empty else {},
                    '2024': {
                        'total': channel_data_2024['notification_count'].sum(),
                        'mean': channel_data_2024['notification_count'].mean(),
                        'max_value': channel_data_2024['notification_count'].max(),
                        'max_period': channel_data_2024['notification_count'].idxmax(),
                        'min_value': channel_data_2024['notification_count'].min(),
                        'min_period': channel_data_2024['notification_count'].idxmin(),
                    } if not channel_data_2024.empty else {}
                }

                # Calcular estatísticas das previsões para 2025
                pred_2025 = {}
                if channel in predictions:
                    max_pred = float('-inf')
                    min_pred = float('inf')
                    max_model = ''
                    min_model = ''
                    total_pred = 0
                    count_models = 0

                    for model, pred_data in predictions[channel].items():
                        pred_2025_data = pred_data[pred_data.index.astype(str).str.startswith('2025')]
                        if not pred_2025_data.empty:
                            model_sum = pred_2025_data.sum()
                            total_pred += model_sum
                            count_models += 1

                            model_max = pred_2025_data.max()
                            if model_max > max_pred:
                                max_pred = model_max
                                max_model = model
                                max_period = pred_2025_data.idxmax()

                            model_min = pred_2025_data.min()
                            if model_min < min_pred:
                                min_pred = model_min
                                min_model = model
                                min_period = pred_2025_data.idxmin()

                    if count_models > 0:
                        pred_2025 = {
                            'total_predicted': total_pred / count_models,
                            'max_value': max_pred,
                            'max_period': max_period,
                            'max_model': max_model,
                            'min_value': min_pred,
                            'min_period': min_period,
                            'min_model': min_model
                        }

                template += f"""
                <div class="channel-section">
                    <h2>Canal: {channel}</h2>
                    <div class="stats-section">
                        <h3>Estatísticas 2023</h3>
                        <div class="stats">
                            <div class="stat-card">
                                <h4>Total de Notificações</h4>
                                <div class="stat-value">{f"{channel_stats['2023'].get('total'):,.0f}" if channel_stats['2023'].get('total') is not None else 'N/A'}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Média de Notificações</h4>
                                <div class="stat-value">{f"{channel_stats['2023'].get('mean'):,.0f}" if channel_stats['2023'].get('mean') is not None else 'N/A'}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Máximo e Período</h4>
                                <div class="stat-value">{f"{channel_stats['2023'].get('max_value'):,.0f}" if channel_stats['2023'].get('max_value') is not None else 'N/A'}</div>
                                <div class="stat-period">{channel_stats['2023'].get('max_period', 'N/A')}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Mínimo e Período</h4>
                                <div class="stat-value">{f"{channel_stats['2023'].get('min_value'):,.0f}" if channel_stats['2023'].get('min_value') is not None else 'N/A'}</div>
                                <div class="stat-period">{channel_stats['2023'].get('min_period', 'N/A')}</div>
                            </div>
                        </div>

                        <h3>Estatísticas 2024</h3>
                        <div class="stats">
                            <div class="stat-card">
                                <h4>Total de Notificações</h4>
                                <div class="stat-value">{f"{channel_stats['2024'].get('total'):,.0f}" if channel_stats['2024'].get('total') is not None else 'N/A'}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Média de Notificações</h4>
                                <div class="stat-value">{f"{channel_stats['2024'].get('mean'):,.0f}" if channel_stats['2024'].get('mean') is not None else 'N/A'}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Máximo e Período</h4>
                                <div class="stat-value">{f"{channel_stats['2024'].get('max_value'):,.0f}" if channel_stats['2024'].get('max_value') is not None else 'N/A'}</div>
                                <div class="stat-period">{channel_stats['2024'].get('max_period', 'N/A')}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Mínimo e Período</h4>
                                <div class="stat-value">{f"{channel_stats['2024'].get('min_value'):,.0f}" if channel_stats['2024'].get('min_value') is not None else 'N/A'}</div>
                                <div class="stat-period">{channel_stats['2024'].get('min_period', 'N/A')}</div>
                            </div>
                        </div>

                        <h3>Previsões 2025</h3>
                        <div class="stats">
                            <div class="stat-card">
                                <h4>Total Previsto (Média dos Modelos)</h4>
                                <div class="stat-value">{f"{pred_2025.get('total_predicted'):,.0f}" if pred_2025.get('total_predicted') is not None else 'N/A'}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Máximo Previsto</h4>
                                <div class="stat-value">{f"{pred_2025.get('max_value'):,.0f}" if pred_2025.get('max_value') is not None else 'N/A'}</div>
                                <div class="stat-period">{pred_2025.get('max_period', 'N/A')}</div>
                                <div class="stat-model">Modelo: {pred_2025.get('max_model', 'N/A')}</div>
                            </div>
                            <div class="stat-card">
                                <h4>Mínimo Previsto</h4>
                                <div class="stat-value">{f"{pred_2025.get('min_value'):,.0f}" if pred_2025.get('min_value') is not None else 'N/A'}</div>
                                <div class="stat-period">{pred_2025.get('min_period', 'N/A')}</div>
                                <div class="stat-model">Modelo: {pred_2025.get('min_model', 'N/A')}</div>
                            </div>
                        </div>
                    </div>
                    <div class="plot-container">
                        <h3>Histórico e Previsões Completas</h3>
                        <iframe src="plots/plot_{channel}.html"></iframe>
                    </div>
                </div>
                """

        template += f"""
                <div class="footer">
                    <p>Relatório gerado automaticamente | Frequência: {frequency} | Data: {current_time}</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Salvar arquivo HTML
        output_path = "output/report.html"
        print(f"Salvando relatório em {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(template)

        print(f"Relatório HTML gerado com sucesso em {output_path}")
        return output_path

    except Exception as e:
        print(f"Erro ao gerar relatório HTML: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
