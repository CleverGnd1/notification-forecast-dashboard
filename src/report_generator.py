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

        # Gerar gráficos para cada canal
        print("\nGerando gráficos...")
        for channel in data.keys():
            if channel in predictions and predictions[channel]:
                print(f"\nProcessando gráficos para o canal {channel}...")

                # Gerar gráfico completo
                plot_path = f"output/plots/plot_{channel}.png"
                print(f"Gerando gráfico completo em {plot_path}")
                plot_predictions(data[channel], predictions[channel], channel, plot_path, frequency)

                # Gerar gráfico focado em 2024
                plot_2024_path = f"output/plots/plot_{channel}_2024.png"
                print(f"Gerando gráfico 2024 em {plot_2024_path}")
                plot_predictions_2024(data[channel], predictions[channel], channel, plot_2024_path, frequency)
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
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-card h4 {{
                    color: #5f6368;
                    margin: 0 0 10px 0;
                }}
                .stat-value {{
                    font-size: 1.5em;
                    color: #1a73e8;
                    font-weight: bold;
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

        # Adicionar seções para cada canal
        for channel in data.keys():
            if channel in predictions and predictions[channel]:
                channel_data = data[channel]
                channel_stats = {
                    'total_records': len(channel_data),
                    'mean': channel_data['notification_count'].mean(),
                    'max': channel_data['notification_count'].max(),
                    'min': channel_data['notification_count'].min(),
                    'last_value': channel_data['notification_count'].iloc[-1]
                }

                template += f"""
                <div class="channel-section">
                    <h2>Canal: {channel}</h2>
                    <div class="stats">
                        <div class="stat-card">
                            <h4>Total de Registros</h4>
                            <div class="stat-value">{channel_stats['total_records']}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Média de Notificações</h4>
                            <div class="stat-value">{channel_stats['mean']:.1f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Máximo</h4>
                            <div class="stat-value">{channel_stats['max']:.0f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Mínimo</h4>
                            <div class="stat-value">{channel_stats['min']:.0f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>Último Valor</h4>
                            <div class="stat-value">{channel_stats['last_value']:.0f}</div>
                        </div>
                    </div>
                    <div class="plot-container">
                        <h3>Histórico e Previsões Completas</h3>
                        <iframe src="plots/plot_{channel}.html"></iframe>
                    </div>
                    <div class="plot-container">
                        <h3>Foco em 2024</h3>
                        <iframe src="plots/plot_{channel}_2024.html"></iframe>
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
