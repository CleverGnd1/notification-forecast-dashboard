import os
from datetime import datetime
import pandas as pd
from models.evaluation import evaluate_models

def generate_html_report(data, predictions, evaluations):
    """
    Gera um relatório HTML com análises e previsões.
    """
    # Criar diretório de saída se não existir
    os.makedirs("output", exist_ok=True)
    output_path = "output/relatorio_analise.html"

    # Gerar gráficos
    from plot_generator import plot_predictions, plot_predictions_2024

    # Criar HTML com chaves escapadas no CSS
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Relatório de Análise de Notificações</title>
    <style>{{
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1500px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 8px 8px 0 0; }}
        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 16px; }}
        .tab button:hover {{ background-color: #ddd; }}
        .tab button.active {{ background-color: #3498db; color: white; }}
        .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; border-radius: 0 0 8px 8px; }}
        .plot-container {{ width: 100%; margin: 20px 0; text-align: center; }}
        .plot-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
    }}</style>
</head>
<body>
    <div class="container">
        <h1>Relatório de Análise de Notificações</h1>
        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'graficos')" id="defaultOpen">Gráficos Completos</button>
            <button class="tablinks" onclick="openTab(event, 'graficos2024')">Gráficos 2024</button>
            <button class="tablinks" onclick="openTab(event, 'metricas')">Métricas de Avaliação</button>
        </div>
        <div id="graficos" class="tabcontent">
            <h2>Gráficos de Previsão</h2>
            {graficos_content}
        </div>
        <div id="graficos2024" class="tabcontent">
            <h2>Gráficos de Previsão - 2024</h2>
            {graficos_2024_content}
        </div>
        <div id="metricas" class="tabcontent">
            <h2>Métricas de Avaliação dos Modelos</h2>
            {metricas_content}
        </div>
    </div>
    <script>
        function openTab(evt, tabName) {{{{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{{{
                tabcontent[i].style.display = "none";
            }}}}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{{{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}}}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}}}
        document.getElementById("defaultOpen").click();
    </script>
</body>
</html>"""

    # Gerar conteúdo dos gráficos
    graficos_content = ""
    graficos_2024_content = ""
    for channel in data['channels'].unique():
        if channel in predictions:
            # Filtrar dados do canal
            channel_data = data[data['channels'] == channel].copy()
            channel_data = channel_data.set_index('month')
            channel_data = channel_data.sort_index()

            # Gerar e salvar gráficos
            plot_path = f"output/plot_{channel}.png"
            plot_2024_path = f"output/plot_2024_{channel}.png"

            plot_predictions(channel_data, predictions[channel], channel, plot_path)
            plot_predictions_2024(channel_data, predictions[channel], channel, plot_2024_path)

            # Adicionar ao HTML
            graficos_content += f'<div class="plot-container"><h3>Canal: {channel}</h3><img src="{os.path.basename(plot_path)}" alt="Gráfico {channel}"></div>'
            graficos_2024_content += f'<div class="plot-container"><h3>Canal: {channel}</h3><img src="{os.path.basename(plot_2024_path)}" alt="Gráfico 2024 {channel}"></div>'

    # Gerar conteúdo das métricas
    metricas_content = ""
    for channel in evaluations:
        metricas_content += f'<h3>Canal: {channel}</h3><table><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R²</th></tr>'

        for model, metrics in evaluations[channel].items():
            if metrics:
                metricas_content += f'<tr><td>{model}</td><td>{metrics["MSE"]:.2f}</td><td>{metrics["RMSE"]:.2f}</td><td>{metrics["MAE"]:.2f}</td><td>{metrics["R2"]:.2f}</td></tr>'

        metricas_content += "</table>"

    # Substituir placeholders no template
    html_content = html_content.format(
        graficos_content=graficos_content,
        graficos_2024_content=graficos_2024_content,
        metricas_content=metricas_content
    )

    # Salvar arquivo HTML
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
