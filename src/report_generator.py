import os
from datetime import datetime
import pandas as pd
from models.evaluation import (
    evaluate_models,
    create_evaluation_summary,
    find_best_models,
    create_best_models_summary
)

def generate_html_report(df_notifications, prediction_models, future_predictions, evaluations):
    """
    Gera um relatório HTML completo com todas as análises e resultados.
    """
    # Criar diretório de relatórios se não existir
    os.makedirs("output/reports", exist_ok=True)

    # Nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"output/reports/relatorio_analise_{timestamp}.html"

    # Gerar conteúdo HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análise de Notificações</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 95%;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
                margin-bottom: 15px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .metric-card {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-family: monospace;
            }}
            .tabs {{
                display: flex;
                margin-bottom: 20px;
                border-bottom: 2px solid #ddd;
            }}
            .tab {{
                padding: 10px 20px;
                cursor: pointer;
                background: #f8f9fa;
                border: none;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
                transition: background-color 0.3s;
            }}
            .tab:hover {{
                background: #e9ecef;
            }}
            .tab.active {{
                background: #007bff;
                color: white;
            }}
            .tab-content {{
                display: none;
                padding: 20px;
                background: white;
                border-radius: 0 0 5px 5px;
                height: calc(100vh - 200px);
                min-height: 800px;
            }}
            .tab-content.active {{
                display: block;
            }}
            iframe {{
                width: 100%;
                height: 100%;
                border: none;
                overflow: auto;
            }}
            .plot-container {{
                width: 100%;
                height: 100%;
                overflow: visible;
            }}
            .js-plotly-plot {{
                width: 100% !important;
            }}
            .main-svg {{
                width: 100% !important;
            }}
        </style>
        <script>
            function reloadIframe(containerId) {{
                const iframe = document.querySelector('#' + containerId + ' iframe');
                if (iframe) {{
                    const currentSrc = iframe.src;
                    iframe.src = 'about:blank';
                    setTimeout(() => {{
                        iframe.src = currentSrc;
                    }}, 100);
                }}
            }}

            function openTab(evt, tabName) {{
                // Ocultar todos os conteúdos das abas
                const tabcontent = document.getElementsByClassName("tab-content");
                for (let i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}

                // Remover a classe active de todas as abas
                const tablinks = document.getElementsByClassName("tab");
                for (let i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}

                // Mostrar o conteúdo da aba selecionada e marcar como ativa
                const selectedTab = document.getElementById(tabName);
                selectedTab.style.display = "block";
                evt.currentTarget.className += " active";

                // Recarregar o iframe da aba selecionada
                reloadIframe(tabName);
            }}

            function adjustIframeHeight(iframe) {{
                try {{
                    const height = iframe.contentWindow.document.documentElement.scrollHeight;
                    iframe.style.height = height + 'px';
                }} catch (e) {{
                    console.log('Erro ao ajustar altura do iframe:', e);
                }}
            }}

            window.onload = function() {{
                // Abrir a primeira aba por padrão
                document.getElementById("defaultOpen").click();

                // Configurar ajuste automático de altura para todos os iframes
                const iframes = document.getElementsByTagName('iframe');
                for (let iframe of iframes) {{
                    iframe.onload = function() {{
                        adjustIframeHeight(this);
                    }};
                }}
            }};
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Análise de Notificações</h1>

            <div class="tabs">
                <button class="tab active" id="defaultOpen" onclick="openTab(event, 'graficosCompletos')">Gráficos Completos</button>
                <button class="tab" onclick="openTab(event, 'graficos2024')">Gráficos 2024</button>
                <button class="tab" onclick="openTab(event, 'relatorio')">Relatório</button>
            </div>

            <div id="graficosCompletos" class="tab-content">
                <iframe src="../analise_completa.html" scrolling="yes"></iframe>
            </div>

            <div id="graficos2024" class="tab-content">
                <iframe src="../analise_2024.html" scrolling="yes"></iframe>
            </div>

            <div id="relatorio" class="tab-content">
                <div class="section">
                    <h2>Resumo dos Dados</h2>
                    <div class="metric-card">
                        <p><strong>Período Analisado:</strong> {df_notifications.index.min().strftime("%d/%m/%Y")} a {df_notifications.index.max().strftime("%d/%m/%Y")}</p>
                        <p><strong>Canais Analisados:</strong> {", ".join(df_notifications['channels'].unique())}</p>
                        <p><strong>Total de Registros:</strong> {len(df_notifications):,}</p>
                    </div>
                </div>

                <div class="section">
                    <h2>Avaliação dos Modelos</h2>
                    <pre>{create_evaluation_summary(evaluations)}</pre>
                </div>

                <div class="section">
                    <h2>Melhores Modelos</h2>
                    <pre>{create_best_models_summary(find_best_models(evaluations))}</pre>
                </div>

                <div class="section">
                    <h2>Conclusões e Recomendações</h2>
                    <div class="metric-card">
                        <h3>Principais Observações:</h3>
                        <ul>
                            <li>Os modelos estatísticos (ARIMA, SARIMA, ETS) são eficazes para padrões lineares e sazonalidade.</li>
                            <li>Os modelos de Machine Learning (Random Forest, XGBoost, LightGBM) capturam relações complexas nos dados.</li>
                            <li>Os modelos de Deep Learning (LSTM, N-BEATS) são adequados para padrões não-lineares e de longo prazo.</li>
                        </ul>

                        <h3>Recomendações:</h3>
                        <ul>
                            <li>Utilizar uma combinação de modelos para diferentes cenários e canais.</li>
                            <li>Monitorar e atualizar os modelos periodicamente.</li>
                            <li>Considerar fatores externos que podem influenciar as notificações.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Salvar relatório
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return report_file
