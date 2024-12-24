import pandas as pd
import os
import tensorflow as tf
from models.predictions import (
    make_future_predictions,
    prepare_data_for_prediction
)
from visualization.plots import create_combined_visualization
from models.evaluation import evaluate_models
from report_generator import generate_html_report

# Configurar nível de log do TensorFlow e desabilitar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força uso apenas da CPU

def load_data(csv_path='data/notifications_data.csv'):
    """
    Carrega dados do arquivo CSV local.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    print(f"Carregando dados do arquivo CSV: {csv_path}")
    return pd.read_csv(csv_path)

def main():
    """
    Função principal que coordena o fluxo de análise.
    """
    print("Iniciando análise preditiva de notificações...")

    # Criar diretórios necessários
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Carregar dados
    try:
        df_notifications = load_data()
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # Preparar dados
    df_notifications = prepare_data_for_prediction(df_notifications)

    # Análise por canal
    channels = df_notifications['channels'].unique()
    future_predictions = {}
    evaluations = {}

    for channel in channels:
        print(f"\nAnalisando canal: {channel}")
        try:
            # Gerar previsões
            future_predictions[channel] = make_future_predictions(
                df_notifications,
                channel,
                months_ahead=12
            )

            if future_predictions[channel] is not None:
                # Avaliar modelos
                evaluations[channel] = evaluate_models(
                    df_notifications,
                    future_predictions[channel]
                )
            else:
                print(f"Não foi possível gerar previsões para o canal {channel}")

        except Exception as e:
            print(f"Erro ao processar canal {channel}: {e}")
            continue

    # Criar e salvar visualizações
    try:
        os.makedirs("output", exist_ok=True)

        # Visualização completa
        figures = create_combined_visualization(
            df_notifications,
            future_predictions
        )

        if figures:
            # Criar o HTML combinado com todos os gráficos
            with open("output/analise_completa.html", "w", encoding="utf-8") as f:
                f.write("""
                <html>
                <head>
                    <title>Análise de Notificações</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: #f5f5f5;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        }
                        .plot-container {
                            background-color: white;
                            margin: 20px auto;
                            padding: 30px;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            width: 95%;
                            max-width: 1500px;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        }
                        .plot-container > div {
                            width: 100% !important;
                            height: 100% !important;
                        }
                        .js-plotly-plot {
                            width: 100% !important;
                        }
                        .main-svg {
                            width: 100% !important;
                        }
                    </style>
                </head>
                <body>
                """)

                # Adicionar cada gráfico em um container
                for i, fig in enumerate(figures):
                    f.write(f'<div class="plot-container">{fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)}</div>')

                f.write("</body></html>")
            print(f"\nVisualização completa salva em: output/analise_completa.html")

        # Visualização específica para 2024
        figures_2024 = create_combined_visualization(
            df_notifications,
            future_predictions,
            year_filter=2024
        )

        if figures_2024:
            # Criar o HTML combinado com todos os gráficos de 2024
            with open("output/analise_2024.html", "w", encoding="utf-8") as f:
                f.write("""
                <html>
                <head>
                    <title>Análise de Notificações - 2024</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: #f5f5f5;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        }
                        .plot-container {
                            background-color: white;
                            margin: 20px auto;
                            padding: 30px;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            width: 95%;
                            max-width: 1500px;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        }
                        .plot-container > div {
                            width: 100% !important;
                            height: 100% !important;
                        }
                        .js-plotly-plot {
                            width: 100% !important;
                        }
                        .main-svg {
                            width: 100% !important;
                        }
                    </style>
                </head>
                <body>
                """)

                # Adicionar cada gráfico em um container
                for i, fig in enumerate(figures_2024):
                    f.write(f'<div class="plot-container">{fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)}</div>')

                f.write("</body></html>")
            print(f"Visualização 2024 salva em: output/analise_2024.html")

        # Gerar relatório completo com abas
        report_path = generate_html_report(
            df_notifications,
            future_predictions,
            evaluations
        )
        print(f"Relatório detalhado salvo em: {report_path}")

    except Exception as e:
        print(f"Erro ao criar visualizações: {e}")
        raise e  # Adicionar raise para ver o erro completo durante o desenvolvimento

    print("\nAnálise concluída!")

if __name__ == "__main__":
    main()
