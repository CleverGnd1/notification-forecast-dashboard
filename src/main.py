import pandas as pd
import os
import tensorflow as tf
from models.predictions import (
    train_all_models,
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
    prediction_models = {}
    future_predictions = {}
    evaluations = {}

    for channel in channels:
        print(f"\nAnalisando canal: {channel}")
        try:
            # Filtrar dados do canal
            channel_data = df_notifications[df_notifications['channels'] == channel].copy()

            if len(channel_data) < 3:
                print(f"Dados insuficientes para treinar modelos no canal {channel}.")
                continue

            # Treinar todos os modelos disponíveis
            prediction_models[channel] = train_all_models(channel_data)

            # Gerar previsões
            last_date = channel_data.index.max()
            future_predictions[channel] = make_future_predictions(
                prediction_models[channel],
                channel,
                last_date,
                data=channel_data
            )

            # Avaliar modelos
            evaluations[channel] = evaluate_models(
                channel_data,
                prediction_models[channel],
                future_predictions[channel]
            )

        except Exception as e:
            print(f"Erro ao processar canal {channel}: {e}")
            continue

    # Criar e salvar visualizações
    try:
        os.makedirs("output", exist_ok=True)

        # Visualização completa
        combined_figure = create_combined_visualization(
            df_notifications,
            prediction_models,
            future_predictions
        )

        if combined_figure is not None:
            output_path = "output/analise_completa.html"
            combined_figure.write_html(output_path)
            print(f"\nVisualização completa salva em: {output_path}")

        # Visualização específica para 2024
        combined_figure_2024 = create_combined_visualization(
            df_notifications,
            prediction_models,
            future_predictions,
            year_filter=2024
        )

        if combined_figure_2024 is not None:
            output_path_2024 = "output/analise_2024.html"
            combined_figure_2024.write_html(output_path_2024)
            print(f"Visualização 2024 salva em: {output_path_2024}")

        # Gerar relatório completo com abas
        report_path = generate_html_report(
            df_notifications,
            prediction_models,
            future_predictions,
            evaluations
        )
        print(f"Relatório detalhado salvo em: {report_path}")

    except Exception as e:
        print(f"Erro ao criar visualizações: {e}")

    print("\nAnálise concluída!")

if __name__ == "__main__":
    main()
