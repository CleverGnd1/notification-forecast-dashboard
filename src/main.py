import os
import pandas as pd
from data_loader import load_data
from models import prepare_data_for_prediction, generate_predictions, generate_total_predictions
from report_generator import generate_html_report

def main(frequency='monthly', target_year=2025):
    """
    Função principal que executa a análise de dados e gera o relatório.

    Args:
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')
        target_year (int): Ano alvo para as previsões
    """
    try:
        print("\nIniciando análise de dados...")

        # Carregar dados
        print("\nCarregando dados...")
        data = load_data(frequency=frequency)
        if data is None or data.empty:
            print("Erro: Não foi possível carregar os dados")
            return
        print(f"Dados carregados com sucesso: {len(data)} registros")
        print(f"Colunas disponíveis: {', '.join(data.columns)}")
        print(f"Período: {data.index[0]} até {data.index[-1]}")

        # Preparar dados para previsão
        print("\nPreparando dados para previsão...")
        prepared_data = prepare_data_for_prediction(data, frequency=frequency)
        if prepared_data is None or prepared_data.empty:
            print("Erro: Não foi possível preparar os dados para previsão")
            return
        print(f"Dados preparados com sucesso: {len(prepared_data)} registros")
        print(f"Colunas disponíveis: {', '.join(prepared_data.columns)}")
        print(f"Período: {prepared_data.index[0]} até {prepared_data.index[-1]}")

        # Obter lista de canais únicos
        channels = prepared_data['channel'].unique()
        print(f"\nCanais encontrados: {', '.join(channels)}")

        # Dicionário para armazenar previsões
        all_predictions = {}
        all_data = {}

        # Gerar previsões para cada canal
        for channel in channels:
            print(f"\nProcessando canal: {channel}")

            # Filtrar dados do canal
            channel_data = prepared_data[prepared_data['channel'] == channel].copy()
            if channel_data is None or channel_data.empty:
                print(f"Aviso: Não há dados para o canal {channel}")
                continue

            print(f"Dados do canal: {len(channel_data)} registros")
            if len(channel_data.index) > 0:
                print(f"Período: {channel_data.index[0]} até {channel_data.index[-1]}")

            # Agrupar por período e preparar para previsão
            channel_data_grouped = channel_data.groupby(channel_data.index)['notification_count'].sum()
            print(f"Dados agrupados: {len(channel_data_grouped)} registros")

            # Criar DataFrame com os dados agrupados
            channel_df = pd.DataFrame({
                'notification_count': channel_data_grouped,
                'channel': channel
            })

            # Gerar previsões
            print(f"\nGerando previsões para o canal {channel}...")
            predictions = generate_predictions(channel_df, channel, frequency=frequency, target_year=target_year)

            if predictions is None or not predictions:
                print(f"Aviso: Não foi possível gerar previsões para o canal {channel}")
                continue

            print(f"Previsões geradas com sucesso para {len(predictions)} modelos")
            for model_name, forecast in predictions.items():
                if forecast is not None and not forecast.empty:
                    print(f"Modelo {model_name}: {len(forecast)} registros")
                    print(f"Período: {forecast.index[0]} até {forecast.index[-1]}")

            # Armazenar dados e previsões
            all_data[channel] = channel_df
            all_predictions[channel] = predictions

        if not all_predictions:
            print("\nErro: Nenhuma previsão foi gerada")
            return

        # Gerar previsões totais
        print("\nGerando previsões totais...")
        total_predictions = generate_total_predictions(prepared_data, frequency=frequency, target_year=target_year)
        if total_predictions:
            all_predictions['total'] = total_predictions

        print("\nGerando relatório HTML...")
        generate_html_report(all_data, all_predictions, frequency=frequency)
        print("Relatório HTML gerado com sucesso")

    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Gerar previsões de notificações')
    parser.add_argument('--frequency', choices=['monthly', 'weekly'], default='monthly',
                      help='Frequência dos dados (monthly ou weekly)')
    parser.add_argument('--target-year', type=int, default=2025,
                      help='Ano alvo para as previsões')

    args = parser.parse_args()
    main(frequency=args.frequency, target_year=args.target_year)
