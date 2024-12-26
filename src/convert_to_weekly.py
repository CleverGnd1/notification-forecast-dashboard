import pandas as pd
import numpy as np
import os

def convert_to_weekly(input_file, output_file):
    """
    Converte dados mensais em semanais.
    """
    print(f"Lendo arquivo de entrada: {input_file}")
    df = pd.read_csv(input_file)

    # Converter coluna de data
    df['month'] = pd.to_datetime(df['month'])

    # Criar um DataFrame com todas as semanas no período
    start_date = df['month'].min()
    end_date = df['month'].max()

    print(f"Período: {start_date} até {end_date}")

    # Criar dados semanais
    weekly_data = []

    for _, row in df.iterrows():
        # Obter o mês atual e o próximo mês
        current_month = row['month']
        next_month = current_month + pd.DateOffset(months=1)

        # Criar semanas para o mês atual
        weeks = pd.date_range(start=current_month, end=next_month, freq='W')

        # Distribuir o valor mensal pelas semanas
        weekly_count = int(row['notification_count'] / len(weeks))

        # Adicionar uma entrada para cada semana
        for week_start in weeks:
            weekly_data.append({
                'week': week_start.strftime('%Y-%m-%d %H:%M:%S%z'),
                'channels': row['channels'],
                'notification_count': weekly_count
            })
            print(f"Adicionando registro: {week_start.strftime('%Y-%m-%d')} - {row['channels']} - {weekly_count}")

    # Criar DataFrame semanal
    weekly_df = pd.DataFrame(weekly_data)

    print(f"\nSalvando arquivo de saída: {output_file}")
    weekly_df.to_csv(output_file, index=False)
    print("Conversão concluída com sucesso!")
    print(f"Total de registros: {len(weekly_df)}")

    # Mostrar primeiros registros
    print("\nPrimeiros registros:")
    print(weekly_df.head())

if __name__ == "__main__":
    # Definir caminhos dos arquivos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    input_file = os.path.join(data_dir, 'notifications_data.csv')
    output_file = os.path.join(data_dir, 'notifications_data_week.csv')

    # Converter dados
    convert_to_weekly(input_file, output_file)
