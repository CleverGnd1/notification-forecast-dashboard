import os
import pandas as pd

def load_data(file_path=None, frequency='monthly'):
    """
    Carrega os dados do arquivo CSV.

    Args:
        file_path (str): Caminho do arquivo CSV (opcional)
        frequency (str): Frequência dos dados ('monthly' ou 'weekly')

    Returns:
        pd.DataFrame: DataFrame com os dados carregados
    """
    try:
        print("\nCarregando dados...")

        # Se o caminho não foi fornecido, usar o padrão
        if file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(current_dir), 'data')
            file_path = os.path.join(data_dir, 'notifications_data.csv' if frequency == 'monthly' else 'notifications_data_week.csv')

        print(f"Arquivo de dados: {file_path}")

        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            print(f"Erro: Arquivo não encontrado - {file_path}")
            return pd.DataFrame()

        # Carregar dados
        data = pd.read_csv(file_path)
        print(f"Dados carregados: {len(data)} registros")
        print(f"Colunas disponíveis: {', '.join(data.columns)}")

        # Validar colunas necessárias
        time_col = 'week' if frequency == 'weekly' else 'month'
        required_columns = [time_col, 'channels', 'notification_count']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Erro: Colunas ausentes no arquivo - {', '.join(missing_columns)}")
            return pd.DataFrame()

        # Renomear coluna de canal
        data = data.rename(columns={'channels': 'channel'})

        # Converter para datetime e definir como índice
        print("\nPreparando dados...")
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.set_index(time_col)

        # Definir frequência do índice
        if frequency == 'weekly':
            data.index = pd.DatetimeIndex(data.index).to_period('W')
        else:
            data.index = pd.DatetimeIndex(data.index).to_period('M')

        print(f"Período dos dados: {data.index[0]} até {data.index[-1]}")
        return data

    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
