import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from datetime import datetime, timedelta
from .ml_models import train_ml_models

def get_forecast_steps(last_date_str, target_year, frequency='monthly'):
    """Helper function to calculate forecast steps."""
    try:
        # Convert string to datetime if needed
        if isinstance(last_date_str, str):
            last_date = pd.to_datetime(last_date_str)
        elif isinstance(last_date_str, pd.Period):
            last_date = last_date_str.to_timestamp()
        else:
            last_date = pd.to_datetime(str(last_date_str))

        if frequency == 'monthly':
            months_remaining = 12 - last_date.month if last_date.year == target_year else 12
            return (target_year - last_date.year) * 12 + months_remaining
        else:  # weekly
            target_end = pd.to_datetime(f"{target_year}-12-31")
            days_diff = (target_end - last_date).days
            return max(1, days_diff // 7)
    except Exception as e:
        print(f"Error in get_forecast_steps: {str(e)}")
        return 12  # Default to one year of monthly forecasts

def create_forecast_dates(last_date, steps, frequency='monthly'):
    """Helper function to create forecast dates."""
    try:
        # Convert to timestamp if needed
        if isinstance(last_date, pd.Period):
            last_date = last_date.to_timestamp()
        elif isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        else:
            last_date = pd.to_datetime(str(last_date))

        # Create date range
        freq = 'M' if frequency == 'monthly' else 'W'
        if frequency == 'monthly':
            next_date = last_date + pd.DateOffset(months=1)
        else:
            next_date = last_date + timedelta(days=7)

        dates = pd.date_range(start=next_date, periods=steps, freq=freq)
        return dates
    except Exception as e:
        print(f"Error in create_forecast_dates: {str(e)}")
        return pd.date_range(start=datetime.now(), periods=steps, freq='M')

def prepare_time_series(data, frequency='monthly'):
    """Helper function to prepare time series data."""
    try:
        # Convert index to datetime
        if isinstance(data.index[0], pd.Period):
            dates = data.index.to_timestamp()
        else:
            dates = pd.to_datetime(data.index)

        # Create proper time series
        freq = 'M' if frequency == 'monthly' else 'W'
        ts_index = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)

        # Reindex and forward fill any gaps
        ts_data = pd.Series(data.values, index=dates)
        ts_data = ts_data.reindex(ts_index, method='ffill')

        return ts_data
    except Exception as e:
        print(f"Error in prepare_time_series: {str(e)}")
        return pd.Series(data.values)

def prepare_data_for_prediction(data, frequency='monthly'):
    """
    Prepara dados para modelagem preditiva.
    """
    try:
        print("\nPreparando dados para previsão...")
        print(f"Dados recebidos: {len(data)} registros")
        print(f"Colunas disponíveis: {', '.join(data.columns)}")
        print(f"Índice: {data.index}")

        # Criar cópia para não modificar o original
        df = data.copy()

        # Remover valores nulos
        df = df.dropna()
        print(f"Registros após remover nulos: {len(df)}")

        # Garantir que temos as colunas corretas
        if 'channel' not in df.columns:
            print("Erro: Coluna 'channel' não encontrada")
            return pd.DataFrame()
        print(f"Registros após agrupamento: {len(df)}")

        print(f"Dados preparados com sucesso")
        print(f"Período: {df.index.min()} até {df.index.max()}")
        print(f"Canais únicos: {', '.join(df['channel'].unique())}")

        return df

    except Exception as e:
        print(f"Erro ao preparar dados: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def generate_predictions(data, channel, frequency='monthly', target_year=2025):
    """
    Gera previsões usando diferentes modelos.
    """
    try:
        print(f"\nGerando previsões para o canal {channel}...")
        print(f"Frequência: {frequency}")
        print(f"Ano alvo: {target_year}")

        if data is None or data.empty:
            print("Aviso: DataFrame está vazio")
            return {}

        # Verificar se há dados suficientes
        if len(data) < 2:
            print("Aviso: Não há dados suficientes para gerar previsões")
            return {}

        try:
            # Preparar dados do canal
            if isinstance(data, pd.DataFrame):
                if 'channel' in data.columns:
                    channel_data = data[data['channel'] == channel]['notification_count']
                else:
                    channel_data = data['notification_count']
            else:
                channel_data = data

            # Garantir que os dados estão ordenados pelo índice
            channel_data = pd.Series(channel_data).sort_index()

            if channel_data.empty:
                print(f"Aviso: Não há dados para o canal {channel}")
                return {}

            # Preparar série temporal
            ts_data = prepare_time_series(channel_data, frequency)
            ts_values = ts_data.values.astype(float)

            # Calcular número de períodos para previsão
            forecast_steps = get_forecast_steps(ts_data.index[-1], target_year, frequency)
            print(f"Períodos para previsão: {forecast_steps}")

            # Criar datas para previsões
            forecast_dates = create_forecast_dates(ts_data.index[-1], forecast_steps, frequency)

        except Exception as e:
            print(f"Erro ao preparar dados do canal: {str(e)}")
            return {}

        predictions = {}

        # ARIMA predictions
        print("\nTentando modelo ARIMA...")
        try:
            arima = ARIMA(ts_data, order=(1, 1, 1))
            arima_fit = arima.fit()
            forecast_values = arima_fit.forecast(steps=forecast_steps)
            predictions['Estatísticos - ARIMA'] = pd.Series(forecast_values, index=forecast_dates)
            print(f"Previsões ARIMA geradas: {len(predictions['Estatísticos - ARIMA'])} registros")
        except Exception as e:
            print(f"Erro ao gerar previsões ARIMA: {str(e)}")

        # SARIMA predictions
        print("\nTentando modelo SARIMA...")
        try:
            seasonal_order = (1, 1, 1, 12) if frequency == 'monthly' else (1, 1, 1, 52)
            sarima = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=seasonal_order)
            sarima_fit = sarima.fit(disp=False)
            forecast_values = sarima_fit.forecast(steps=forecast_steps)
            predictions['Estatísticos - SARIMA'] = pd.Series(forecast_values, index=forecast_dates)
            print(f"Previsões SARIMA geradas: {len(predictions['Estatísticos - SARIMA'])} registros")
        except Exception as e:
            print(f"Erro ao gerar previsões SARIMA: {str(e)}")

        # ETS predictions
        print("\nTentando modelo ETS...")
        try:
            # Criar DataFrame com índice temporal para o ETS
            ts_df = pd.DataFrame({'y': ts_data})
            ts_df.index = pd.date_range(start=ts_data.index[0], periods=len(ts_data), freq='M')

            model = ExponentialSmoothing(
                ts_df['y'],
                seasonal_periods=12 if frequency == 'monthly' else 52,
                trend='add',
                seasonal='add',
                initialization_method='estimated'
            )
            results = model.fit()
            forecast_values = results.forecast(steps=forecast_steps)
            predictions['Estatísticos - ETS'] = pd.Series(forecast_values, index=forecast_dates)
            print(f"Previsões ETS geradas: {len(predictions['Estatísticos - ETS'])} registros")
        except Exception as e:
            print(f"Erro ao gerar previsões ETS: {str(e)}")

        # Machine Learning predictions
        print("\nTentando modelos de Machine Learning...")
        try:
            # Preparar dados para ML
            df = pd.DataFrame({'notification_count': ts_data})
            ml_predictions = train_ml_models(df, forecast_steps=forecast_steps, frequency=frequency, target_year=target_year)
            predictions.update(ml_predictions)
            print(f"Previsões ML geradas: {len(ml_predictions)} modelos")
        except Exception as e:
            print(f"Erro ao gerar previsões ML: {str(e)}")

        # Convert predictions to PeriodIndex
        freq = 'M' if frequency == 'monthly' else 'W'
        for model_name in predictions:
            predictions[model_name].index = predictions[model_name].index.to_period(freq)

        # Verificar se alguma previsão foi gerada
        if not predictions:
            print("\nAviso: Nenhum modelo conseguiu gerar previsões")
            return {}

        print(f"\nPrevisões geradas com sucesso para {len(predictions)} modelos")
        for model_name, forecast in predictions.items():
            print(f"Modelo {model_name}: {len(forecast)} registros")
            if not forecast.empty:
                print(f"Período: {forecast.index[0]} até {forecast.index[-1]}")
                print(f"Valores: min={forecast.min():.2f}, max={forecast.max():.2f}")

        return predictions

    except Exception as e:
        print(f"Erro ao gerar previsões: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def generate_total_predictions(data, frequency='monthly', target_year=2025):
    """
    Gera previsões para o total de notificações de todos os canais.
    """
    print("\nGerando previsões para o total de notificações...")

    # Agrupar dados por período, somando todas as notificações
    total_data = data.groupby(data.index)['notification_count'].sum().to_frame()
    total_data['channel'] = 'total'  # Adicionar coluna de canal para manter compatibilidade

    # Gerar previsões usando a função existente
    predictions = generate_predictions(
        total_data,
        channel='total',
        frequency=frequency,
        target_year=target_year
    )

    return predictions
