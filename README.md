# Análise de Notificações

Este projeto realiza análise preditiva de notificações usando diferentes modelos de séries temporais (ARIMA, SARIMA, e ETS).

## Estrutura do Projeto

```
.
├── README.md
├── requirements.txt
└── src/
    ├── data/
    │   └── notifications_data.csv
    ├── models/
    │   ├── __init__.py
    │   ├── time_series.py
    │   └── predictions.py
    ├── visualization/
    │   ├── __init__.py
    │   └── plots.py
    └── main.py
```

## Requisitos

- Python 3.8 ou superior
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone [url-do-repositorio]
cd [nome-do-repositorio]
```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Certifique-se de que seus dados estão no arquivo `src/data/notifications_data.csv`

2. Execute o script principal:
```bash
python src/main.py
```

3. Os resultados serão salvos na pasta `output/` como arquivos HTML interativos.

## Funcionalidades

- Análise de séries temporais usando múltiplos modelos
- Previsões futuras para cada canal de notificação
- Visualizações interativas usando Plotly
- Análise total e por canal
- Exportação de gráficos em formato HTML interativo

## Estrutura de Dados

O arquivo CSV deve conter as seguintes colunas:
- `month`: Data no formato YYYY-MM-DD
- `channels`: Nome do canal de notificação
- `notification_count`: Número de notificações

## Saída

O script gera visualizações interativas em HTML para:
- Análise individual de cada canal
- Análise total combinando todos os canais

Os arquivos são salvos em:
- `output/{canal}_analysis.html` para cada canal
- `output/total_analysis.html` para a análise total
