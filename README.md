# tsknn

`tsknn` é uma implementação em Python do algoritmo k-nearest neighbors (KNN) aplicada à predição de séries temporais. O pacote oferece funções para seleção de lags, otimização de hiperparâmetros e diferentes estratégias de previsão.

## Recursos principais

- Suporte a várias métricas de distância: `euclidean`, `manhattan`, `chebyshev` e `cosine`.
- Diferentes formas de agregação dos vizinhos (`cf`): média, mediana, ponderada ou trimmed.
- Modos de previsão MIMO, recursivo e direto (`msas`).
- Transformações aditivas ou multiplicativas para remoção de tendência.
- Função `optimize_params` para busca de melhores hiperparâmetros.

## Instalação

```bash
pip install tsknn
```

Para desenvolvimento local, clone o repositório e instale em modo editável:

```bash
pip install -e .
```

## Exemplo rápido

O exemplo abaixo utiliza os dados de passageiros aéreos disponíveis em `data/AirPassengers.csv` para prever os próximos 12 meses.

```python
import pandas as pd
from tsknn import tsknn

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]
X = df["passengers"].values

lags = 3
X_pred = X[-lags:]
model = tsknn(k=3, h=12, transform="multiplicative", lags=lags)
model.fit(X)
previsao = model.predict(X_pred)
print(previsao)
```

## Otimização de hiperparâmetros

Utilize `optimize_params` para encontrar a melhor combinação de parâmetros. Informe um `param_grid` com os valores a serem testados e uma série de validação.

```python
from tsknn import optimize_params

param_grid = {
    "k": [2, 3, 4],
    "lags": [3, 5],
    "h": [12]
}
melhores, score = optimize_params(X, param_grid)
print(melhores, score)
```

## Testes

Os testes unitários podem ser executados com `pytest` após instalar as dependências de desenvolvimento:

```bash
pip install -e .[dev]
pytest
```

## Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.
