# tsknn

[![PyPI version](https://badge.fury.io/py/tsknn.svg)](https://badge.fury.io/py/tsknn)
[![Build Status](https://github.com/ricardozago/tsknn/actions/workflows/python-app.yml/badge.svg)](https://github.com/ricardozago/tsknn/actions)
[![Coverage Status](https://coveralls.io/repos/github/ricardozago/tsknn/badge.svg?branch=main)](https://coveralls.io/github/ricardozago/tsknn?branch=main)

TSKNN (Time Series K-Nearest Neighbors) é uma biblioteca Python para previsão de séries temporais baseada em KNN, suportando estratégias multi-step, diferentes funções de combinação e transformações.

## Instalação

```bash
pip install tsknn
```

Ou clone o repositório e instale localmente:

```bash
git clone https://github.com/ricardozago/tsknn.git
cd tsknn
pip install .
```

## Exemplo de uso

```python
import numpy as np
from tsknn import tsknn

# Série temporal de exemplo
y = np.sin(np.linspace(0, 10, 100))

# Instancia e ajusta o modelo
model = tsknn(k=3, lags=5, h=10)
model.fit(y)

# Faz previsão
forecast = model.predict()
print(forecast)
```

## Testes

Para rodar os testes e verificar cobertura:

```bash
cd src
python -m pytest --cov=tsknn --cov-report term-missing
```

## Contribuindo

Contribuições são bem-vindas! Abra issues ou pull requests.

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/nome-feature`)
3. Commit suas mudanças (`git commit -am 'feat: nova feature'`)
4. Push para o branch (`git push origin feature/nome-feature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.