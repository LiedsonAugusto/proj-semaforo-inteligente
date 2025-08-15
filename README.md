# Semáforo Inteligente

O **Semáforo Inteligente** é um sistema baseado em rede neural que recebe como entrada a configuração de veículos parados em um cruzamento e retorna o tempo de verde ideal para otimizar o fluxo de tráfego.  

O objetivo do projeto é **reduzir o tempo médio de espera** e **melhorar a gestão do tráfego urbano** por meio de ciclos semafóricos dinâmicos, adaptados em tempo real às condições detectadas.

⚠️ **Requisito de hardware:** Para executar o treinamento do modelo de forma eficiente, é **necessário** possuir uma **placa de vídeo NVIDIA** com suporte a CUDA. O uso de GPU é altamente recomendado para evitar longos tempos de processamento.

## Funcionalidades
- Simulação de diferentes cenários de tráfego utilizando o **SUMO**.
- Geração e armazenamento de configurações de veículos e tempos de escoamento em banco de dados.
- Treinamento de rede neural com dados simulados para prever o tempo de verde ideal.
- Avaliação do modelo com diferentes níveis de tolerância de erro.

## Tecnologias utilizadas
- **Python**  
- **SUMO (Simulation of Urban MObility)**  
- **Rede Neural Artificial (PyTorch ou TensorFlow)**  
- **Banco de dados** para armazenamento das simulações  

## Estrutura geral
1. Simulador gera múltiplas configurações de tráfego.
2. Os dados são armazenados com tempos reais de escoamento.
3. A rede neural é treinada e avaliada.
4. O agente semafórico calcula o tempo de verde com base em novas configurações recebidas.

## Estrutura da pasta do modelo
Dentro da pasta `Sumo` (ou pasta equivalente onde o modelo está armazenado), temos os seguintes arquivos principais:

- **`DATASET_FINAL.csv`** → Arquivo bruto contendo os dados coletados para treinar o modelo. Inclui configurações de tráfego e tempos de escoamento.
- **`model_dataset.json`** → Versão dos dados em formato JSON, já convertidos para facilitar o carregamento pelo modelo, incluindo também informações de hiperparâmetros utilizados no treinamento.
- **`semaforo_inteligente.py`** → Implementação do modelo de rede neural responsável por receber a configuração de veículos e calcular o tempo de verde ideal.
- **`train_and_plot.py`** → Script responsável por carregar o modelo, executar o treinamento e gerar gráficos de desempenho.

## Status do projeto
O simulador e a coleta de dados estão concluídos.  
O modelo apresenta acurácia de até **84%** considerando 3 segundos de tolerância de erro, mas ainda não foi integrado a um semáforo funcional no SUMO.

## Link do vídeo:

https://drive.google.com/file/d/1t0JaDEbCoWOqjnIoU9KSosFGz5yBPHIo/view?usp=sharing
