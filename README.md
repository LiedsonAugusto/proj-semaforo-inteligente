# Semáforo Inteligente

O **Semáforo Inteligente** é um sistema baseado em rede neural que recebe como entrada a configuração de veículos parados em um cruzamento e retorna o tempo de verde ideal para otimizar o fluxo de tráfego.  

O objetivo do projeto é **reduzir o tempo médio de espera** e **melhorar a gestão do tráfego urbano** por meio de ciclos semafóricos dinâmicos, adaptados em tempo real às condições detectadas.

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

## Status do projeto
O simulador e a coleta de dados estão concluídos.  
O modelo apresenta acurácia de até **84%** considerando 3 segundos de tolerância de erro, mas ainda não foi integrado a um semáforo funcional no SUMO.
