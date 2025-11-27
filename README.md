# Análise Comparativa de Classificadores para Deteção de Fugas em Sistemas de Tubulação

## 1. Resumo Executivo

Este projeto consiste na avaliação comparativa de diferentes técnicas de Machine Learning (ML) aplicadas à classificação de anomalias (fugas binárias, *burst* = 1) em dados multivariados. O principal objetivo é determinar o algoritmo mais adequado, robusto e interpretable para a implementação em sistemas de monitoramento de infraestrutura.

A avaliação rigorosa incluiu um modelo de **Árvore de Decisão Otimizada (DTC)**, **Naive Bayes (GaussianNB)**, **Perceptron Multicamadas (MLP)** e uma **Rede Neural Híbrida CNN-LSTM** customizada em PyTorch.

O **Árvore de Decisão Otimizada** demonstrou ser o classificador de melhor desempenho, atingindo uma **Acurácia de 0.9732** no conjunto de teste, ao mesmo tempo que mantém alta interpretabilidade das regras de classificação.

## 2. Estrutura do Repositório

| Arquivo/Pasta | Descrição |
| :--- | :--- |
| **CODE\_ANALYSIS.md** | **Documento Técnico Principal.** Contém o código-fonte integral de todas as seções do Google Colab, todas as saídas de console e tabelas detalhadas, resultados de testes estatísticos e visualizações. |
| `images/` | Repositório de arquivos gráficos (PNGs) referenciados no **CODE\_ANALYSIS.md** (Matriz de Correlação de Spearman, Diagramas de Árvore, Heatmaps). |

## 3. Metodologia de Análise

O processo analítico seguiu as fases:
1.  **Análise Estatística:** Avaliação de distribuição, desbalanceamento de classes e correlação (Spearman).
2.  **Modelagem e Treinamento:** Implementação de quatro classificadores (DTC, GaussianNB, MLP, CNN-LSTM) com treinamento em $70\%$ dos dados.
3.  **Otimização e Validação:** Ajuste de hiperparâmetros (Poda $\text{ccp}\_\alpha$ no DTC) e validação final da acurácia e da matriz de confusão no conjunto de teste.

## 4. Resultados Estatísticos Chave

### Matriz de Correlação de Spearman

A correlação não-paramétrica de Spearman foi utilizada para avaliar as relações monotônicas entre os pares de variáveis. O resultado confirma a forte correlação entre diversos sensores, o que é um fator de atenção para modelos lineares ou modelos que assumem independência, como o Naive Bayes.

(A matriz de correlação completa e seu *heatmap* estão disponíveis na Seção 2 do **CODE\_ANALYSIS.md**.)

### Teste ANOVA

O teste ANOVA para as variáveis de pressão e fluxo determinou a significância estatística das diferenças entre as médias das colunas, sendo um passo preliminar para a seleção de *features*.

| Variáveis | F-statistic | P-value |
| :--- | :--- | :--- |
| Colunas 'flow\_meter' | 58482.55070340394 | > 0.001 |
| Colunas 'press' | 47656.75028190731 | > 0.001 |

## 5. Avaliação Comparativa de Classificadores (Seção 3)

A tabela abaixo resume o desempenho dos modelos no conjunto de teste, evidenciando o desempenho superior do DTC.

| Modelo de Classificação | Acurácia no Teste | Falsos Negativos (FN) | Falsos Positivos (FP) | Destaque |
| :--- | :--- | :--- | :--- | :--- |
| **DTC Otimizado** | $\mathbf{0.9555}$ | $45$ | $1827$ | **Melhor Deteção de Fugas (Menor FN)** |
| **PyTorch MNN (CNN-LSTM)** | $0.9678$ | **$1317$** | $37$ | Maior Acurácia Global **Melhor na Prevenção de Alarmes Falsos (Menor FP)**  |
| **MLPClassifier** | $0.9656$ | $58$ | **$1388$** |  |
| **GaussianNB** | $0.8529$ | $4260$ | $1924$ | Pior Desempenho |

### Análise Detalhada das Matrizes de Confusão

| Modelo | Truth 0.0 (TN) | Truth 1.0 (FN) | Predicted 1.0 (FP) | Predicted 1.0 (VP) |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch MNN** | $39732$ | $1317$ | $37$ | $962$ |
| **DTC Otimizado** | $39724$ | $45$ | $1827$ | $452$ |
| **MLPClassifier** | $39711$ | $58$ | $1388$ | $891$ |
| **GaussianNB** | $35509$ | $4260$ | $1924$ | $355$ |

## 6. Conclusões Finais

A análise comparativa demonstra a trade-off fundamental na engenharia de Machine Learning:

* **Mitigação de Riscos (Segurança):** O **DTC Otimizado** é a solução de escolha se o custo de uma **fuga não detetada (FN)** for o mais alto, pois minimiza esse erro com $FN=45$.
* **Eficiência Operacional (Alarmes):** O **PyTorch** é ideal se o custo de um **alarme falso (FP)** for o mais alto, minimizando a intervenção desnecessária com $FP=37$.
* **Acurácia Global:** O **Pytorch** oferece a maior acurácia ($0.9732$), servindo como um bom *baseline* interpretável.
