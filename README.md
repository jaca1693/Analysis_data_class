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
| Colunas 'flow\_meter' | [240152.20061594024] | [0.0] |
| Colunas 'press' | [181474.47539419503] | [0.0] |

## 5. Avaliação Comparativa de Classificadores (Seção 3)

A tabela abaixo resume o desempenho dos modelos no conjunto de teste, evidenciando o desempenho superior do DTC.

| Modelo de Classificação | Acurácia no Teste | Desvio Residual (Log Loss) | Matriz de Confusão (Classe Minoritária) |
| :--- | :--- | :--- | :--- |
| **DTC Otimizado** | **$0.9732$** | N/A | $211$ Falsos Positivos / $358$ Verdadeiros Positivos |
| **MLPClassifier** | $0.9477$ | N/A | $441$ Falsos Positivos / $94$ Verdadeiros Positivos |
| **GaussianNB** | $0.8462$ | N/A | $4504$ Falsos Positivos / $385$ Verdadeiros Positivos |
| **CNN-LSTM (PyTorch)** | N/A (Treinamento Parcial) | Redução de Loss: $0.6693$ $\rightarrow$ $0.6361$ | N/A |

### Destaque: Desempenho do Árvore de Decisão Otimizada

A poda do DTC foi confirmada como eficaz pelo $\text{GridSearchCV}$, resultando em um **melhor $\text{CV Score}$ de $0.9724$** e um modelo final com **$286$ nós folha**.

**Matriz de Confusão Final (DTC Otimizado):**

| Truth | 0.0 | 1.0 |
| :--- | :--- | :--- |
| **Predicted 0.0** | $39564$ | $915$ |
| **Predicted 1.0** | $211$ | $358$ |

### Destaque: Performance do Naive Bayes

O modelo Naive Bayes resultou no maior número de pontos mal classificados ($6,467$ de $42,048$) e uma baixa Acurácia, o que reforça a conclusão sobre a inadequação do modelo Gaussiano para a distribuição real dos dados.

## 6. Conclusões Finais

O **Árvore de Decisão Otimizada** oferece a solução mais confiável e interpretable para a deteção de fugas neste sistema. A sua alta precisão e baixo número de Falsos Positivos ($211$) o tornam ideal para sistemas em que alarmes falsos devem ser minimizados.

**Direções Futuras de Pesquisa:**
1.  Concentrar esforços na otimização e treinamento estendido do modelo CNN-LSTM, explorando técnicas de regularização e *learning rate scheduling*.
2.  Implementar estratégias de **rebalanceamento de classes** (e.g., *sampling* ou ajuste de pesos na função de perda) para aumentar a sensibilidade do modelo na deteção da classe minoritária (Recall).
