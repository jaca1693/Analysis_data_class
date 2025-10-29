# Projeto de Classificação: Avaliação Comparativa de Modelos para Deteção de Fugas

## Resumo Executivo

Este projeto é um estudo prático de Machine Learning focado na avaliação de diferentes algoritmos de classificação para a **deteção de fugas (classe binária 'burst')** em um conjunto de dados multivariado de sensores industriais. O objetivo principal foi comparar a performance de modelos paramétricos, não-paramétricos e de redes neurais.

O **Árvore de Decisão Otimizada (DTC Pruned)** destacou-se como o modelo mais eficaz, atingindo uma precisão de aproximadamente **$97.32\%$** no conjunto de teste, com um excelente balanço na identificação da classe minoritária.

## Estrutura do Repositório

| Arquivo/Pasta | Descrição |
| :--- | :--- |
| **CODE_ANALYSIS.md** | Documento técnico completo. Contém o código-fonte detalhado, todas as saídas de console, tabelas estatísticas e o registro de treinamento dos modelos. |
| `media/` | Armazena todas as visualizações (matrizes de correlação, árvores de decisão, heatmaps) incrustadas neste README. |

## Metodologia de Análise

O fluxo de trabalho seguiu as seguintes etapas:

1.  **Análise e Pré-processamento:** Limpeza, transformação e análise estatística (descritiva, testes paramétricos/não-paramétricos como T-test, ANOVA, Kruskal-Wallis).
2.  **Engenharia de Features:** Seleção das variáveis de entrada e *split* dos dados em conjuntos de treino e teste.
3.  **Modelagem Comparativa:** Treinamento e avaliação de quatro técnicas de classificação: Árvore de Decisão, Naive Bayes, MLP (Scikit-learn) e CNN-LSTM (PyTorch).

## Resultados Chave da Análise de Dados

### Estatísticas de Correlação

A Matriz de Correlação de Pearson foi crucial para identificar a redundância (multicolinearidade) entre as features, especialmente entre sensores do mesmo tipo (fluxo e pressão), o que impacta a interpretabilidade e a performance de alguns modelos.

![Pearson Correlation Matrix](media/correlation_matrix.png)

## Avaliação de Desempenho dos Modelos

### 1. Árvore de Decisão (DTC)

O DTC otimizado através de Poda por Complexidade de Custos ($\text{ccp}\_\alpha$) demonstrou ser o modelo de melhor desempenho e maior interpretabilidade.

| Métrica | Inicial ($\text{max\_depth}=3$) | Sem Poda (Geral) | Otimizado (Podado) |
| :--- | :--- | :--- | :--- |
| **Acurácia no Teste** | $0.9488$ | $0.9732$ | **$0.9732$** |
| **Melhor $\text{CV Score}$** | N/A | N/A | $0.9724$ |

**Matriz de Confusão do DTC Otimizado (Conjunto de Teste):**
A matriz confirma a robustez do modelo na identificação da classe minoritária (Fuga).

| Truth | 0.0 | 1.0 |
| :--- | :--- | :--- |
| **Predicted 0.0** | $39564$ | $915$ |
| **Predicted 1.0** | $211$ | $358$ |

![Pruned Decision Tree (Optimized via Cost-Complexity Pruning)](media/pruned_decision_tree.png)

### 2. Naive Bayes (GaussianNB)

O modelo Naive Bayes teve o pior desempenho, sugerindo que as distribuições dos dados violam as suposições de independência ou normalidade do modelo.

| Métrica | Resultado |
| :--- | :--- |
| **Acurácia no Teste** | $0.8462$ |
| **Pontos Mal Classificados** | $6,467$ de $42,048$ |

### 3. Perceptron Multicamadas (MLPClassifier)

O MLP da Scikit-learn alcançou uma acurácia competitiva, indicando que a capacidade de aprender relações não-lineares é benéfica para o problema.

| Métrica | Resultado |
| :--- | :--- |
| **Acurácia no Teste** | $0.9477$ |

### 4. CNN-LSTM (Modelo Customizado PyTorch)

A arquitetura CNN-LSTM foi implementada para explorar a natureza sequencial e as relações locais dos dados dos sensores.

**Arquitetura:**
```text
LeakDetectionModel(
  (cnn): Sequential(...)
  (lstm): LSTM(64, 128, num_layers=2, ...)
  (fc): Sequential(...)
)
