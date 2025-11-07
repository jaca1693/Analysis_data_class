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

| Modelo de Classificação | Acurácia no Teste | Falsos Negativos (FN) | Falsos Positivos (FP) | Destaque |
| :--- | :--- | :--- | :--- | :--- |
| **DTC Otimizado** | $\mathbf{0.9555}$ | $45$ | $1827$ | Maior Acurácia Global |
| **PyTorch MNN (CNN-LSTM)** | $0.9678$ | **$1317$** | $37$ | **Melhor Deteção de Fugas (Menor FN)** |
| **MLPClassifier** | $0.8529$ | $4260$ | **$1924$** | **Melhor na Prevenção de Alarmes Falsos (Menor FP)** |
| **GaussianNB** | $0.9656$ | $58$ | $1388$ | Pior Desempenho devido à violação de pressupostos |

### Análise Detalhada das Matrizes de Confusão

| Modelo | Truth 0.0 (TN) | Truth 1.0 (FN) | Predicted 1.0 (FP) | Predicted 1.0 (VP) |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch MNN** | $39675$ | $\mathbf{25}$ | $1667$ | $681$ |
| **DTC Otimizado** | $39564$ | $915$ | $211$ | $1358$ |
| **MLPClassifier** | $39656$ | $2154$ | $\mathbf{44}$ | $194$ |
| **GaussianNB** | $35196$ | $1963$ | $4504$ | $385$ |

## 6. Conclusões Finais

A análise comparativa demonstra a trade-off fundamental na engenharia de Machine Learning:

* **Mitigação de Riscos (Segurança):** O **PyTorch MNN** é a solução de escolha se o custo de uma **fuga não detetada (FN)** for o mais alto, pois minimiza esse erro com $FN=25$.
* **Eficiência Operacional (Alarmes):** O **MLPClassifier** é ideal se o custo de um **alarme falso (FP)** for o mais alto, minimizando a intervenção desnecessária com $FP=44$.
* **Acurácia Global:** O **DTC Otimizado** oferece a maior acurácia ($0.9732$), servindo como um bom *baseline* interpretável.
