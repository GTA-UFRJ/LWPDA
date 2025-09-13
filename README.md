# LWPDA: Um Algoritmo Leve para Detecção de Diferença Perceptual em Vídeos

![GitHub language count](https://img.shields.io/github/languages/count/GTA-UFRJ/LWPDA)
![GitHub top language](https://img.shields.io/github/languages/top/GTA-UFRJ/LWPDA)
![GitHub repo size](https://img.shields.io/github/repo-size/GTA-UFRJ/LWPDA)
![GitHub last commit](https://img.shields.io/github/last-commit/GTA-UFRJ/LWPDA)

Este repositório apresenta o **LWPDA (Lightweight Perceptual Difference Algorithm)**, um novo algoritmo de baixo custo computacional para medir a similaridade visual entre quadros de vídeo. O projeto inclui um framework de testes completo para comparar a performance do LWPDA com outros métodos de referência, como SSIM, MSE, Image Hashing e Histogramas.

Desenvolvido pelo Grupo de Teleinformática e Automação (GTA) da UFRJ.

---

## Tabela de Conteúdos
- [O Problema](#o-problema)
- [A Solução: O Algoritmo LWPDA](#a-solução-o-algoritmo-lwpda)
- [Funcionalidades do Repositório](#funcionalidades-do-repositório)
- [Resultados Comparativos](#resultados-comparativos)
- [Começando](#começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Métodos de Comparação (Baselines)](#métodos-de-comparação-baselines)
- [Licença](#licença)
- [Agradecimentos](#agradecimentos)

---

## O Problema

Em sistemas de vigilância e análise de vídeo, o processamento contínuo de cada quadro com modelos de detecção de objetos (como o YOLO) é computacionalmente intensivo e, muitas vezes, redundante. Cenas estáticas ou com pouca movimentação geram uma grande quantidade de quadros visualmente similares, cujo processamento completo consome recursos desnecessariamente.

## A Solução: O Algoritmo LWPDA

Para resolver essa questão, propomos o **LWPDA**, um algoritmo projetado para ser uma primeira camada de filtragem rápida e eficiente. Ele compara quadros consecutivos e determina se a mudança visual entre eles é significativa o suficiente para justificar o acionamento de análises mais pesadas.

#### Intuição do LWPDA
[* **(Importante: Descreva aqui a ideia do seu algoritmo)**. Exemplo: O LWPDA opera no domínio da cor e da textura, extraindo um descritor compacto de cada quadro. A distância entre os descritores de quadros consecutivos indica o nível de similaridade perceptual entre eles, sendo robusto a pequenas variações de iluminação e ruído.*]

#### Vantagens
-   **Leve e Rápido**: Projetado para ter baixa latência e overhead computacional mínimo.
-   **Eficaz**: Capaz de discernir entre mudanças triviais e eventos relevantes no vídeo.
-   **Customizável**: Permite o ajuste de um limiar de similaridade para adaptar a sensibilidade do filtro.

---

## Funcionalidades do Repositório

-   **Implementação do LWPDA**: O código-fonte completo do algoritmo proposto.
-   **Framework de Comparação**: Um ambiente de testes robusto para executar e avaliar o LWPDA contra outros quatro algoritmos de referência.
-   **Análise de Performance**: Scripts para medir métricas essenciais, como tempo de processamento (latência) e distribuição (CDF, P95).
-   **Geração Automática de Gráficos**: Ferramentas para criar visualizações comparativas de performance, facilitando a análise dos resultados.

---

## Resultados Comparativos

Os gráficos gerados demonstram a eficiência do LWPDA em comparação com os métodos de referência e o custo de processar todos os frames com YOLO.

**1. CDF (Função de Distribuição Cumulativa) do Tempo de Processamento**
Este gráfico ilustra a velocidade de cada algoritmo. O LWPDA se destaca por [ *descreva o resultado, ex: processar uma alta proporção de quadros em tempo muito baixo* ].

*(Dica: Substitua esta imagem pela sua `grafico_cdf_estilo_final.png`)*
![CDF dos Algoritmos](caminho/para/sua/imagem_cdf.png)

**2. Performance vs. Limiar de Similaridade**
Aqui, comparamos o tempo de processamento em diferentes limiares. O gráfico mostra que o LWPDA [ *descreva o resultado, ex: mantém um desempenho estável e baixo em comparação com alternativas como o SSIM* ].

*(Dica: Substitua esta imagem pela sua `grafico_linhas_com_tracejado.png`)*
![Comparativo de Performance](caminho/para/sua/imagem_linhas.png)

---

## Começando

Siga estas instruções para configurar o ambiente e executar os testes.

### Pré-requisitos
-   Python 3.8+
-   Pip (Gerenciador de pacotes do Python)

### Instalação
1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/GTA-UFRJ/LWPDA.git](https://github.com/GTA-UFRJ/LWPDA.git)
    ```
2.  **Navegue até o diretório do projeto:**
    ```bash
    cd LWPDA
    ```
3.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
4.  **Instale as dependências a partir do `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```
    (Se o arquivo `requirements.txt` não existir, crie-o com `pip freeze > requirements.txt` ou instale as dependências manualmente).

---

## Como Usar

Para executar um teste de processamento em um vídeo, use o script principal. O foco é habilitar a comparação entre os métodos.

```bash
# Exemplo de execução com o algoritmo LWPDA
python seu_script_principal.py --video entrada.mp4 --method lwpda --threshold 0.85

# Exemplo de execução com um método de baseline para comparação
python seu_script_principal.py --video entrada.mp4 --method ssim --threshold 0.9