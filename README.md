# LWPDA: Lightweight Perceptual Difference Algorithms

![GitHub language count](https://img.shields.io/github/languages/count/GTA-UFRJ/LWPDA)
![GitHub top language](https://img.shields.io/github/languages/top/GTA-UFRJ/LWPDA)
![GitHub repo size](https://img.shields.io/github/repo-size/GTA-UFRJ/LWPDA)
![GitHub last commit](https://img.shields.io/github/last-commit/GTA-UFRJ/LWPDA)


Um framework para análise e comparação de algoritmos leves de detecção de diferença perceptual em quadros de vídeo, visando a otimização de pipelines de processamento. Desenvolvido pelo Grupo de Teleinformática e Automação (GTA) da UFRJ.

---

## Tabela de Conteúdos
- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Resultados Visuais](#resultados-visuais)
- [Começando](#começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Licença](#licença)
- [Agradecimentos](#agradecimentos)

---

## Sobre o Projeto

Em muitos sistemas de análise de vídeo em tempo real, o processamento de cada quadro individualmente com modelos de deep learning (como o YOLO) pode ser computacionalmente caro e redundante, especialmente quando há pouca mudança entre quadros consecutivos.

O **LWPDA** surge como uma solução para este problema. Ele fornece um ambiente de teste para avaliar diferentes algoritmos de baixo custo computacional (ex: SSIM, Hashing, Histogramas) que podem rapidamente determinar se um quadro é suficientemente "novo" ou "diferente" do anterior para justificar o processamento por um modelo mais pesado. O objetivo é reduzir a carga de processamento sem perder informações visuais relevantes.

---

## Funcionalidades

-   **Implementação de Múltiplos Algoritmos**: Inclui implementações para SSIM, MSE, Image Hashing e Comparação de Histogramas.
-   **Framework de Análise de Performance**: Mede e compara o tempo de processamento (latência) de cada algoritmo.
-   **Análise de Precisão**: Avalia a eficácia de cada método com base em um limiar de similaridade.
-   **Geração de Gráficos Comparativos**: Cria visualizações detalhadas, como CDFs e gráficos de linha, para facilitar a análise dos resultados.

---

## Resultados Visuais

A performance dos algoritmos pode ser comparada através de gráficos gerados pelo framework.

**1. CDF (Função de Distribuição Cumulativa) do Tempo de Processamento**
Este gráfico mostra a proporção de quadros processados dentro de um determinado tempo. Curvas mais à esquerda e que sobem mais rápido representam algoritmos mais eficientes.

*(Dica: Substitua esta imagem pela sua `grafico_cdf_estilo_final.png`)*
![CDF dos Algoritmos](caminho/para/sua/imagem_cdf.png)

**2. Comparativo de Performance vs. Limiar**
Este gráfico compara métricas de tempo de processamento (como Mediana e P95) em diferentes limiares de similaridade, com o YOLO como baseline de custo.

*(Dica: Substitua esta imagem pela sua `grafico_linhas_com_tracejado.png`)*
![Comparativo de Performance](caminho/para/sua/imagem_linhas.png)

---

## Começando

Siga estas instruções para obter uma cópia local do projeto e executá-la.

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
4.  **Instale as dependências:**
    (É uma boa prática criar um arquivo `requirements.txt`)
    ```bash
    pip install -r requirements.txt
    ```
    Se o arquivo `requirements.txt` não existir, instale as bibliotecas principais manualmente:
    ```bash
    pip install opencv-python-headless matplotlib numpy scikit-image imagehash
    ```
---

## Como Usar

Para executar a análise principal, utilize o script principal a partir da linha de comando. Exemplo:

```bash
python seu_script_principal.py --video entrada.mp4 --output saida.mp4 --method ssim --threshold 0.9