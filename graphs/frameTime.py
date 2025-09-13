import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- 1. CONFIGURAÇÃO ---
# Altere esta linha para o caminho completo do seu diretório de experimentos.
# Baseado na sua imagem, o caminho seria algo como:
DIRETORIO_PRINCIPAL = Path('/home/hugo/allTests/segmentationNew/yolov8n-seg/')

# Nomes dos diretórios dos experimentos
NOMES_EXPERIMENTOS = [str(i) for i in range(11)] # Cria uma lista de '0' a '10'

# --- 2. COLETA DE DADOS ---
# Dicionário para guardar os dados: {'0': [t1, t2, ...], '1': [tA, tB, ...]}
dados_por_experimento = {}

print("Iniciando a coleta de dados...")

for exp_nome in NOMES_EXPERIMENTOS:
    # O enunciado diz "diretórioPrincipal/0/videos/txt". Isso pode ser interpretado de algumas formas.
    # O código abaixo assume que os arquivos .txt estão dentro da pasta 'videos'.
    # Ex: .../yolov8n-seg/0/videos/arquivo1.txt
    # Se seus arquivos estiverem em outra subpasta, ajuste a linha abaixo.
    # Ex: diretorio_txts = DIRETORIO_PRINCIPAL / exp_nome / 'videos' / 'txts'
    diretorio_txts = DIRETORIO_PRINCIPAL / exp_nome / 'frames'  # Ajuste conforme necessário
    
    if not diretorio_txts.is_dir():
        print(f"Aviso: Diretório não encontrado, pulando: {diretorio_txts}")
        continue

    tempos = []
    # Busca por todos os arquivos que terminam com .txt na pasta
    arquivos_txt = list(diretorio_txts.glob('*.txt'))
    
    if not arquivos_txt:
        print(f"Aviso: Nenhum arquivo .txt encontrado em {diretorio_txts}")
        continue

    for arquivo_path in arquivos_txt:
        try:
            with open(arquivo_path, 'r') as f:
                for linha in f:
                    try:
                        # Remove espaços em branco e converte para número
                        tempo = float(linha.strip())
                        tempos.append(tempo)
                    except ValueError:
                        # Ignora linhas que não são números
                        pass
        except Exception as e:
            print(f"Erro ao ler o arquivo {arquivo_path}: {e}")
    
    if tempos:
        dados_por_experimento[exp_nome] = tempos
        print(f"Experimento '{exp_nome}': {len(tempos)} medições de tempo coletadas.")

# --- 3. GERAÇÃO DOS GRÁFICOS (COM ESCALA AJUSTADA) ---
if not dados_por_experimento:
    print("\nNenhum dado foi coletado. Verifique o caminho em DIRETORIO_PRINCIPAL e a estrutura das pastas.")
else:
    labels = list(dados_por_experimento.keys())
    data = list(dados_por_experimento.values())

    # --- OPÇÃO 1: Box Plot com Escala Logarítmica (Recomendado) ---
    # Ideal para quando há grande variação nos dados.
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.boxplot(data, patch_artist=True)
    
    # A MUDANÇA PRINCIPAL ESTÁ AQUI:
    ax1.set_yscale('log')
    
    ax1.set_title('Comparação do Tempo de Processamento (Escala Logarítmica)', fontsize=16)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Tempo (segundos) - Escala Log', fontsize=12)
    ax1.set_xticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    ax1.yaxis.grid(True, which='both') # Adiciona grade para escala log
    
    plt.tight_layout() # Ajusta o layout para evitar cortes nos rótulos
    plt.savefig('/home/hugo/Desktop/LWPDA/graphs/output/frame/boxplot_tempos_escala_log.png')
    print("\nGráfico 'boxplot_tempos_escala_log.png' salvo com sucesso!")

    # --- OPÇÃO 2: Box Plot com Eixo Y Limitado ---
    # Útil para focar na maior parte dos dados e cortar os outliers visuais.
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.boxplot(data, patch_artist=True)

    # A MUDANÇA PRINCIPAL ESTÁ AQUI:
    # Vamos calcular um limite superior com base no 99º percentil
    # para ignorar apenas o 1% de dados mais extremos.
    try:
        todos_os_dados = np.concatenate([d for d in data if d]) # Junta todos os dados em uma lista
        limite_superior = np.percentile(todos_os_dados, 99)
        ax2.set_ylim(0, limite_superior * 1.05) # Damos uma margem de 5%
    except (ValueError, IndexError):
        print("Não foi possível calcular o limite do eixo Y. Pulando este ajuste.")

    ax2.set_title('Comparação do Tempo de Processamento (Eixo Y Limitado)', fontsize=16)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Tempo (segundos)', fontsize=12)
    ax2.set_xticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    ax2.yaxis.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/hugo/Desktop/LWPDA/graphs/output/frame/boxplot_tempos_eixo_limitado.png')
    print("Gráfico 'boxplot_tempos_eixo_limitado.png' salvo com sucesso!")
    
    # --- Bônus: Gráfico de Barras também com Escala Log ---
    medias = [np.mean(tempos) for tempos in data]
    desvios_padrao = [np.std(tempos) for tempos in data]
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.bar(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], medias, capsize=5, color='skyblue')
    # Apenas o erro positivo para funcionar bem com a escala log
    ax3.errorbar(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], medias, yerr=desvios_padrao, fmt='none', capsize=5, ecolor='gray')
    
    # Aplicando a escala log no gráfico de barras
    ax3.set_yscale('log')
    
    ax3.set_title('Média do Tempo de Processamento (Escala Logarítmica)', fontsize=16)
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Tempo Médio (segundos) - Escala Log', fontsize=12)
    ax3.yaxis.grid(True, which='both')

    plt.tight_layout()
    plt.savefig('/home/hugo/Desktop/LWPDA/graphs/output/frame/barras_tempos_media_escala_log.png')
    print("Gráfico 'barras_tempos_media_escala_log.png' salvo com sucesso!")
    labels = list(dados_por_experimento.keys())
    data = list(dados_por_experimento.values())

    # --- Gráfico 1: Box Plot (Recomendado para comparar distribuições) ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.boxplot(data, patch_artist=True)
    
    ax1.set_title('Comparação do Tempo de Processamento por Threshold', fontsize=16)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Tempo (segundos)', fontsize=12)
    ax1.set_xticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    ax1.yaxis.grid(True) # Adiciona uma grade para facilitar a leitura
    
    plt.savefig('/home/hugo/Desktop/LWPDA/graphs/output/frame/boxplot_tempos.png')
    print("\nGráfico 'boxplot_tempos.png' salvo com sucesso!")
    # plt.show() # Descomente para exibir o gráfico na tela

    # --- Gráfico 2: Gráfico de Barras com Média e Desvio Padrão ---
    medias = [np.mean(tempos) for tempos in data]
    desvios_padrao = [np.std(tempos) for tempos in data]
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.bar(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], medias, yerr=desvios_padrao, capsize=5, color='skyblue', ecolor='gray')

    ax2.set_title('Média do Tempo de Processamento por Threshold', fontsize=16)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Tempo Médio (segundos)', fontsize=12)
    ax2.yaxis.grid(True)

    plt.savefig('/home/hugo/Desktop/LWPDA/graphs/output/frame/barras_tempos_media.png')
    print("Gráfico 'barras_tempos_media.png' salvo com sucesso!")
    # plt.show() # Descomente para exibir o gráfico na tela