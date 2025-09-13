import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- 1. CONFIGURAÇÃO ---
# !! IMPORTANTE !!
# Altere esta linha para o caminho completo do seu diretório de experimentos.
DIRETORIO_PRINCIPAL = Path('/home/hugo/allTests/segmentationNew/yolov8n-seg')

# Nomes dos diretórios dos experimentos
NOMES_EXPERIMENTOS = [str(i) for i in range(11)]

# --- 2. CARREGAMENTO DOS DADOS ---
dados_por_experimento = {}
print("Iniciando o carregamento dos dados de tempo...")

for exp_nome in NOMES_EXPERIMENTOS:
    diretorio_txts = DIRETORIO_PRINCIPAL / exp_nome / 'frames'
    
    if not diretorio_txts.is_dir():
        continue

    tempos = []
    arquivos_txt = list(diretorio_txts.glob('*.txt'))
    
    if not arquivos_txt:
        continue

    for arquivo_path in arquivos_txt:
        try:
            with open(arquivo_path, 'r') as f:
                for linha in f:
                    try:
                        tempo_em_segundos = float(linha.strip())
                        tempos.append(tempo_em_segundos * 1000) # Converte para ms
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Erro ao ler o arquivo {arquivo_path}: {e}")
    
    if tempos:
        dados_por_experimento[exp_nome] = np.array(tempos)
        print(f"Experimento '{exp_nome}': {len(tempos)} medições de tempo carregadas.")

# --- VERIFICAÇÃO ANTES DE PLOTAR ---
if not dados_por_experimento:
    print("\nERRO: Nenhum dado foi carregado. Verifique o caminho em DIRETORIO_PRINCIPAL.")
else:
    # --- GRÁFICO DE LINHAS ESTILIZADO com TRACEJADO ---
    print("\nGerando Gráfico de Linhas com tracejado para os pontos...")
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    experimentos_labels = sorted(dados_por_experimento.keys(), key=int)
    
    limiares_x = [int(label) * 10 for label in experimentos_labels]
    
    medianas = [np.median(dados_por_experimento[nome]) for nome in experimentos_labels]
    percentis_95 = [np.percentile(dados_por_experimento[nome], 95) for nome in experimentos_labels]

    # --- MUDANÇA AQUI: NOVO BLOCO PARA DESENHAR OS TRACEJADOS ---
    # Usamos um zorder baixo para que as linhas fiquem atrás dos pontos de dados.
    
    # Tracejados para os pontos da Mediana
    for x, y in zip(limiares_x, medianas):
        # Linha vertical do ponto até o eixo X
        ax2.vlines(x=x, ymin=0, ymax=y, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
        # Linha horizontal do ponto até o eixo Y
        ax2.hlines(y=y, xmin=0, xmax=x, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)

    # Tracejados para os pontos do Percentil 95
    for x, y in zip(limiares_x, percentis_95):
        # Linha vertical do ponto até o eixo X
        ax2.vlines(x=x, ymin=0, ymax=y, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
        # Linha horizontal do ponto até o eixo Y
        ax2.hlines(y=y, xmin=0, xmax=x, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
    
    # --- FIM DO NOVO BLOCO ---

    # Plotagem das linhas de dados principais (com zorder maior para ficarem na frente)
    ax2.plot(limiares_x, medianas, marker='o', linestyle='-', label='Mediana (P50)', markersize=8, zorder=2)
    ax2.plot(limiares_x, percentis_95, marker='D', linestyle='--', label='Percentil 95 (P95)', markersize=7, zorder=2)

    # Rótulos dos eixos
    ax2.set_xlabel('Limiar de similaridade (%)', fontsize=16)
    ax2.set_ylabel('Tempo de processamento (ms)', fontsize=16)

    # Legenda
    ax2.legend(loc='upper left', fontsize=12)

    # Eixo X com rótulo "YOLO"
    ticks_posicoes = range(0, 101, 10)
    ax2.set_xticks(ticks_posicoes)
    ticks_labels = [str(p) for p in ticks_posicoes]
    ticks_labels[-1] = 'YOLO'
    ax2.set_xticklabels(ticks_labels)

    # Limites e fontes
    ax2.set_xlim(left=-2, right=102)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig('grafico_linhas_com_tracejado.png', dpi=300)
    print("Gráfico 'grafico_linhas_com_tracejado.png' salvo com sucesso!")

    plt.show()