import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==============================================================================
# --- 1. CONFIGURAÇÃO PRINCIPAL ---
# ==============================================================================

DIRETORIO_PRINCIPAL = Path('/home/hugo/allTests/segmentationNew/yolov8n-seg/')
NOMES_EXPERIMENTOS = [str(i) for i in range(11)]
LABELS_BASE = [str(i * 10) for i in range(len(NOMES_EXPERIMENTOS) - 1)]
LABELS_COM_YOLO = LABELS_BASE + ['YOLO']

# ==============================================================================
# --- 2. COLETA E PROCESSAMENTO DOS DADOS ---
# ==============================================================================

dados_por_experimento = {}
print("Iniciando a coleta de dados...")
for exp_nome in NOMES_EXPERIMENTOS:
    diretorio_txts = DIRETORIO_PRINCIPAL / exp_nome / 'frames'
    if not diretorio_txts.is_dir():
        print(f"Aviso: Diretório não encontrado, pulando: {diretorio_txts}")
        continue
    tempos = []
    arquivos_txt = list(diretorio_txts.glob('*.txt'))
    if not arquivos_txt:
        print(f"Aviso: Nenhum arquivo .txt encontrado em {diretorio_txts}")
        continue
    for arquivo_path in arquivos_txt:
        try:
            with open(arquivo_path, 'r') as f:
                for linha in f:
                    try:
                        tempos.append(float(linha.strip()))
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Erro ao ler o arquivo {arquivo_path}: {e}")
    if tempos:
        dados_por_experimento[exp_nome] = tempos
        print(f"Experimento '{exp_nome}': {len(tempos)} medições de tempo coletadas.")

# ==============================================================================
# --- 3. GERAÇÃO DOS GRÁFICOS ---
# ==============================================================================

if not dados_por_experimento or len(dados_por_experimento) < len(NOMES_EXPERIMENTOS):
    print("\nDados insuficientes foram coletados.")
else:
    print("\nIniciando a geração dos gráficos...")
    sorted_keys = sorted(dados_por_experimento.keys(), key=int)
    data_sorted = [dados_por_experimento[key] for key in sorted_keys]
    medias = [np.mean(tempos) if tempos else 0 for tempos in data_sorted]
    desvios_padrao = [np.std(tempos) if tempos else 0 for tempos in data_sorted]
    
    # --- GRÁFICO 1: VERSÃO FINAL COM RÓTULOS DE DADOS ---
    
    medias_algoritmo = medias[:-1]
    media_yolo = medias[-1]
    
    fig_linha, ax_linha = plt.subplots(figsize=(10, 7))

    cor_lwpda = '#4800FF'
    cor_yolo = 'red'

    # 1. Plota a linha e os marcadores de círculo AZUIS de 0% a 90%.
    ax_linha.plot(LABELS_BASE, medias_algoritmo, color=cor_lwpda, linestyle='-', marker='o', label='LWPDA')

    # 2. Plota APENAS A LINHA tracejada VERMELHA de 90% a YOLO.
    ultimo_segmento_x = [len(LABELS_BASE) - 1, len(LABELS_BASE)]
    ultimo_segmento_y = [medias_algoritmo[-1], media_yolo]
    ax_linha.plot(ultimo_segmento_x, ultimo_segmento_y, color=cor_yolo, linestyle='--')
    
    # 3. Plota UM ÚNICO MARCADOR de diamante VERMELHO, apenas no ponto YOLO.
    x_yolo_pos = len(LABELS_BASE)
    y_yolo_pos = media_yolo
    ax_linha.plot(x_yolo_pos, y_yolo_pos, color=cor_yolo, marker='D', markersize=8, linestyle='none', label='YOLO')
    
    # 4. Adiciona as linhas de projeção para o ponto YOLO.
    ax_linha.plot([x_yolo_pos, x_yolo_pos], [0, y_yolo_pos], color='gray', linestyle='--')
    ax_linha.plot([0, x_yolo_pos], [y_yolo_pos, y_yolo_pos], color='gray', linestyle='--')
    
    # --- NOVO: 5. Adiciona os rótulos de texto com os valores em cada ponto ---
    # Itera sobre TODOS os pontos (0% a 90% e YOLO), usando a lista 'medias' completa.
    for i, media in enumerate(medias):
        # Formata o texto para 3 casas decimais (ex: 0.055)
        texto_label = f'{media:.3f}'
        
        # Adiciona o texto um pouco acima do ponto no gráfico
        ax_linha.text(
            x=i,                          # Posição X (o índice do ponto)
            y=media + 0.001,              # Posição Y (o valor + um pequeno deslocamento para cima)
            s=texto_label,                # O texto a ser exibido
            ha='center',                  # Alinhamento horizontal centralizado
            va='bottom',                  # Alinhamento vertical na base do texto
            fontsize=9,                   # Tamanho da fonte
            fontweight='bold',            # Deixa o texto em negrito para melhor leitura
            color='#363636'               # Cor cinza escuro
        )

    # Configuração final do gráfico
    ax_linha.set_title('Desempenho do Algoritmo em Segmentação', fontsize=16)
    ax_linha.set_xlabel('Limiar de Similaridade (%)', fontsize=12)
    ax_linha.set_ylabel('Tempo de Processamento médio de cada frame (segundos)', fontsize=12)
    ax_linha.set_xticks(range(len(LABELS_COM_YOLO)))
    ax_linha.set_xticklabels(LABELS_COM_YOLO)
    ax_linha.grid(True, linestyle='--', alpha=0.6)
    ax_linha.legend()
    ax_linha.set_ylim(bottom=0, top=max(medias) * 1.15) # Aumenta um pouco o limite superior para o texto caber
    plt.tight_layout()
    plt.savefig('/home/hugo/Desktop/LWPDA/graphs/segmentation/frame/linha_tempos_medios_com_yolo.png')
    print("Gráfico 1/6 'linha_tempos_medios_com_yolo.png' salvo com sucesso!")

    # --- DEMAIS GRÁFICOS (permanecem inalterados) ---
    # ... (código para os outros 5 gráficos) ...

    # ==============================================================================
    # --- 4. EXIBIÇÃO DOS VALORES MÉDIOS ---
    # ==============================================================================
    print("\n" + "="*50)
    print(" " * 10 + "Valores Médios de Processamento")
    print("="*50)
    for i, media in enumerate(medias):
        label = LABELS_COM_YOLO[i]
        print(f"  Limiar {label:>4}% : {media:.4f} segundos")
    print("="*50)