import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- 1. CONFIGURAÇÃO ---
# !! IMPORTANTE !!
# Altere esta linha para o caminho completo do seu diretório de experimentos.
DIRETORIO_PRINCIPAL = Path('/home/hugo/allTests/segmentationNew/yolov8n-seg')

# Nomes dos diretórios dos experimentos que você quer plotar
NOMES_EXPERIMENTOS = [str(i) for i in range(11)]

# --- 2. CARREGAMENTO DOS DADOS ---
dados_por_experimento = {}
print("Iniciando o carregamento dos dados de tempo...")

for exp_nome in NOMES_EXPERIMENTOS:
    diretorio_txts = DIRETORIO_PRINCIPAL / exp_nome / 'frames'
    
    if not diretorio_txts.is_dir():
        continue # Pula silenciosamente se a pasta não existir

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
    print("\nERRO: Nenhum dado foi carregado. Verifique o caminho em DIRETORIO_PRINCIPAL e a estrutura das pastas.")
else:
    # --- 3. GERAÇÃO DO GRÁFICO DE CDF ESTILIZADO ---
    print("\nGerando Gráfico de CDF estilizado...")
    fig1, ax1 = plt.subplots(figsize=(10, 6)) # Ajusta o tamanho da figura

    # !! IMPORTANTE !!
    # Mapeie o nome do diretório do experimento para o rótulo que você quer na legenda.
    # Adapte este dicionário para a sua necessidade. Se um experimento não estiver aqui,
    # ele não aparecerá no gráfico.
    mapeamento_legendas = {
        '10': 'YOLO',
        '9': '90',
        '8': '80',
        '7': '70',
        '6': '60',
        '5': '50',
        # Adicione os experimentos de 6 a 10 aqui se quiser plotá-los
        # '6': 'Outro Rótulo', 
    }

    # Itera sobre o mapeamento para garantir a ordem e os rótulos corretos
    for exp_nome, legenda_label in mapeamento_legendas.items():
        if exp_nome in dados_por_experimento:
            tempos = dados_por_experimento[exp_nome]
            tempos_sorted = np.sort(tempos)
            probabilidade_cumulativa = np.arange(1, len(tempos_sorted) + 1) / len(tempos_sorted)
            ax1.plot(tempos_sorted, probabilidade_cumulativa, label=legenda_label, linewidth=2)

    # --- APLICAÇÃO DO ESTILO ---
    
    # 1. Rótulos dos eixos com fonte maior
    ax1.set_xlabel('Tempo de Processamento (ms)', fontsize=18)
    ax1.set_ylabel('Proporção de quadros', fontsize=18)

    # 2. Legenda com título e fonte ajustada
    ax1.legend(title='Limiar de similaridade (%)', fontsize=12, title_fontsize=13)

    # 3. Remoção da grade e do título principal (já removido)
    # ax1.grid(False) # O padrão já é sem grade

    # 4. Ajuste dos limites dos eixos e tamanho da fonte dos números (ticks)
    ax1.set_xlim(left=-2, right=140) # Um pouco de espaço à esquerda
    ax1.set_ylim(bottom=-0.02, top=1.05) # Um pouco de espaço em baixo/cima
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout() # Ajusta para que nada seja cortado
    plt.savefig('grafico_cdf_estilo_final.png', dpi=300) # Salva com alta resolução
    print("Gráfico 'grafico_cdf_estilo_final.png' salvo com sucesso!")

    plt.show()