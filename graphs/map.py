import matplotlib.pyplot as plt
import numpy as np

# --- 1. Dados Extraídos ---
# Dados extraídos do seu texto.
experimentos = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
map_valores = [0.0719, 0.1039, 0.2090, 0.3330, 0.4644, 0.6024, 0.7450, 0.8561, 0.9298, 0.9747, 0.9999]

# --- 2. Criação do Gráfico ---
# Define o tamanho da figura para melhor visualização
fig, ax = plt.subplots(figsize=(15, 7))

# Plota a linha principal com marcadores nos pontos
ax.plot(experimentos, map_valores, marker='o', linestyle='-', color='b', label='mAP')

# --- 3. Melhorias no Gráfico ---
# Adiciona o valor de cada ponto diretamente no gráfico
for i, valor in enumerate(map_valores):
    ax.text(experimentos[i], valor + 0.02, f'{valor:.4f}', ha='center', fontsize=10)

# Adiciona Título e Rótulos aos Eixos
ax.set_title('Evolução do Mean Average Precision (mAP) por Limiar', fontsize=16)
ax.set_xlabel('Limiar de similaridade (%)', fontsize=15)
ax.set_ylabel('mAP Relativo', fontsize=15)

# Garante que todos os números de experimento apareçam no eixo X
ax.set_xticks(experimentos)

# Define os limites do eixo Y para dar um pouco de espaço
ax.set_ylim(0, 1.1)

# Adiciona uma grade para facilitar a leitura
ax.grid(True, linestyle='--', alpha=0.6)

# Ajusta o layout e salva o arquivo
plt.tight_layout()
plt.savefig('/home/hugo/Desktop/LWPDA/graphs/segmentation/grafico_map_evolucao.png')

print("Gráfico 'grafico_map_evolucao.png' salvo com sucesso!")