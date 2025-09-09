import cv2
import pandas as pd
from pathlib import Path
import os
import datetime

# --- CONFIGURAÇÃO ---
# O diretório que você forneceu.
# ATENÇÃO: Verifique se este caminho está correto no seu sistema.
VIDEOS_DIRECTORY = Path("/home/guardai/antunes/IMAGENET/all_videos/")

# Lista de extensões de vídeo a serem consideradas. Você pode adicionar outras se necessário.
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
# --------------------

def format_bytes(byte_size):
    """Converte um tamanho em bytes para um formato legível (KB, MB, GB)."""
    if byte_size is None:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while byte_size > power and n < len(power_labels) -1 :
        byte_size /= power
        n += 1
    return f"{byte_size:.2f} {power_labels[n]}"

def get_video_properties(video_path: Path):
    """
    Extrai as propriedades de um único arquivo de vídeo.
    Retorna um dicionário com as características ou None se ocorrer um erro.
    """
    try:
        # Pega o tamanho do arquivo em bytes
        file_size = video_path.stat().st_size

        # Abre o vídeo com o OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Aviso: Não foi possível abrir o vídeo: {video_path.name}")
            return None

        # Extrai as propriedades
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calcula a duração. Adiciona uma verificação para evitar divisão por zero.
        duration_seconds = 0
        if fps and fps > 0:
            duration_seconds = frame_count / fps
        
        # Libera o objeto de vídeo
        cap.release()

        return {
            "Arquivo": video_path.name,
            "Tamanho": file_size,
            "Resolucao": f"{width}x{height}",
            "Duracao_Seg": duration_seconds
        }

    except Exception as e:
        print(f"Erro ao processar o arquivo {video_path.name}: {e}")
        return None

def main():
    """
    Função principal que varre o diretório, analisa os vídeos e exibe os resultados.
    """
    print(f"🔎 Analisando vídeos no diretório: {VIDEOS_DIRECTORY}")

    if not VIDEOS_DIRECTORY.is_dir():
        print(f"❌ Erro: O diretório especificado não existe ou não é um diretório.")
        return

    all_videos_data = []
    
    # Percorre todos os arquivos no diretório de forma recursiva (inclui subpastas)
    # Use .glob('*') se quiser apenas os arquivos no diretório principal
    for file_path in VIDEOS_DIRECTORY.rglob('*'):
        # Verifica se é um arquivo e se a extensão é de vídeo
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            properties = get_video_properties(file_path)
            if properties:
                all_videos_data.append(properties)
    
    if not all_videos_data:
        print("✅ Nenhum arquivo de vídeo encontrado no diretório.")
        return

    # Converte a lista de dicionários em um DataFrame do Pandas
    df = pd.DataFrame(all_videos_data)
    
    # Formata as colunas para melhor visualização
    df['Tamanho'] = df['Tamanho'].apply(format_bytes)
    df['Duracao'] = df['Duracao_Seg'].apply(lambda s: str(datetime.timedelta(seconds=round(s))))
    
    # Reordena e remove a coluna de segundos original
    df = df[['Arquivo', 'Tamanho', 'Resolucao', 'Duracao']]
    
    print("\n--- 📊 Características dos Vídeos ---")
    print(df.to_string())
    print("------------------------------------")

    # Verifica se todas as resoluções são iguais
    unique_resolutions = df['Resolucao'].unique()
    
    print("\n--- 🎬 Análise da Resolução ---")
    if len(unique_resolutions) == 1:
        print(f"✅ Sim, todos os {len(df)} vídeos têm a mesma resolução: {unique_resolutions[0]}")
    else:
        print(f"⚠️ Não, os vídeos possuem resoluções diferentes.")
        print("Resoluções encontradas:")
        for res in unique_resolutions:
            count = df[df['Resolucao'] == res].shape[0]
            print(f"  - {res} ({count} vídeo(s))")
    print("------------------------------")


if __name__ == "__main__":
    main()
