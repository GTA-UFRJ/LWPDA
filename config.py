# config.py

from pathlib import Path

# --- Configurações de Caminhos (Paths) ---
BASE_DIR = Path(__file__).resolve().parent

# 1. Defina o diretório dos seus vídeos
VIDEOS_DIRECTORY = "/home/guardai/antunes/IMAGENET/all_videos/"

# 2. Defina o diretório ONDE os resultados serão salvos.
#    O script criará a subpasta /yolov8n-seg/ e as pastas /0, /1, etc. dentro dela.
#    Se você quiser a pasta 'yolov8nseg1', pode renomear o modelo ou ajustar o main.py.
#    A forma mais simples é usar o caminho base aqui.
RESULTS_DIRECTORY = Path("/home/hugo/allTests/segmentationNew/")

# --- Configurações do Modelo ---
# 3. Defina o nome do modelo
MODEL_NAME = "yolov8n-seg.pt"
MODEL_PATH = BASE_DIR / MODEL_NAME # Deixe assim se o .pt estiver na mesma pasta

# --- Parâmetros do Experimento ---
# 4. show=False já está configurado aqui
SHOW_VIDEO_DURING_PROCESSING = False

# 5. O loop de 0 a 10 já está configurado aqui
THRESHOLD_LEVELS = range(11)

# --- Parâmetros de Avaliação (não serão usados nesta execução) ---
IOU_THRESHOLD = 0.5
IS_SEGMENTATION_MODEL = False
