# utils.py

import os
import json
from pathlib import Path

def get_video_name(video_path: str) -> str:
    """Extrai o nome do arquivo de vídeo sem a extensão."""
    return Path(video_path).stem

def create_experiment_paths(base_result_path: str, model_name: str):
    """Cria a estrutura de diretórios para os resultados de um experimento."""
    model_path = Path(base_result_path) / model_name
    print(f"Criando diretórios de resultados em: {model_path}")
    for i in range(11):
        threshold_path = model_path / str(i)
        threshold_path.mkdir(parents=True, exist_ok=True)
        (threshold_path / 'bb').mkdir(exist_ok=True)
        (threshold_path / 'videos').mkdir(exist_ok=True)
        (threshold_path / 'frames').mkdir(exist_ok=True)
        (threshold_path / 'masks').mkdir(exist_ok=True)
    print("Estrutura de diretórios criada com sucesso.")

def write_to_json(data: list, output_path: str, file_name: str):
    """Salva uma lista de dados em um arquivo JSON."""
    file_path = Path(output_path) / f"{file_name}.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_from_json(file_path: str) -> list:
    """Carrega dados de um arquivo JSON."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_times_to_txt(times: list, output_path: str, file_name: str):
    """Salva uma lista de tempos (floats) em um arquivo .txt, um por linha."""
    file_path = Path(output_path) / f"{file_name}.txt"
    with open(file_path, 'w') as f:
        for t in times:
            f.write(f"{t}\n")