# find_bad_json.py

import json
from pathlib import Path
import config  # Importa suas configurações para pegar os caminhos

def check_json_files(directory: Path):
    """
    Tenta ler todos os arquivos .json em um diretório para encontrar os corrompidos.
    """
    print(f"\n--- Verificando arquivos em: {directory} ---")
    found_corrupted = False
    
    if not directory.is_dir():
        print(f"ERRO: Diretório não encontrado: {directory}")
        return

    json_files = list(directory.glob('*.json'))
    
    if not json_files:
        print("Nenhum arquivo .json encontrado no diretório.")
        return

    for i, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            # Imprime o progresso para sabermos que está funcionando
            print(f"  ({i+1}/{len(json_files)}) OK: {file_path.name}", end='\r')
        except json.JSONDecodeError:
            print(f"\n\n!!! ARQUIVO CORROMPIDO ENCONTRADO: {file_path} !!!\n")
            found_corrupted = True
    
    if not found_corrupted:
        print(f"\nNenhum arquivo corrompido encontrado em {directory}.")


if __name__ == "__main__":
    # O script vai usar as configurações do seu config.py
    model_stem = Path(config.MODEL_NAME).stem
    base_results_path = config.RESULTS_DIRECTORY / model_stem
    
    # Verifique a pasta do Nível 4
    level_to_check = 4
    folder_to_check = base_results_path / str(level_to_check)
    
    if config.IS_SEGMENTATION_MODEL:
        check_json_files(folder_to_check / "masks")
    else:
        check_json_files(folder_to_check / "bb")