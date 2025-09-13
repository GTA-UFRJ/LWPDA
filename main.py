# main.py

from pathlib import Path
import config  # <- Importa o novo arquivo de configurações

from LWPDA import LWPDA
from utils import (
    get_video_name, create_experiment_paths, write_to_json, write_times_to_txt
)
from evaluation import (
    build_results_dictionary, calculate_map_from_dict
)

def run_experiment_for_all_thresholds(
    model_path: str, 
    videos_dir: str, 
    results_dir: str, 
    threshold_levels: range,
    show_video: bool = True
):
    """
    Executa o experimento LWPDA para um modelo com todos os thresholds configurados.
    """
    model_name = Path(model_path).stem
    base_model_results_path = Path(results_dir) / model_name
    create_experiment_paths(results_dir, model_name)

    video_files = list(Path(videos_dir).glob('*.mp4'))
    
    for level in threshold_levels:
        threshold_percent = level * 10
        print(f"\n--- Processando para THRESHOLD = {threshold_percent}% (Nível {level}) ---\n")
        
        current_results_path = base_model_results_path / str(level)
        bb_path = current_results_path / 'bb'
        masks_path = current_results_path / 'masks'
        frames_path = current_results_path / 'frames'
        videos_path = current_results_path / 'videos'
        
        lwpda_processor = LWPDA(model_path, threshold=threshold_percent, verbose=False, show=show_video)
        
        video_times = []
        for video_path in video_files:
            video_name = get_video_name(str(video_path))
            print(f"Processando vídeo: {video_name}")
            
            results = lwpda_processor.process_video(str(video_path))
            
            write_to_json(results["bounding_boxes"], bb_path, video_name)
            if results["masks"]:
                write_to_json(results["masks"], masks_path, f"{video_name}_mask")
            
            write_times_to_txt(results["frame_times"], frames_path, video_name)
            video_times.append(results["total_video_time"])

        write_times_to_txt(video_times, videos_path, "video_processing_times")

def evaluate_map_for_experiment(
    results_dir: str, 
    model_name: str, 
    ground_truth_dir: str,
    threshold_levels: range,
    iou_threshold: float,
    is_segmentation: bool = False
):
    """
    Calcula o mAP para os resultados de um experimento com múltiplos thresholds.
    """
    print(f"\n--- Iniciando Avaliação de mAP para o modelo: {model_name} ---\n")
    base_model_results_path = Path(results_dir) / model_name

    for level in threshold_levels:
        print(f"Avaliando para Nível de Threshold: {level}")
        
        pred_dir = base_model_results_path / str(level)
        pred_data_path = pred_dir / 'masks' if is_segmentation else pred_dir / 'bb'

        try:
            results_dict = build_results_dictionary(
                gt_dir=ground_truth_dir,
                pred_dir=str(pred_data_path),
                iou_threshold=iou_threshold,
                is_segmentation=is_segmentation
            )
            
            mean_ap, ap_per_class = calculate_map_from_dict(results_dict)
            
            print(f"  - mAP: {mean_ap:.4f}")

        except Exception as e:
            print(f"  - Erro ao avaliar nível {level}: {e}")

if __name__ == "__main__":
    # --- Etapa 1: Executar o processamento dos vídeos ---
    # A função agora usa as variáveis importadas do arquivo config.py
    run_experiment_for_all_thresholds(
        model_path=str(config.MODEL_PATH),
        videos_dir=str(config.VIDEOS_DIRECTORY),
        results_dir=str(config.RESULTS_DIRECTORY),
        threshold_levels=config.THRESHOLD_LEVELS,
        show_video=config.SHOW_VIDEO_DURING_PROCESSING
    )
    
    # --- Etapa 2: Avaliar os resultados e calcular o mAP ---
    # O diretório de Ground Truth é construído dinamicamente a partir das configurações.
    # Ele assume que os resultados do nível 0 (YOLO puro) são a referência.
    model_stem = Path(config.MODEL_NAME).stem
    gt_base_path = config.RESULTS_DIRECTORY / model_stem / "10"
    
    ground_truth_dir = gt_base_path / "masks" if config.IS_SEGMENTATION_MODEL else gt_base_path / "bb"
    
    evaluate_map_for_experiment(
         results_dir=str(config.RESULTS_DIRECTORY),
         model_name=model_stem,
         ground_truth_dir=str(ground_truth_dir),
         threshold_levels=config.THRESHOLD_LEVELS,
         iou_threshold=config.IOU_THRESHOLD,
         is_segmentation=config.IS_SEGMENTATION_MODEL
    )
