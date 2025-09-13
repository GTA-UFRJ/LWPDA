'''
- Author: Hugo Antunes (antunes@gta.ufrj.br)

- From: Grupo de Teleinformática e Automação (UFRJ/COPPE) - Rio de Janeiro - Brazil

- Professors:
  Rodrigo de Souza Couto (rodrigo@gta.ufrj.br)
  Luís Henrique Maciel Kosmalski Costa (luish@gta.ufrj.br)
  Pedro Henrique Cruz Caminha (cruz@gta.ufrj.br)
 
- Description: This code is the class used for trying to decrease a potencially delay
when using Object Detection on real-time applications. The main idea is in the README file on GitHub.
'''

import time
import cv2 as cv
from ultralytics import YOLO
import numpy as np

class LWPDA:
    # ALTERADO: O construtor volta a ser mais simples
    def __init__(self, model_path: str, threshold: int = 0, verbose: bool = True, show: bool = True):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        # Armazena o threshold (0-1) diretamente
        self.threshold_percent = threshold/100
        self.verbose = verbose
        self.show_video = show
        print(f'LWPDA inicializado com o modelo "{model_path}" e threshold de {self.threshold_percent}%.')

    # ALTERADO: Usando a sua função de comparação original
    def is_similar(self, actual_frame, previous_frame, threshold) -> bool:
        """
        Compare two images (A and B) using RGB values.
        Threshold assists us to know when images is similar.
        """
        if previous_frame is None:
            return False
        
        x = abs((previous_frame - actual_frame))
        z = ((0 <= x) & (x <= 10)).sum()
        return (z >= threshold)
        
    def _process_video_generator(self, video_path: str):
        """Gerador que processa um vídeo frame a frame, aplicando a lógica LWPDA."""
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {video_path}")

        # ALTERADO: O limiar dinâmico é calculado aqui, como no seu código original
        video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_rgb_values = video_width * video_height * 3
        dynamic_threshold = total_rgb_values * self.threshold_percent
        
        if self.verbose:
            print(f"Limiar dinâmico de pixels similares calculado para {video_width}x{video_height}: {int(dynamic_threshold)}")

        previous_frame = None
        last_results = None
        
        while cap.isOpened():
            ret, actual_frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            
            # ALTERADO: A chamada para is_similar agora passa o limiar dinâmico
            if self.is_similar(actual_frame, previous_frame, dynamic_threshold):
                if self.verbose: print('Frame similar. Repetindo detecções.')
                if last_results:
                    annotated_frame = last_results[0].plot(img=actual_frame)
                else:
                    annotated_frame = actual_frame
                results_to_save = last_results
            else:
                if self.verbose: print('Processando frame com YOLO.')
                last_results = self.model(actual_frame, verbose=False)
                annotated_frame = last_results[0].plot()
                previous_frame = actual_frame
                results_to_save = last_results
            
            end_time = time.time()
            frame_time = end_time - start_time

            yield {
                "annotated_frame": annotated_frame,
                "results": results_to_save,
                "frame_time": frame_time
            }
        
        cap.release()
        cv.destroyAllWindows()

    def process_video(self, video_path: str) -> dict:
        """Processa um vídeo completo e retorna todas as detecções e tempos."""
        all_bounding_boxes = []
        all_masks = []
        all_frame_times = []
        
        video_start_time = time.time()

        for frame_data in self._process_video_generator(video_path):
            all_frame_times.append(frame_data['frame_time'])
            
            if frame_data['results']:
                result = frame_data['results'][0]
                classes = (result.boxes.cls.tolist(), result.boxes.conf.tolist())
                
                boxes = result.boxes.xyxy.tolist()
                all_bounding_boxes.append([classes, boxes])
                
                if result.masks:
                    masks = [m.tolist() for m in result.masks.xy]
                    all_masks.append([classes, masks])

            if self.show_video:
                cv.imshow(f"{self.model_path} - LWPDA Inference", frame_data['annotated_frame'])
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
        total_video_time = time.time() - video_start_time

        return {
            "bounding_boxes": all_bounding_boxes,
            "masks": all_masks,
            "frame_times": all_frame_times,
            "total_video_time": total_video_time
        }
