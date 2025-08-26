# lwpda.py

import time
import cv2 as cv
from ultralytics import YOLO

class LWPDA:
    def __init__(self, model_path: str, threshold: int = 0, verbose: bool = True, show: bool = True):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.threshold_percent = threshold
        self.verbose = verbose
        self.show_video = show
        print(f'LWPDA inicializado com o modelo "{model_path}" e threshold de {threshold}%.')

    @staticmethod
    def is_similar(frame_a, frame_b, similarity_threshold: float) -> bool:
        """Compara dois frames e retorna True se forem similares."""
        if frame_b is None:
            return False
        
        diff = cv.absdiff(frame_a, frame_b)
        non_similar_pixels = cv.countNonZero(cv.cvtColor(diff, cv.COLOR_BGR2GRAY))
        total_pixels = frame_a.shape[0] * frame_a.shape[1]
        
        # Esta é uma métrica de similaridade mais robusta que a original
        similarity = 1 - (non_similar_pixels / total_pixels)
        
        # A lógica original foi mantida como referência, mas a de cima é melhor
        # x = abs((frame_b - frame_a))
        # z = ((0 <= x) & (x <= 10)).sum()
        # return (z >= similarity_threshold)
        
        return similarity * 100 >= (100 - self.threshold_percent)


    def _process_video_generator(self, video_path: str):
        """
        Gerador que processa um vídeo frame a frame, aplicando a lógica LWPDA.
        Retorna (yields) informações de cada frame.
        """
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {video_path}")

        previous_frame = None
        last_results = None
        
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        dinamic_threshold = (width * height * 3) * self.threshold_percent / 100

        while cap.isOpened():
            ret, actual_frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            
            is_processed_by_yolo = False
            if self.is_similar(actual_frame, previous_frame, dinamic_threshold):
                if self.verbose: print('Frame similar. Repetindo detecções.')
                annotated_frame = last_results[0].plot(img=actual_frame)
                results_to_save = last_results
            else:
                if self.verbose: print('Processando frame com YOLO.')
                last_results = self.model(actual_frame, verbose=False)
                annotated_frame = last_results[0].plot()
                previous_frame = actual_frame
                results_to_save = last_results
                is_processed_by_yolo = True
            
            end_time = time.time()
            frame_time = end_time - start_time

            yield {
                "annotated_frame": annotated_frame,
                "results": results_to_save,
                "frame_time": frame_time,
                "is_processed_by_yolo": is_processed_by_yolo
            }
        
        cap.release()
        cv.destroyAllWindows()

    def process_video(self, video_path: str) -> dict:
        """
        Processa um vídeo completo e retorna todas as detecções e tempos.
        """
        all_bounding_boxes = []
        all_masks = []
        all_frame_times = []
        
        video_start_time = time.time()

        for frame_data in self._process_video_generator(video_path):
            all_frame_times.append(frame_data['frame_time'])
            
            if frame_data['results']:
                result = frame_data['results'][0]
                classes = (result.boxes.cls.tolist(), result.boxes.conf.tolist())
                
                # Salva Bounding Boxes
                boxes = result.boxes.xyxy.tolist()
                all_bounding_boxes.append([classes, boxes])
                
                # Salva Máscaras de Segmentação (se existirem)
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