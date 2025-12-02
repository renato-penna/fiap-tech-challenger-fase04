"""
Analisador de Cenas de Vídeo

Orquestra o pipeline completo de análise de vídeo:
1. Detecção de Cenas
2. Detecção de Faces e Emoções
3. Detecção de Atividades
4. Agregação de Resultados por Cena
"""

import cv2
import numpy as np
from typing import List, Dict, Any
from collections import Counter
import logging

from src.models.scene import Scene, SceneResult
from src.services.scene_detector import SceneDetector
from src.services.detectors.face_detector import FaceDetector
from src.services.detectors.emotion_analyzer import EmotionAnalyzer
from src.services.detectors.activity_detector import ActivityDetector

logger = logging.getLogger(__name__)

class VideoSceneAnalyzer:
    """
    Analisador principal que processa o vídeo cena por cena.
    """
    
    def __init__(
        self,
        video_path: str,
        scene_threshold: float = 0.5,
        sample_rate: int = 5  # Analisar a cada N frames para performance
    ):
        """
        Inicializa o analisador.
        
        Args:
            video_path: Caminho para o vídeo.
            scene_threshold: Limiar para detecção de cenas (0.0-1.0).
            sample_rate: Taxa de amostragem de frames (processar 1 a cada N frames).
        """
        self.video_path = video_path
        self.sample_rate = sample_rate
        
        # Inicializar serviços
        self.scene_detector = SceneDetector(threshold=scene_threshold)
        self.face_detector = FaceDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.activity_detector = ActivityDetector()
        
    def analyze(self) -> List[SceneResult]:
        """
        Executa a análise completa do vídeo.
        
        Returns:
            Lista de SceneResult com os dados agregados de cada cena.
        """
        # 1. Detectar Cenas
        print("Detectando cenas...")
        scenes = self.scene_detector.detect_scenes(self.video_path)
        print(f"Detectadas {len(scenes)} cenas.")
        
        results = []
        
        # Abrir vídeo para processamento
        cap = cv2.VideoCapture(self.video_path)
        
        # 2. Processar cada cena
        for scene in scenes:
            print(f"Processando Cena {scene.scene_id} (Frames {scene.start_frame}-{scene.end_frame})...")
            scene_result = self._process_scene(cap, scene)
            results.append(scene_result)
            
        cap.release()
        return results
        
    def _process_scene(self, cap: cv2.VideoCapture, scene: Scene) -> SceneResult:
        """
        Processa uma única cena.
        
        Args:
            cap: Objeto VideoCapture aberto.
            scene: Objeto Scene definindo o intervalo.
            
        Returns:
            SceneResult com dados agregados.
        """
        # Ir para o frame inicial da cena
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene.start_frame)
        
        emotions_counter = Counter()
        actions_counter = Counter()
        total_faces = 0
        unique_tracks = set()
        
        current_frame = scene.start_frame
        
        while current_frame <= scene.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Processar apenas frames de acordo com sample_rate
            if (current_frame - scene.start_frame) % self.sample_rate == 0:
                # 1. Detectar Atividades (com tracking)
                activities = self.activity_detector.detect(frame)
                for activity in activities:
                    if activity.activity != "unknown":
                        actions_counter[activity.activity] += 1
                    if activity.track_id is not None:
                        unique_tracks.add(activity.track_id)
                
                # 2. Detectar Faces e Emoções
                faces = self.face_detector.detect(frame)
                total_faces += len(faces)
                
                if faces:
                    emotions = self.emotion_analyzer.analyze(frame, faces)
                    for emotion in emotions:
                        if emotion.emotion != "neutral": # Opcional: ignorar neutro se quiser focar em emoções fortes
                            emotions_counter[emotion.emotion] += 1
                        else:
                            emotions_counter["neutral"] += 1
            
            current_frame += 1
            
        # Agregar resultados
        return SceneResult(
            scene=scene,
            faces_detected=total_faces,
            unique_faces=len(unique_tracks) if unique_tracks else max(1, total_faces // max(1, (scene.duration_frames // self.sample_rate))), # Estimativa simples se tracking falhar
            dominant_emotions=dict(emotions_counter.most_common(3)),
            dominant_actions=dict(actions_counter.most_common(3)),
            anomalies=self._detect_anomalies(actions_counter, emotions_counter)
        )
        
    def _detect_anomalies(self, actions: Counter, emotions: Counter) -> List[str]:
        """
        Detecta anomalias simples baseadas em regras.
        
        Args:
            actions: Contagem de ações.
            emotions: Contagem de emoções.
            
        Returns:
            Lista de strings descrevendo anomalias.
        """
        anomalies = []
        
        # Exemplo de regra: Medo ou Raiva detectados
        if emotions.get("fear", 0) > 5:
            anomalies.append("Detecção de MEDO significativa")
        if emotions.get("angry", 0) > 5:
            anomalies.append("Detecção de RAIVA significativa")
            
        # Exemplo de regra: Ação incomum (ex: nenhuma ação clara por muito tempo)
        # Isso é apenas um placeholder, regras reais dependeriam do contexto
        
        return anomalies
