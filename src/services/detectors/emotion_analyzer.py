"""
Emotion Analyzer - Emotion Analysis

Analyzes emotions in detected faces using DeepFace.
"""

from typing import List
import numpy as np
from deepface import DeepFace

from src.models.face_detection import FaceDetection
from src.models.emotion_analysis import EmotionAnalysis


class EmotionAnalyzer:
    """
    Analyzes emotions in detected faces using DeepFace.

    This class uses the DeepFace library to analyze emotions in facial regions.
    It receives a frame and a list of detected faces, crops each facial region,
    and analyzes the emotion using pre-trained CNN models.
    """

    def __init__(
        self,
        enforce_detection: bool = False,
        detector_backend: str = 'opencv'
    ):
        """
        Initialize the emotion analyzer.

        Args:
            enforce_detection: If True, raises error if no face is detected.
                             If False, returns neutral emotion with low confidence.
            detector_backend: Face detection backend to use.
                             Options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
        """
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend

        print(
            f"EmotionAnalyzer initialized "
            f"(enforce_detection: {enforce_detection}, backend: {detector_backend})"
        )

    def analyze(
        self,
        frame: np.ndarray,
        faces: List[FaceDetection]
    ) -> List[EmotionAnalysis]:
        """
        Analyze emotions for each detected face in the frame.

        Process:
        1. For each detected face, crop the facial region
        2. Convert from BGR (OpenCV) to RGB (DeepFace)
        3. Call DeepFace.analyze() to get emotion scores
        4. Identify dominant emotion (highest score)
        5. Normalize confidence to 0-1
        6. Return list of EmotionAnalysis

        Args:
            frame: Video frame (numpy array, BGR format from OpenCV)
            faces: List of FaceDetection objects with detected faces

        Returns:
            List of EmotionAnalysis objects, one for each face
        """
        emotions = []

        for face in faces:
            # Step 1: Extract facial region from frame
            bbox = face.bounding_box

            # Ensure coordinates are within frame limits
            x = max(0, bbox.x)
            y = max(0, bbox.y)
            x_end = min(frame.shape[1], x + bbox.width)
            y_end = min(frame.shape[0], y + bbox.height)

            # Crop facial region
            face_region = frame[y:y_end, x:x_end]

            # Step 2: Validate region
            if face_region.size == 0:
                emotions.append(EmotionAnalysis(
                    emotion="neutral",
                    confidence=0.0
                ))
                continue

            try:
                # Step 3: Convert BGR to RGB (DeepFace expects RGB)
                face_rgb = np.array(face_region)
                # OpenCV uses BGR, DeepFace expects RGB
                if len(face_rgb.shape) == 3:
                    face_rgb = face_rgb[:, :, ::-1]  # BGR to RGB

                # Step 4: Analyze emotion with DeepFace
                result = DeepFace.analyze(
                    face_rgb,  # numpy array in RGB
                    actions=['emotion'],
                    enforce_detection=self.enforce_detection,
                    detector_backend=self.detector_backend,
                    silent=True  # Suppress DeepFace output
                )

                # Step 5: Extract emotion information
                # DeepFace returns a list if multiple faces, or dict if single face
                if isinstance(result, list):
                    result = result[0]

                # Get dominant emotion (highest score)
                dominant_emotion = result['dominant_emotion']

                # Get all emotion scores
                emotion_scores = result['emotion']

                # Step 6: Normalize confidence
                # Confidence is the score of dominant emotion (normalized to 0-1)
                # DeepFace returns scores as percentages (0-100), so divide by 100
                confidence = emotion_scores[dominant_emotion] / 100.0

                # Step 7: Create EmotionAnalysis object
                emotion_analysis = EmotionAnalysis(
                    emotion=dominant_emotion,
                    confidence=confidence
                )

                emotions.append(emotion_analysis)

            except Exception as e:
                # If analysis fails, return neutral emotion with low confidence
                emotions.append(EmotionAnalysis(
                    emotion="neutral",
                    confidence=0.0
                ))

        return emotions

