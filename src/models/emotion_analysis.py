from dataclasses import dataclass


@dataclass
class EmotionAnalysis:
    """
    Represents emotion analysis detected in a face.

    This class stores the result of emotion analysis performed by
    DeepFace or other emotion recognition models. Identifies the
    dominant emotion expressed by the face and the confidence level
    of this detection.

    Attributes:
        emotion (str): Name of the detected emotion. Possible values include:
            - "happy"
            - "sad"
            - "angry"
            - "surprise"
            - "neutral"
            - "fear"
            - "disgust"
        confidence (float): Confidence level of emotion detection, ranging
            from 0.0 (no confidence) to 1.0 (full confidence). Indicates how
            certain the AI is about the identified emotion.

    Example:
        >>> emotion = EmotionAnalysis(emotion="happy", confidence=0.87)
        >>> conf = emotion.confidence * 100
        >>> print(f"Emotion: {emotion.emotion} ({conf}% confidence)")
    """
    emotion: str
    confidence: float
