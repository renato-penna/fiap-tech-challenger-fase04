from dataclasses import dataclass
from typing import Optional


@dataclass
class ActivityDetection:
    """
    Representa uma atividade detectada no vídeo com base na pose corporal.

    Esta classe armazena informações sobre atividades ou ações identificadas
    através da análise de estimativa de pose (detecção de pontos-chave do corpo).
    O YOLOv11 com estimativa de pose pode detectar várias pessoas e suas
    atividades mesmo com oclusão parcial ou poses complexas.

    A detecção de atividade é realizada analisando a posição relativa
    dos pontos-chave do corpo (ombros, cotovelos, pulsos, joelhos, etc.) para
    inferir ações como "mãos para cima", "sentado", "em pé", etc.

    Atributos:
        activity (str): Nome da atividade detectada. Exemplos comuns:
            - "talking" (falando)
            - "listening" (ouvindo)
            - "standing" (em pé)
            - "sitting" (sentado)
            - "waving" (acenando)
        confidence (float): Nível de confiança da detecção da atividade,
            variando de 0.0 (sem confiança) a 1.0 (confiança total).
            Indica quão certa a IA está sobre a atividade identificada.
        track_id (Optional[int]): ID de rastreamento da pessoa (para consistência temporal).
            Permite acompanhar a mesma pessoa ao longo dos frames.

    Exemplo:
        >>> activity = ActivityDetection(activity="waving", confidence=0.92, track_id=1)
        >>> conf = activity.confidence * 100
        >>> print(f"Atividade: {activity.activity} ({conf}% confiança)")
    """
    activity: str
    confidence: float
    track_id: Optional[int] = None
