from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bouding_box import BoundingBox


@dataclass
class FaceDetection:
    """
    Represents a face detection in a video frame.

    This class stores information about a detected face, including
    its location in the image (bounding box) and the confidence level
    of the detection. Used by YOLOv11 to store facial detection results.

    Attributes:
        bounding_box (BoundingBox): Bounding box that defines the position
            and size of the detected face in the image.
        confidence (float): Detection confidence level, ranging from 0.0
            (no confidence) to 1.0 (full confidence). Typical values above
            0.5 indicate reliable detections.

    Example:
        >>> from .bouding_box import BoundingBox
        >>> bbox = BoundingBox(x=100, y=50, width=200, height=300)
        >>> face = FaceDetection(bounding_box=bbox, confidence=0.95)
        >>> conf = face.confidence * 100
        >>> print(f"Face detected with {conf}% confidence")
    """
    bounding_box: "BoundingBox"
    confidence: float
