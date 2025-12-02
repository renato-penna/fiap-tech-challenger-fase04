from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Scene:
    """
    Represents a detected scene in the video.
    
    Attributes:
        scene_id: Unique identifier for the scene
        start_frame: Starting frame number (inclusive)
        end_frame: Ending frame number (inclusive)
        start_time: Start time in seconds
        end_time: End time in seconds
        duration_frames: Duration in frames
        duration_seconds: Duration in seconds
    """
    scene_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    
    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1
        
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

@dataclass
class SceneResult:
    """
    Aggregated analysis results for a scene.
    
    Attributes:
        scene: The Scene object
        faces_detected: Total number of faces detected (sum of all frames)
        unique_faces: Estimated number of unique people
        dominant_emotions: Dictionary of emotion counts
        dominant_actions: Dictionary of action counts
        anomalies: List of detected anomalies
    """
    scene: Scene
    faces_detected: int = 0
    unique_faces: int = 0
    dominant_emotions: Dict[str, int] = field(default_factory=dict)
    dominant_actions: Dict[str, int] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
