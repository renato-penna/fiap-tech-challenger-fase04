from dataclasses import dataclass


@dataclass
class ActivityDetection:
    """
    Represents an activity detected in the video based on body pose.

    This class stores information about activities or actions identified
    through pose estimation analysis (body keypoint detection).
    YOLOv11 with pose estimation can detect multiple people and their
    activities even with partial occlusion or complex poses.

    Activity detection is performed by analyzing the relative position
    of body keypoints (shoulders, elbows, wrists, knees, etc.) to
    infer actions such as "hands up", "sitting", "standing", etc.

    Attributes:
        activity (str): Name of the detected activity. Common examples:
            - "hands_up"
            - "sitting"
            - "standing"
            - "walking"
            - "raising_hand"
            - "pointing"
        confidence (float): Confidence level of activity detection,
            ranging from 0.0 (no confidence) to 1.0 (full confidence).
            Indicates how certain the AI is about the identified activity.

    Example:
        >>> activity = ActivityDetection(activity="hands_up", confidence=0.92)
        >>> conf = activity.confidence * 100
        >>> print(f"Activity: {activity.activity} ({conf}% confidence)")

    Note:
        Unlike MediaPipe which has limitations with multiple people and
        occlusion, YOLOv11 offers better accuracy and support for more
        complex scenarios with multiple simultaneous people.
    """
    activity: str
    confidence: float
