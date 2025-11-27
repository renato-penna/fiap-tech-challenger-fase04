from ultralytics import YOLO
import numpy as np

from src.models.bouding_box import BoundingBox
from src.models.face_detection import FaceDetection


class FaceDetector:
    """
    Detects faces in video frames using YOLO Pose.

    Uses YOLO Pose to detect facial keypoints (nose, eyes, ears)
    and create precise bounding boxes based on these real points.

    Advantages over person detection + proportions:
    - Precision based on real facial keypoints
    - Works with different poses and angles
    - No need to estimate body proportions
    """

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        confidence_threshold: float = 0.5,
        keypoint_confidence: float = 0.5
    ):
        """
        Initializes the face detector using YOLO Pose.

        Args:
            model_path: Path to YOLO Pose model (default: yolo11n-pose.pt)
            confidence_threshold: Minimum confidence for person detections
            keypoint_confidence: Minimum confidence for facial keypoints

        Note:
            YOLO Pose detects body keypoints including facial points:
            - 0: nose
            - 1: left eye
            - 2: right eye
            - 3: left ear
            - 4: right ear
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.keypoint_confidence = keypoint_confidence
        self.person_class_id = 0

        print(
            f"✅ FaceDetector initialized (Pose model, "
            f"threshold: {confidence_threshold})"
        )

    def detect(self, frame):
        """
        Detects faces in a video frame using facial keypoints.

        Args:
            frame: Video frame (numpy array, BGR format from OpenCV)

        Returns:
            List of FaceDetection with bounding boxes and confidence
        """
        # Execute inference with pose estimation
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )

        faces = []
        frame_height, frame_width = frame.shape[:2]

        # Process each result
        for result in results:
            if result.keypoints is None:
                continue

            # Shape: (num_persons, num_keypoints, 3)
            keypoints = result.keypoints.data

            for person_keypoints in keypoints:
                # Facial keypoints in COCO pose:
                # 0: nose, 1: left eye, 2: right eye
                # 3: left ear, 4: right ear

                nose = person_keypoints[0]  # [x, y, confidence]
                left_eye = person_keypoints[1]
                right_eye = person_keypoints[2]
                left_ear = person_keypoints[3]
                right_ear = person_keypoints[4]

                # Filter keypoints with sufficient confidence
                face_keypoints = []
                if nose[2] > self.keypoint_confidence:
                    face_keypoints.append([nose[0].item(), nose[1].item()])
                if left_eye[2] > self.keypoint_confidence:
                    face_keypoints.append(
                        [left_eye[0].item(), left_eye[1].item()]
                    )
                if right_eye[2] > self.keypoint_confidence:
                    face_keypoints.append(
                        [right_eye[0].item(), right_eye[1].item()]
                    )
                if left_ear[2] > self.keypoint_confidence:
                    face_keypoints.append(
                        [left_ear[0].item(), left_ear[1].item()]
                    )
                if right_ear[2] > self.keypoint_confidence:
                    face_keypoints.append(
                        [right_ear[0].item(), right_ear[1].item()]
                    )

                # Need at least 2 facial keypoints to create bounding box
                if len(face_keypoints) >= 2:
                    face_keypoints = np.array(face_keypoints)

                    # Calculate bounding box based on facial keypoints
                    x_min = int(face_keypoints[:, 0].min())
                    x_max = int(face_keypoints[:, 0].max())
                    y_min = int(face_keypoints[:, 1].min())
                    y_max = int(face_keypoints[:, 1].max())

                    # Expand to capture complete face
                    # Padding based on distance between keypoints
                    if len(face_keypoints) > 2:
                        # Calculate average distance between keypoints
                        distances = []
                        for i in range(len(face_keypoints)):
                            for j in range(i + 1, len(face_keypoints)):
                                dist = np.linalg.norm(
                                    face_keypoints[i] - face_keypoints[j]
                                )
                                distances.append(dist)
                        avg_distance = (
                            np.mean(distances) if distances else 30
                        )
                        # 30% of average distance
                        padding = int(avg_distance * 0.3)
                    else:
                        padding = 30

                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame_width, x_max + padding)
                    y_max = min(frame_height, y_max + padding)

                    # Calculate average confidence of keypoints
                    keypoint_confs = [
                        nose[2].item(),
                        (left_eye[2].item()
                         if left_eye[2] > self.keypoint_confidence else 0),
                        (right_eye[2].item()
                         if right_eye[2] > self.keypoint_confidence else 0),
                    ]
                    avg_confidence = np.mean(
                        [c for c in keypoint_confs if c > 0]
                    )

                    # Create BoundingBox
                    bounding_box = BoundingBox(
                        x=x_min,
                        y=y_min,
                        width=x_max - x_min,
                        height=y_max - y_min
                    )

                    face = FaceDetection(
                        bounding_box=bounding_box,
                        confidence=float(avg_confidence)
                    )

                    faces.append(face)

        return faces
