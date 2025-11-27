"""
Visualization module for face detection in videos.

This module contains reusable functions for:
- Drawing bounding boxes on frames
- Visualizing detections in real-time
- Saving processed videos
- Displaying frames with matplotlib

All functions are independent and can be used in different contexts.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from src.models.face_detection import FaceDetection
from src.models.bouding_box import BoundingBox


# Default colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


def draw_bounding_box(
    frame: np.ndarray,
    bbox: BoundingBox,
    color: Tuple[int, int, int] = COLOR_GREEN,
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draws a bounding box on a frame.

    This is a pure function (does not modify the original frame) and returns
    a copy with the bounding box drawn.

    Args:
        frame: Video frame (numpy array, BGR format)
        bbox: BoundingBox to be drawn
        color: Rectangle color in BGR format (default: green)
        thickness: Line thickness (default: 2)
        label: Optional text to display above the bounding box

    Returns:
        Frame with the bounding box drawn (copy, does not modify original)

    Example:
        >>> frame_with_box = draw_bounding_box(frame, bbox, label="Face 0.95")
    """
    frame_copy = frame.copy()

    # Draw rectangle
    cv2.rectangle(
        frame_copy,
        (bbox.x, bbox.y),
        (bbox.x + bbox.width, bbox.y + bbox.height),
        color,
        thickness
    )

    # Add label if provided
    if label:
        # Calculate text position (above the bounding box)
        text_y = max(bbox.y - 10, 20)  # Ensure it doesn't go off screen

        cv2.putText(
            frame_copy,
            label,
            (bbox.x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame_copy


def draw_face_detections(
    frame: np.ndarray,
    faces: List[FaceDetection],
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draws multiple bounding boxes for detected faces in a frame.

    This function is a high-level abstraction that uses draw_bounding_box
    internally to draw all detected faces.

    Args:
        frame: Video frame (numpy array, BGR format)
        faces: List of FaceDetection with detected faces
        show_confidence: If True, displays confidence above each face

    Returns:
        Frame with all bounding boxes drawn

    Example:
        >>> faces = detector.detect(frame)
        >>> frame_with_faces = draw_face_detections(frame, faces)
    """
    frame_copy = frame.copy()

    for i, face in enumerate(faces):
        bbox = face.bounding_box

        # Create label with confidence if requested
        label = None
        if show_confidence:
            label = f"Face {face.confidence:.2f}"

        # Draw bounding box using reusable function
        frame_copy = draw_bounding_box(
            frame_copy,
            bbox,
            color=COLOR_GREEN,
            thickness=3,
            label=label
        )

    return frame_copy


def add_frame_info(
    frame: np.ndarray,
    frame_number: int,
    total_frames: Optional[int] = None,
    faces_count: Optional[int] = None,
    timestamp: Optional[float] = None
) -> np.ndarray:
    """
    Adds frame information in the top-left corner.

    This function is useful for debugging and visualization, showing information
    such as frame number, timestamp, number of detected faces, etc.

    Args:
        frame: Video frame (numpy array, BGR format)
        frame_number: Current frame number
        total_frames: Total frames in video (optional)
        faces_count: Number of detected faces (optional)
        timestamp: Timestamp in seconds (optional)

    Returns:
        Frame with information added

    Example:
        >>> frame_info = add_frame_info(
        ...     frame,
        ...     frame_num,
        ...     total_frames=1000,
        ...     faces_count=3,
        ...     timestamp=5.2
        ... )
    """
    frame_copy = frame.copy()

    # Build information text
    info_parts = [f"Frame {frame_number}"]

    if total_frames is not None:
        info_parts.append(f"/{total_frames}")

    if timestamp is not None:
        info_parts.append(f" | {timestamp:.2f}s")

    if faces_count is not None:
        info_parts.append(f" | {faces_count} face(s)")

    info_text = "".join(info_parts)

    # Draw text
    cv2.putText(
        frame_copy,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_WHITE,
        2
    )

    return frame_copy


def process_video_with_detections(
    video_processor,
    face_detector,
    output_path: Optional[str] = None,
    show_preview: bool = False,
    preview_window_name: str = "Face Detection - Preview"
) -> dict:
    """
    Processes a complete video detecting faces in each frame.

    This function integrates VideoProcessor and FaceDetector to process
    a complete video, optionally saving the result and/or showing
    real-time preview.

    Args:
        video_processor: Initialized VideoProcessor instance
        face_detector: Initialized FaceDetector instance
        output_path: Path to save processed video (optional)
        show_preview: If True, shows real-time preview
        preview_window_name: Preview window name

    Returns:
        Dictionary with processing statistics:
        {
            'total_frames': int,
            'total_faces': int,
            'avg_faces_per_frame': float,
            'output_path': str (if provided)
        }

    Example:
        >>> processor = VideoProcessor("video.mp4")
        >>> detector = FaceDetector()
        >>> stats = process_video_with_detections(
        ...     processor,
        ...     detector,
        ...     output_path="output.mp4",
        ...     show_preview=True
        ... )
    """
    # Get video information
    video_info = video_processor.get_video_info()
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    total_frames = video_info['frame_count']

    # Create VideoWriter if necessary
    video_writer = None
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

    # Statistics
    frame_count = 0
    total_faces = 0

    # Process each frame
    for frame_num, timestamp, frame in video_processor.get_frames():
        # Detect faces
        faces = face_detector.detect(frame)

        # Draw detections
        frame_with_detections = draw_face_detections(frame, faces)

        # Add frame information
        frame_with_detections = add_frame_info(
            frame_with_detections,
            frame_num,
            total_frames=total_frames,
            faces_count=len(faces),
            timestamp=timestamp
        )

        # Save frame if necessary
        if video_writer:
            video_writer.write(frame_with_detections)

        # Show preview if requested
        if show_preview:
            cv2.imshow(preview_window_name, frame_with_detections)
            # Pause to allow visualization (adjust according to FPS)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to exit
                break

        # Update statistics
        frame_count += 1
        total_faces += len(faces)

        # Show progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% "
                  f"({frame_count}/{total_frames} frames)")

    # Clean up resources
    if video_writer:
        video_writer.release()

    if show_preview:
        cv2.destroyAllWindows()

    # Return statistics
    return {
        'total_frames': frame_count,
        'total_faces': total_faces,
        'avg_faces_per_frame': total_faces / frame_count if frame_count > 0 else 0,
        'output_path': output_path if output_path else None
    }


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Converts frame from BGR (OpenCV) to RGB (matplotlib).

    This function is useful when you want to display OpenCV frames
    using matplotlib, which expects RGB format.

    Args:
        frame: Frame in BGR format (OpenCV)

    Returns:
        Frame in RGB format (matplotlib)

    Example:
        >>> frame_rgb = frame_to_rgb(frame)
        >>> plt.imshow(frame_rgb)
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
