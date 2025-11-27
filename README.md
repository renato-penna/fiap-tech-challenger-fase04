# Tech Challenge Fase 4 - Video Analysis with AI

A comprehensive video analysis system that performs face detection, emotion analysis, and activity recognition using state-of-the-art AI models.

## Overview

This project implements a complete video analysis pipeline that:
1. Reads video files frame by frame
2. Detects faces in each frame using YOLOv11
3. Analyzes emotions for each detected face using DeepFace
4. Recognizes activities based on pose estimation (YOLOv11)
5. Generates statistical reports in Markdown format

## Technologies

- **Python 3.13**: Main programming language
- **OpenCV**: Video processing (frame reading/writing)
- **YOLOv11 (Ultralytics)**: Face detection and pose estimation
  - Supports multiple people simultaneously
  - Better accuracy with partial occlusion
  - Handles complex poses
- **DeepFace**: Emotion recognition using pre-trained CNNs
- **NumPy, Pandas, Matplotlib**: Data processing and visualization

## Project Structure

```
fiap-tech-challenger-fase04/
├── src/
│   ├── models/              # Data models (domain layer)
│   │   ├── bouding_box.py
│   │   ├── face_detection.py
│   │   ├── emotion_analysis.py
│   │   └── activity_detection.py
│   ├── services/            # Business logic
│   │   ├── video_processor.py
│   │   └── detectors/      # Detection services
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for development
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LucasBiason/fiap-tech-challenger-fase04.git
cd fiap-tech-challenger-fase04
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.services.video_processor import VideoProcessor

# Initialize video processor
processor = VideoProcessor("path/to/video.mp4")

# Get video information
info = processor.get_video_info()
print(f"Video: {info['width']}x{info['height']}")
print(f"FPS: {info['fps']}")
print(f"Duration: {info['duration']:.2f} seconds")

# Process frames
for frame_num, timestamp, frame in processor.get_frames():
    # Process frame here (detection, analysis, etc.)
    print(f"Processing frame {frame_num} at {timestamp:.2f}s")

# Release resources
processor.release()
```

## Development

The project follows Clean Architecture principles:
- **Domain Layer** (`src/models/`): Data structures and business entities
- **Services Layer** (`src/services/`): Business logic and orchestration
- **Utils Layer** (`src/utils/`): Helper functions and utilities

### Data Models

- `BoundingBox`: Represents a bounding box with x, y, width, height
- `FaceDetection`: Face detection result with bounding box and confidence
- `EmotionAnalysis`: Emotion analysis result with emotion name and confidence
- `ActivityDetection`: Activity detection result with activity name and confidence

## Features

### Implemented
- ✅ Video frame-by-frame reading (VideoProcessor)
- ✅ Domain models with complete docstrings
- ✅ Project structure setup
- ✅ YOLOv11 integration (replacing MediaPipe)
- ✅ Face detection with YOLOv11 (FaceDetector)
- ✅ Emotion analysis with DeepFace (EmotionAnalyzer)
- ✅ Activity recognition with pose estimation (ActivityDetector)
- ✅ Complete pipeline (VideoAnalyzer)
- ✅ Report generation (ReportGenerator)
- ✅ Main entry point (main.py)

## Requirements

- Python 3.13+
- OpenCV 4.8.0+
- Ultralytics 8.3.0+
- PyTorch 2.0.0+
- DeepFace 0.0.75+

See `requirements.txt` for complete list.

## License

This project is part of the FIAP Tech Challenge Phase 4.

## Author

Lucas Biason - [GitHub](https://github.com/LucasBiason)

