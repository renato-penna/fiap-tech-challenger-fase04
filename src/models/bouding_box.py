from dataclasses import dataclass


@dataclass
class BoundingBox:
    """
    Represents a bounding box in an image.

    A bounding box defines the location and size of a detected object
    using rectangular coordinates. Used to mark the position of faces,
    people, or other objects in the video.

    Attributes:
        x (int): X coordinate of the top-left corner (in pixels).
        y (int): Y coordinate of the top-left corner (in pixels).
        width (int): Width of the bounding box (in pixels).
        height (int): Height of the bounding box (in pixels).

    Example:
        >>> bbox = BoundingBox(x=100, y=50, width=200, height=300)
        >>> print(f"Object at ({bbox.x}, {bbox.y}), size {bbox.width}x{bbox.height}")
    """
    x: int
    y: int
    width: int
    height: int
