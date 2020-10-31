"""
Have defined a few exceptions
"""


class ImageNotFoundError(Exception):
    """
    Raised when given path to image doesn't exist
    """


class InvalidImageError(Exception):
    """
    Raised when cv2 couldn't read the file successfully
    """
