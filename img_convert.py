import cv2
import numpy as np

def convert_image_to_data(image_source):
    """
    Reads an image from a file path or a file storage object,
    converts it to grayscale, resizes it to 50x50, and flattens it
    into a 1D numpy array.

    Args:
        image_source (str or FileStorage): The file path to the image or the
                                           image file object from Flask's request.files.

    Returns:
        numpy.ndarray: The flattened image data, or None if the image cannot be read.
    """
    if isinstance(image_source, str):
        # If the source is a string, treat it as a file path
        img = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_source, np.ndarray):
        img = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
    else:
        # Otherwise, assume it's a file-like object
        in_memory_file = image_source.read()
        np_array = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Could not read image. Skipping.")
        return None

    # Resize the image to a fixed size (e.g., 50x50)
    img_resized = cv2.resize(img, (50, 50))

    # Flatten the image into a 1D array
    img_flattened = img_resized.flatten()

    return img_flattened