from PIL import Image

class ImageScaler:
    """A utility class to scale images."""

    def __init__(self, max_width: int, max_height: int):
        """
        Initializes the scaler with maximum dimensions.

        Args:
            max_width: The maximum width for the scaled image.
            max_height: The maximum height for the scaled image.
        """
        self.max_width = max_width
        self.max_height = max_height

    def scale(self, image_path: str) -> Image.Image:
        """
        Scales an image to fit within the maximum dimensions while preserving aspect ratio.

        Args:
            image_path: The path to the image file.

        Returns:
            A Pillow Image object of the scaled image.
        """
        img = Image.open(image_path)
        img.thumbnail((self.max_width, self.max_height))
        return img