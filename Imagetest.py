import unittest
from io import BytesIO
from PIL import Image
from MainPage import load_image

class TestLoadImage(unittest.TestCase):
    def test_load_image(self):
        # Test data
        test_image = Image.new("RGB", (256, 256), (255, 0, 0))
        image_path = BytesIO()
        test_image.save(image_path, "JPEG")
        image_path.seek(0)
        # Test load_image
        image, width, height = load_image(image_path)
        self.assertEqual(width, 256)
        self.assertEqual(height, 256)
        self.assertIsInstance(image, Image.Image)
    
if __name__ == '__main__':
    unittest.main()