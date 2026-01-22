import sys
from unittest.mock import MagicMock, patch
import unittest
import os
import time

# Mock heavy dependencies
sys.modules["moviepy"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()
sys.modules["moviepy.config"] = MagicMock()
sys.modules["whisper"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.generator
import importlib

from src.generator import Generator

class TestGeneratorRetry(unittest.TestCase):

    def setUp(self):
        importlib.reload(src.generator)
        self.mock_requests = sys.modules["requests"]
        self.mock_requests.reset_mock()
        self.mock_pil = sys.modules["PIL"]
        self.generator = Generator(api_token="token")

    def test_retry_on_500(self):
        print("Testing Generator Retry on 500...")

        # Setup mock to fail with 500 first, then succeed with 200
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.text = "Internal Error"

        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.content = b'image'

        # Side effect: first call returns fail, second returns success
        self.mock_requests.post.side_effect = [mock_fail, mock_success]

        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img

        with patch('os.path.exists', return_value=True), patch('time.sleep') as mock_sleep:
             res = self.generator.generate("prompt")

        self.assertIsNotNone(res)
        self.assertEqual(self.mock_requests.post.call_count, 2)
        mock_sleep.assert_called()

    def test_retry_on_exception(self):
        print("Testing Generator Retry on Exception...")

        # Side effect: first call raises exception, second returns success
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.content = b'image'

        self.mock_requests.post.side_effect = [Exception("Connection Error"), mock_success]

        # We need to mock RequestException to be caught
        self.mock_requests.RequestException = Exception

        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img

        with patch('os.path.exists', return_value=True), patch('time.sleep') as mock_sleep:
             res = self.generator.generate("prompt")

        self.assertIsNotNone(res)
        self.assertEqual(self.mock_requests.post.call_count, 2)

if __name__ == '__main__':
    unittest.main()
