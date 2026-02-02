
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Import requests safely
try:
    import requests
except ImportError:
    requests = None

from src.generator import Generator

class TestGeneratorRetry(unittest.TestCase):
    def setUp(self):
        # Fix for requests.RequestException being a Mock if requests is mocked globally
        if "requests" in sys.modules and isinstance(sys.modules["requests"], MagicMock):
            # Create a real Exception class if it doesn't exist or is a Mock
            req_mock = sys.modules["requests"]
            if not isinstance(req_mock.RequestException, type) or not issubclass(req_mock.RequestException, Exception):
                 class MockRequestException(Exception):
                     pass
                 req_mock.RequestException = MockRequestException

        self.generator = Generator(api_token="dummy")
        self.generator.output_dir = "test_output"
        if not os.path.exists("test_output"):
            os.makedirs("test_output")

    def tearDown(self):
        if os.path.exists("test_output"):
            # Clean up files
            import shutil
            shutil.rmtree("test_output")

    @patch("src.generator.requests.post")
    def test_retry_on_503(self, mock_post):
        # Patching src.generator.requests.post ensures we hit the one used in the module

        response_503 = MagicMock()
        response_503.status_code = 503
        response_503.json.return_value = {"estimated_time": 0.1}

        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"fake_image_data"

        mock_post.side_effect = [response_503, response_200]

        with patch("src.generator.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image_open.return_value = mock_image

            result = self.generator.generate("test prompt")

            self.assertEqual(mock_post.call_count, 2)
            self.assertIsNotNone(result)

    @patch("src.generator.requests.post")
    def test_retry_on_request_exception(self, mock_post):
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"fake_image_data"

        # We need to use the SAME exception class that the module uses
        # If requests is mocked, we set it in setUp.
        # We access it via src.generator.requests.RequestException
        from src.generator import requests as module_requests
        ExcClass = module_requests.RequestException

        # If ExcClass is still a MagicMock (because setUp didn't fix it or import happened differently),
        # side_effect won't work as expected.
        # But setUp should have fixed sys.modules["requests"].

        mock_post.side_effect = [ExcClass("Fail"), response_200]

        with patch("src.generator.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image_open.return_value = mock_image

            result = self.generator.generate("test prompt")

            self.assertEqual(mock_post.call_count, 2)
            self.assertIsNotNone(result)

    @patch("src.generator.requests.post")
    def test_retry_on_500(self, mock_post):
        response_500 = MagicMock()
        response_500.status_code = 500

        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.content = b"fake_image_data"

        mock_post.side_effect = [response_500, response_200]

        with patch("src.generator.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image_open.return_value = mock_image

            with patch("time.sleep") as mock_sleep:
                result = self.generator.generate("test prompt")

                self.assertEqual(mock_post.call_count, 2)
                self.assertIsNotNone(result)

    @patch("src.generator.requests.post")
    def test_fail_after_max_retries(self, mock_post):
        response_500 = MagicMock()
        response_500.status_code = 500

        mock_post.return_value = response_500

        with patch("time.sleep") as mock_sleep:
            result = self.generator.generate("test prompt")

            self.assertEqual(mock_post.call_count, 5)
            self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
