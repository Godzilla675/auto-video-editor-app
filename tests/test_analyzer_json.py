import sys
from unittest.mock import MagicMock, patch
import unittest
import os
import json

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

import src.analyzer
import importlib
from src.analyzer import Analyzer

class TestAnalyzerJSON(unittest.TestCase):

    def setUp(self):
        importlib.reload(src.analyzer)
        self.mock_openai = sys.modules["openai"]
        # Mock APIError
        self.mock_openai.APIError = Exception
        # We delay Analyzer instantiation or re-instantiate in tests if we need specific client mocks
        # But let's just instantiate here and we can replace self.analyzer.client in tests
        self.analyzer = Analyzer(api_key="key")

    def test_json_in_markdown(self):
        print("Testing JSON in Markdown...")

        response_text = """
        Here is the plan:
        ```json
        {
            "segments": [{"start": 0, "end": 10}]
        }
        ```
        Hope you like it.
        """

        mock_client = MagicMock()
        mock_completion = MagicMock()

        choice_mock = MagicMock()
        choice_mock.message.content = response_text
        mock_completion.choices = [choice_mock]

        mock_client.chat.completions.create.return_value = mock_completion

        # Inject the mock client into the analyzer instance
        self.analyzer.client = mock_client

        with patch('src.analyzer.APIError', Exception):
            with patch('os.path.exists', return_value=True):
                with patch.object(self.analyzer, '_extract_frames', return_value=[]):
                    res = self.analyzer.analyze("dummy.mp4", {})

        self.assertIsNotNone(res)
        self.assertEqual(res["segments"][0]["start"], 0)

    def test_json_mixed_content(self):
        print("Testing JSON mixed content...")

        response_text = """
        Sure, here is the JSON:
        {
            "segments": [{"start": 0, "end": 10}]
        }
        """

        mock_client = MagicMock()
        mock_completion = MagicMock()

        choice_mock = MagicMock()
        choice_mock.message.content = response_text
        mock_completion.choices = [choice_mock]

        mock_client.chat.completions.create.return_value = mock_completion
        self.analyzer.client = mock_client

        with patch('src.analyzer.APIError', Exception):
            with patch('os.path.exists', return_value=True):
                with patch.object(self.analyzer, '_extract_frames', return_value=[]):
                    res = self.analyzer.analyze("dummy.mp4", {})

        if res is None:
            print("FAILURE: Failed to parse mixed content JSON.")
        else:
            print("SUCCESS: Parsed mixed content JSON.")

        self.assertIsNotNone(res, "Should be able to parse JSON embedded in text without markdown blocks")

if __name__ == '__main__':
    unittest.main()
