import sys
from unittest.mock import MagicMock

# Mock heavy dependencies globally before importing src
sys.modules["whisper"] = MagicMock()
sys.modules["moviepy"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["numpy"] = MagicMock()

import unittest
import os
import json
from unittest.mock import patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transcriber import Transcriber
from src.analyzer import Analyzer
from src.generator import Generator

class TestComponents(unittest.TestCase):
    
    def test_transcriber(self):
        print("Testing Transcriber (Mocked)...")
        # Setup the mock that was injected into sys.modules
        mock_whisper = sys.modules["whisper"]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello", "segments": []}
        mock_whisper.load_model.return_value = mock_model
        
        t = Transcriber(model_size="base")
        with patch('os.path.exists', return_value=True):
            res = t.transcribe("dummy.mp4")
            self.assertEqual(res["text"], "hello")

    def test_analyzer(self):
        print("Testing Analyzer (Mocked)...")
        mock_openai = sys.modules["openai"]
        mock_cv2 = sys.modules["cv2"]
        
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = '{"segments": [], "captions": []}'
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.OpenAI.return_value = mock_client
        
        # Mock CV2
        mock_video = MagicMock()
        mock_video.get.return_value = 100
        mock_video.read.return_value = (True, MagicMock(shape=(100,100,3)))
        mock_cv2.VideoCapture.return_value = mock_video
        mock_cv2.imencode.return_value = (True, b'data')
        
        with patch('os.path.exists', return_value=True):
            a = Analyzer(api_key="key")
            res = a.analyze("dummy.mp4", {"text": "hi"})
            self.assertIsNotNone(res)
            self.assertIn("segments", res)

    def test_generator(self):
        print("Testing Generator (Mocked)...")
        mock_requests = sys.modules["requests"]
        mock_pil = sys.modules["PIL"]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'fakeimagebytes'
        mock_requests.post.return_value = mock_response
        
        mock_img = MagicMock()
        mock_pil.Image.open.return_value = mock_img
        
        with patch('os.path.exists', return_value=True):
            # Also patch os.makedirs
            with patch('os.makedirs'):
                g = Generator(api_token="token")
                path = g.generate("prompt")
                self.assertIsNotNone(path)
                mock_img.save.assert_called()

if __name__ == '__main__':
    unittest.main()