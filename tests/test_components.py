import sys
import unittest
import os
import json
import importlib
from unittest.mock import MagicMock, Mock, patch

# Mock heavy dependencies globally before importing src
# We need to set up specific mocks for classes we use
sys.modules["moviepy"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()
sys.modules["moviepy.config"] = MagicMock()
sys.modules["moviepy.video.fx.all"] = MagicMock()
sys.modules["moviepy.audio.fx.all"] = MagicMock()

sys.modules["whisper"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.transcriber
import src.analyzer
import src.generator
import src.editor

class TestComponents(unittest.TestCase):
    
    def setUp(self):
        # Reload modules to capture fresh mocks
        importlib.reload(src.transcriber)
        importlib.reload(src.analyzer)
        importlib.reload(src.generator)
        importlib.reload(src.editor)

        self.mock_cv2 = sys.modules["cv2"]
        mock_video = MagicMock()
        mock_video.get.return_value = 100
        mock_video.read.return_value = (True, MagicMock(shape=(100,100,3)))
        mock_video.isOpened.return_value = True
        self.mock_cv2.VideoCapture.return_value = mock_video
        self.mock_cv2.imencode.return_value = (True, b'data')

        self.mock_openai = sys.modules["openai"]
        self.mock_requests = sys.modules["requests"]
        self.mock_moviepy_editor = sys.modules["moviepy.editor"]
        self.mock_whisper = sys.modules["whisper"]
        self.mock_pil = sys.modules["PIL"]

        # Reset mocks to avoid state leakage between tests
        self.mock_openai.reset_mock()
        self.mock_requests.reset_mock()
        self.mock_moviepy_editor.reset_mock()
        self.mock_whisper.reset_mock()
        self.mock_pil.reset_mock()
        self.mock_cv2.reset_mock()

    def test_transcriber(self):
        print("Testing Transcriber (Mocked)...")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello", "segments": []}
        self.mock_whisper.load_model.return_value = mock_model
        
        t = src.transcriber.Transcriber(model_size="base")
        with patch('os.path.exists', return_value=True):
            res = t.transcribe("dummy.mp4")
            self.assertEqual(res["text"], "hello")

    def test_analyzer_success(self):
        print("Testing Analyzer Success (Mocked)...")
        
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = '{"segments": [], "captions": []}'
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client
        
        with patch('os.path.exists', return_value=True):
            a = src.analyzer.Analyzer(api_key="key")
            res = a.analyze("dummy.mp4", {"text": "hi", "segments": []})
            self.assertIsNotNone(res)
            self.assertIn("segments", res)

    def test_analyzer_json_fallback(self):
        print("Testing Analyzer JSON Fallback...")
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = 'Here is JSON: {"key": "value"} end.'
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        with patch('os.path.exists', return_value=True):
            a = src.analyzer.Analyzer(api_key="key")
            res = a.analyze("dummy.mp4", {"text": "hi", "segments": []})
            self.assertEqual(res, {"key": "value"})

    def test_analyzer_json_failure(self):
        print("Testing Analyzer JSON Failure (Mocked)...")
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = 'Invalid JSON'
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        with patch('os.path.exists', return_value=True):
            a = src.analyzer.Analyzer(api_key="key")
            res = a.analyze("dummy.mp4", {"text": "hi", "segments": []})
            self.assertIsNone(res)

    def test_generator_success(self):
        print("Testing Generator Success (Mocked)...")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'fakeimagebytes'
        self.mock_requests.post.return_value = mock_response
        self.mock_requests.post.side_effect = None
        
        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img
        
        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                g = src.generator.Generator(api_token="token")
                path = g.generate("prompt")
                self.assertIsNotNone(path)
                mock_img.save.assert_called()

    def test_generator_retry(self):
        print("Testing Generator Retry...")
        # First 500, then 200
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.content = b'fakeimagebytes'

        self.mock_requests.post.side_effect = [mock_response_500, mock_response_200]

        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img

        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                with patch('time.sleep'):
                    g = src.generator.Generator(api_token="token")
                    path = g.generate("prompt")
                    self.assertIsNotNone(path)
                    self.assertEqual(self.mock_requests.post.call_count, 2)

    def test_generator_failure(self):
        print("Testing Generator Failure (Mocked)...")

        mock_response = MagicMock()
        mock_response.status_code = 400 # Non-transient error
        mock_response.text = "Error"
        self.mock_requests.post.return_value = mock_response
        self.mock_requests.post.side_effect = None # Reset side effect

        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                g = src.generator.Generator(api_token="token")
                path = g.generate("prompt")
                self.assertIsNone(path)

    def test_editor(self):
        print("Testing Editor (Mocked)...")

        # Mock VideoFileClip
        mock_clip = MagicMock()
        mock_clip.duration = 10.0
        mock_clip.w = 100
        mock_clip.h = 100
        # Ensure subclip returns a mock with proper dimensions
        sub_mock = MagicMock()
        sub_mock.w = 100
        sub_mock.h = 100
        sub_mock.duration = 5.0
        mock_clip.subclip.return_value = sub_mock

        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        # Mock TextClip
        self.mock_moviepy_editor.TextClip.return_value = MagicMock()
        # Mock ImageClip
        self.mock_moviepy_editor.ImageClip.return_value = MagicMock()
        # Mock CompositeVideoClip
        self.mock_moviepy_editor.CompositeVideoClip.return_value = MagicMock()
        # Mock concatenate_videoclips
        mock_final = MagicMock()
        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_final

        editor = src.editor.Editor()
        analysis_data = {
            "segments": [{"start": 0, "end": 5}],
            "captions": [{"start": 0, "end": 2, "text": "Hello"}],
            "graphics": [{"timestamp": 3, "duration": 2}]
        }
        graphic_paths = {0: "graphic.png"}

        with patch('os.path.exists', return_value=True):
            output = editor.edit("dummy.mp4", analysis_data, graphic_paths)
            self.assertEqual(output, "output.mp4")
            mock_final.write_videofile.assert_called()

    def test_editor_overlap(self):
        print("Testing Editor Overlap Logic...")

        mock_clip = MagicMock()
        mock_clip.duration = 20.0
        sub_mock = MagicMock()
        sub_mock.h = 100
        sub_mock.w = 100
        sub_mock.duration = 5.0
        mock_clip.subclip.return_value = sub_mock
        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        self.mock_moviepy_editor.ImageClip.return_value = MagicMock()
        self.mock_moviepy_editor.CompositeVideoClip.return_value = MagicMock()

        editor = src.editor.Editor()
        # Segment 5-10. Graphic starts at 4, duration 5 (ends 9). Overlaps.
        analysis_data = {
            "segments": [{"start": 5, "end": 10}],
            "graphics": [{"timestamp": 4.0, "duration": 5.0}]
        }
        graphic_paths = {0: "graphic.png"}

        with patch('os.path.exists', return_value=True):
            editor.edit("dummy.mp4", analysis_data, graphic_paths)
            # Should have called ImageClip
            self.mock_moviepy_editor.ImageClip.assert_called()

if __name__ == '__main__':
    unittest.main()
