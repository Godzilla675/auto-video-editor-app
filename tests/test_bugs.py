import sys
from unittest.mock import MagicMock, Mock, patch
import unittest
import os
import importlib

# Mock heavy dependencies globally before importing src
mock_moviepy = MagicMock()
mock_moviepy_editor = MagicMock()
sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = mock_moviepy_editor
sys.modules["moviepy.config"] = MagicMock()
sys.modules["moviepy.video.fx.all"] = MagicMock()
sys.modules["moviepy.audio.fx.all"] = MagicMock()

sys.modules["whisper"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
# We need a real-ish Exception for requests to catch it
class MockRequestException(Exception):
    pass
mock_requests = MagicMock()
mock_requests.RequestException = MockRequestException
sys.modules["requests"] = mock_requests
sys.modules["numpy"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.editor
import src.generator
import src.main
from src.editor import Editor
from src.generator import Generator
from src.main import download_video

class TestBugs(unittest.TestCase):

    def setUp(self):
        importlib.reload(src.editor)
        importlib.reload(src.generator)
        importlib.reload(src.main)

        self.mock_moviepy_editor = sys.modules["moviepy.editor"]
        self.mock_requests = sys.modules["requests"]
        self.mock_requests.reset_mock()
        self.mock_pil = sys.modules["PIL"]

        # Align vfx/afx
        src.editor.vfx = sys.modules["moviepy.video.fx.all"]
        src.editor.afx = sys.modules["moviepy.audio.fx.all"]

    def test_editor_no_audio_bug(self):
        print("Testing Editor No Audio Bug...")

        # Mock VideoFileClip
        mock_clip = MagicMock()
        mock_clip.duration = 10.0
        mock_clip.w = 100
        mock_clip.h = 100
        mock_clip.subclip.return_value = mock_clip
        mock_clip.audio = None  # SIMULATE NO AUDIO
        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        # Mock concatenate_videoclips returning a clip with no audio
        mock_final = MagicMock()
        mock_final.duration = 10.0
        mock_final.audio = None # The concatenated clip also has no audio
        mock_final.set_audio.return_value = mock_final
        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_final

        # Mock AudioFileClip
        mock_audio = MagicMock()
        mock_audio.duration = 5.0
        mock_audio.volumex.return_value = mock_audio
        mock_audio.subclip.return_value = mock_audio
        self.mock_moviepy_editor.AudioFileClip.return_value = mock_audio

        # Mock CompositeAudioClip to fail if it receives None
        def composite_audio_side_effect(clips):
            if None in clips:
                raise TypeError("CompositeAudioClip cannot handle None")
            return MagicMock()

        self.mock_moviepy_editor.CompositeAudioClip.side_effect = composite_audio_side_effect

        # Mock fx
        mock_afx = sys.modules["moviepy.audio.fx.all"]
        mock_afx.audio_loop.return_value = mock_audio

        editor = Editor()
        analysis_data = {"segments": [{"start": 0, "end": 5}]}

        with patch('os.path.exists', return_value=True):
            # This should NOT fail now
            try:
                editor.edit("dummy.mp4", analysis_data, {}, music="music.mp3")
            except Exception as e:
                self.fail(f"Caught unexpected error: {e}")

    def test_generator_503_retry_bug(self):
        print("Testing Generator 503 Retry Bug...")

        mock_response = MagicMock()
        mock_response.status_code = 503
        # Simulate non-JSON response (e.g. HTML or text)
        mock_response.text = "Service Unavailable"
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")

        self.mock_requests.post.return_value = mock_response

        g = Generator(api_token="token")

        # This should NOT crash now, but return None after retries
        with patch('time.sleep'): # skip sleep
            result = g.generate("prompt")
            self.assertIsNone(result)

    def test_main_download_timeout(self):
        print("Testing Main Download Timeout...")

        # Mock requests.get context manager
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'chunk']

        mock_get_ctx = MagicMock()
        mock_get_ctx.__enter__.return_value = mock_response
        mock_get_ctx.__exit__.return_value = None

        self.mock_requests.get.return_value = mock_get_ctx

        with patch('builtins.open', new_callable=MagicMock):
            download_video("http://example.com/vid.mp4")

            # Verify timeout was passed
            self.mock_requests.get.assert_called_with("http://example.com/vid.mp4", stream=True, timeout=30)

    def test_generator_request_exception_retry(self):
        print("Testing Generator RequestException Retry...")

        self.mock_requests.post.side_effect = [
            self.mock_requests.RequestException("Fail 1"),
            self.mock_requests.RequestException("Fail 2"),
            MagicMock(status_code=200, content=b'data')
        ]

        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img

        g = Generator(api_token="token")

        with patch('time.sleep'):
            with patch('os.path.exists', return_value=True):
                 with patch('os.makedirs'):
                    result = g.generate("prompt")
                    self.assertIsNotNone(result)
                    self.assertEqual(self.mock_requests.post.call_count, 3)

if __name__ == '__main__':
    unittest.main()
