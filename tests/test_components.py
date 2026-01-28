import sys
from unittest.mock import MagicMock, Mock

# Mock heavy dependencies globally before importing src
# We need to set up specific mocks for classes we use
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
from src.editor import Editor
import importlib
import src.editor

class TestComponents(unittest.TestCase):
    
    def setUp(self):
        # Reload src.editor to ensure it uses the current global mocks
        importlib.reload(src.editor)

        # Reset mocks if needed, or setup common return values
        self.mock_cv2 = sys.modules["cv2"]
        mock_video = MagicMock()
        mock_video.get.return_value = 100
        mock_video.read.return_value = (True, MagicMock(shape=(100,100,3)))
        mock_video.isOpened.return_value = True
        self.mock_cv2.VideoCapture.return_value = mock_video
        self.mock_cv2.imencode.return_value = (True, b'data')

        self.mock_openai = sys.modules["openai"]
        self.mock_requests = sys.modules["requests"]
        self.mock_requests.reset_mock()
        self.mock_requests.post.side_effect = None
        # Define a real exception class for RequestException so it can be caught
        class MockRequestException(Exception):
            pass
        self.mock_requests.RequestException = MockRequestException

        self.mock_moviepy_editor = sys.modules["moviepy.editor"]
        self.mock_whisper = sys.modules["whisper"]
        self.mock_pil = sys.modules["PIL"]

    def test_transcriber(self):
        print("Testing Transcriber (Mocked)...")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello", "segments": []}
        self.mock_whisper.load_model.return_value = mock_model
        
        t = Transcriber(model_size="base")
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
            a = Analyzer(api_key="key")
            res = a.analyze("dummy.mp4", {"text": "hi", "segments": []})
            self.assertIsNotNone(res)
            self.assertIn("segments", res)

    def test_analyzer_json_failure(self):
        print("Testing Analyzer JSON Failure (Mocked)...")
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = 'Invalid JSON'
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        with patch('os.path.exists', return_value=True):
            a = Analyzer(api_key="key")
            res = a.analyze("dummy.mp4", {"text": "hi", "segments": []})
            self.assertIsNone(res)

    def test_analyzer_fallback_substring(self):
        print("Testing Analyzer Fallback Substring (Mocked)...")
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        # JSON hidden in garbage
        mock_completion.choices[0].message.content = 'Some text { "segments": [] } more text'
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.OpenAI.return_value = mock_client

        with patch('os.path.exists', return_value=True):
            a = Analyzer(api_key="key")
            res = a.analyze("dummy.mp4", {"text": "hi", "segments": []})
            self.assertIsNotNone(res)
            self.assertIn("segments", res)

    def test_generator_success(self):
        print("Testing Generator Success (Mocked)...")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'fakeimagebytes'
        self.mock_requests.post.return_value = mock_response
        
        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img
        
        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                g = Generator(api_token="token")
                path = g.generate("prompt")
                self.assertIsNotNone(path)
                mock_img.save.assert_called()

    def test_generator_failure(self):
        print("Testing Generator Failure (Mocked)...")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Error"
        self.mock_requests.post.return_value = mock_response

        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                g = Generator(api_token="token")
                path = g.generate("prompt")
                self.assertIsNone(path)

    def test_generator_custom_output_dir(self):
        print("Testing Generator Custom Output Dir (Mocked)...")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'fakeimagebytes'
        self.mock_requests.post.return_value = mock_response

        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img

        # Test with custom dir
        custom_dir = "my_uploads"
        with patch('os.path.exists', return_value=False): # Force makedirs
            with patch('os.makedirs') as mock_makedirs:
                g = Generator(api_token="token", output_dir=custom_dir)
                mock_makedirs.assert_called_with(custom_dir)

                path = g.generate("prompt")
                self.assertIn(custom_dir, path)

    def test_generator_503_invalid_json(self):
        print("Testing Generator 503 Invalid JSON (Mocked)...")

        # Mock 503 response with invalid JSON
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.json.side_effect = Exception("Invalid JSON") # Simulate JSON decode error

        # Mock successful response eventually
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.content = b'fakeimagebytes'

        self.mock_requests.post.side_effect = [mock_response_503, mock_response_success]

        mock_img = MagicMock()
        self.mock_pil.Image.open.return_value = mock_img

        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                with patch('time.sleep') as mock_sleep: # Mock sleep to avoid waiting
                    g = Generator(api_token="token")
                    path = g.generate("prompt")
                    self.assertIsNotNone(path)
                    mock_sleep.assert_called_with(20) # Should default to 20s

    def test_editor(self):
        print("Testing Editor (Mocked)...")

        # Mock VideoFileClip
        mock_clip = MagicMock()
        mock_clip.duration = 10.0
        mock_clip.w = 100
        mock_clip.h = 100
        mock_clip.subclip.return_value = mock_clip # subclip returns itself
        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        # Mock TextClip
        self.mock_moviepy_editor.TextClip.return_value = MagicMock()
        # Mock ImageClip
        self.mock_moviepy_editor.ImageClip.return_value = MagicMock()
        # Mock CompositeVideoClip
        mock_comp = MagicMock()
        mock_comp.set_duration.return_value = mock_comp
        self.mock_moviepy_editor.CompositeVideoClip.return_value = mock_comp

        # Mock concatenate_videoclips
        mock_final = MagicMock()
        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_final

        editor = Editor()
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

    def test_editor_new_features(self):
        print("Testing Editor New Features (Mocked)...")

        # Need to reload module to ensure it captures new mocks if any
        import importlib
        import src.editor
        importlib.reload(src.editor)
        from src.editor import Editor

        # Manually align the module's imported vfx/afx with our global mocks
        # This is necessary because re-importing submodules of mocked packages can be inconsistent
        src.editor.vfx = sys.modules["moviepy.video.fx.all"]
        src.editor.afx = sys.modules["moviepy.audio.fx.all"]

        # Mocks
        mock_clip = MagicMock()
        mock_clip.duration = 10.0
        mock_clip.w = 100
        mock_clip.h = 100
        mock_clip.subclip.return_value = mock_clip
        mock_clip.crossfadein.return_value = mock_clip # Chainable
        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        # Mock CompositeVideoClip set_duration
        mock_comp = MagicMock()
        mock_comp.set_duration.return_value = mock_comp
        self.mock_moviepy_editor.CompositeVideoClip.return_value = mock_comp

        mock_final = MagicMock()
        mock_final.set_audio.return_value = mock_final # Chainable
        mock_final.duration = 20.0 # Set explicit duration for comparison
        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_final

        # Mock Audio
        mock_audio = MagicMock()
        mock_audio.duration = 5.0
        mock_audio.volumex.return_value = mock_audio
        mock_audio.subclip.return_value = mock_audio
        self.mock_moviepy_editor.AudioFileClip.return_value = mock_audio

        # Mock fx
        mock_vfx = sys.modules["moviepy.video.fx.all"]
        mock_vfx.blackwhite.return_value = mock_final

        mock_afx = sys.modules["moviepy.audio.fx.all"]
        mock_afx.audio_loop.return_value = mock_audio

        editor = Editor()
        # 2 segments to trigger crossfade logic
        analysis_data = {
            "segments": [{"start": 0, "end": 5}, {"start": 5, "end": 10}],
        }

        subtitle_config = {"font": "Arial", "fontsize": 50}

        with patch('os.path.exists', return_value=True):
            editor.edit("dummy.mp4", analysis_data, {},
                        music="music.mp3", music_volume=0.5, crossfade=1.0,
                        subtitle_config=subtitle_config, visual_filter="black_white")

            # Check music
            self.mock_moviepy_editor.AudioFileClip.assert_called_with("music.mp3")
            mock_audio.volumex.assert_called_with(0.5)

            # Check filter
            mock_vfx.blackwhite.assert_called()

            # Check crossfade
            # crossfadein should be called on the second clip (which is mock_clip)
            mock_clip.crossfadein.assert_called_with(1.0)

        # Re-run with caption to test subtitle config
        analysis_data["captions"] = [{"start": 0, "end": 2, "text": "Test"}]
        mock_text_clip = MagicMock()
        mock_text_clip.set_start.return_value = mock_text_clip
        mock_text_clip.set_duration.return_value = mock_text_clip
        mock_text_clip.set_position.return_value = mock_text_clip
        self.mock_moviepy_editor.TextClip.return_value = mock_text_clip

        with patch('os.path.exists', return_value=True):
            editor.edit("dummy.mp4", analysis_data, {}, subtitle_config=subtitle_config)
            self.mock_moviepy_editor.TextClip.assert_called()
            call_args = self.mock_moviepy_editor.TextClip.call_args
            self.assertEqual(call_args[1]['font'], "Arial")
            self.assertEqual(call_args[1]['fontsize'], 50)

if __name__ == '__main__':
    unittest.main()
