import sys
from unittest.mock import MagicMock, Mock, patch
import os
import unittest
import json

# Mock heavy dependencies globally before importing src
# We need to set up specific mocks for classes we use
mock_moviepy = MagicMock()
mock_moviepy_editor = MagicMock()
sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = mock_moviepy_editor
sys.modules["moviepy.config"] = MagicMock()

sys.modules["whisper"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transcriber import Transcriber
from src.analyzer import Analyzer
from src.generator import Generator
from src.editor import Editor

class TestComponents(unittest.TestCase):
    
    def setUp(self):
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
        self.mock_moviepy_editor = sys.modules["moviepy.editor"]
        self.mock_whisper = sys.modules["whisper"]
        self.mock_pil = sys.modules["PIL"]

        # Setup specific mocks for editor features
        self.mock_moviepy_editor.VideoFileClip.return_value = MagicMock()
        self.mock_moviepy_editor.AudioFileClip.return_value = MagicMock()
        self.mock_moviepy_editor.CompositeAudioClip.return_value = MagicMock()
        self.mock_moviepy_editor.afx = MagicMock()
        self.mock_moviepy_editor.vfx = MagicMock()

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
        self.mock_moviepy_editor.CompositeVideoClip.return_value = MagicMock()
        # Mock concatenate_videoclips
        mock_final = MagicMock()
        mock_final.w = 100
        mock_final.h = 100
        mock_final.duration = 10.0
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

    def test_editor_enhanced(self):
        print("Testing Editor Enhanced Features (Mocked)...")

        # Mock VideoFileClip
        mock_clip = MagicMock()
        mock_clip.duration = 10.0
        mock_clip.w = 100
        mock_clip.h = 100
        mock_clip.subclip.return_value = mock_clip
        # FX chain
        mock_clip.fx.return_value = mock_clip

        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        mock_final = MagicMock()
        mock_final.w = 100
        mock_final.h = 100
        mock_final.duration = 10.0
        # FX chain for final
        mock_final.set_audio.return_value = mock_final

        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_final

        # Mock Audio
        mock_audio = MagicMock()
        mock_audio.duration = 5.0
        mock_audio.volumex.return_value = mock_audio
        mock_audio.subclip.return_value = mock_audio
        self.mock_moviepy_editor.AudioFileClip.return_value = mock_audio
        self.mock_moviepy_editor.afx.audio_loop.return_value = mock_audio

        editor = Editor()
        analysis_data = {
            "segments": [{"start": 0, "end": 5}, {"start": 6, "end": 10}], # 2 segments to test transition
        }

        with patch('os.path.exists', return_value=True):
            output = editor.edit(
                "dummy.mp4",
                analysis_data,
                {},
                background_music="music.mp3",
                intro_path="intro.mp4",
                outro_path="outro.mp4",
                transition_duration=1.0
            )

            self.assertEqual(output, "output.mp4")
            # Verify music loaded
            self.mock_moviepy_editor.AudioFileClip.assert_called_with("music.mp3")
            # Verify intro loaded
            self.mock_moviepy_editor.VideoFileClip.assert_any_call("intro.mp4")
            # Verify outro loaded
            self.mock_moviepy_editor.VideoFileClip.assert_any_call("outro.mp4")
            # Verify transition fx called
            # clips[0].fx(vfx.fadeout, 1.0)
            # clips[-1].fx(vfx.fadein, 1.0)
            # clips in between fadein and fadeout
            # We have 2 segments -> 2 clips.
            # Clip 0: fadeout. Clip 1: fadein.
            # Check if fx was called
            self.assertTrue(mock_clip.fx.called)

if __name__ == '__main__':
    unittest.main()
