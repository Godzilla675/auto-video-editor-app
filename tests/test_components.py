import sys
from unittest.mock import MagicMock, Mock

# Mock heavy dependencies globally before importing src
# We need to set up specific mocks for classes we use
mock_moviepy = MagicMock()
mock_moviepy_editor = MagicMock()
sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = mock_moviepy_editor
sys.modules["moviepy.video"] = MagicMock()
sys.modules["moviepy.video.fx"] = MagicMock()
sys.modules["moviepy.video.fx.all"] = MagicMock()
sys.modules["moviepy.audio"] = MagicMock()
sys.modules["moviepy.audio.fx"] = MagicMock()
sys.modules["moviepy.audio.fx.all"] = MagicMock()
sys.modules["moviepy.config"] = MagicMock()

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
        # Reuse existing mock setup from test_editor

        # Mock specific video fx and audio fx
        mock_vfx_all = sys.modules["moviepy.video.fx.all"]
        mock_vfx_all.blackwhite = MagicMock()

        mock_afx_all = sys.modules["moviepy.audio.fx.all"]
        mock_afx_all.audio_loop = MagicMock()

        # Mock AudioFileClip
        mock_audio_clip = MagicMock()
        mock_audio_clip.duration = 5.0 # shorter than video (10.0) to trigger loop logic
        # Mock methods that return self or new clip
        mock_audio_clip.volumex.return_value = mock_audio_clip
        mock_audio_clip.subclip.return_value = mock_audio_clip

        # When AudioFileClip(file) is called, return this mock
        self.mock_moviepy_editor.AudioFileClip.return_value = mock_audio_clip
        # Also mock afx.audio_loop to return a mock
        mock_afx_all.audio_loop.return_value = mock_audio_clip
        self.mock_moviepy_editor.CompositeAudioClip.return_value = MagicMock()

        # Mock VideoFileClip instance
        mock_clip = MagicMock()
        mock_clip.duration = 10.0
        mock_clip.w = 100
        mock_clip.h = 100
        mock_clip.subclip.return_value = mock_clip
        # Mock fx method for fluent interface
        mock_clip.fx.return_value = mock_clip
        # Mock set_audio
        mock_clip.set_audio.return_value = mock_clip
        # Mock crossfadein
        mock_clip.crossfadein.return_value = mock_clip

        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip
        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_clip # final clip is also a clip

        editor = Editor()
        analysis_data = {
            "segments": [{"start": 0, "end": 5}, {"start": 5, "end": 10}],
            "captions": [],
            "graphics": []
        }
        graphic_paths = {}

        with patch('os.path.exists', return_value=True):
            editor.edit(
                "dummy.mp4",
                analysis_data,
                graphic_paths,
                output_path="output_features.mp4",
                background_music="music.mp3",
                music_volume=0.5,
                crossfade_duration=1.0,
                visual_filter="black_white",
                subtitle_config={"fontsize": 50}
            )

            # Verify filter application
            # clip.fx(vfx.blackwhite) should be called
            mock_clip.fx.assert_called()
            # Ideally verify vfx.blackwhite was passed, but it's a mock function

            # Verify crossfade application
            # concatenate_videoclips called with padding=-1.0
            self.mock_moviepy_editor.concatenate_videoclips.assert_called()
            args, kwargs = self.mock_moviepy_editor.concatenate_videoclips.call_args
            self.assertEqual(kwargs.get("padding"), -1.0)

            # Verify background music
            self.mock_moviepy_editor.AudioFileClip.assert_called_with("music.mp3")
            # Verify volume adjustment (volumex is a method on AudioClip)
            # We didn't explicitly mock volumex, but it should be called on the AudioFileClip mock
            # Since AudioFileClip is mocked by self.mock_moviepy_editor.AudioFileClip.return_value
            # And volumex returns a new clip (or same), we check if it was called.

            # Since moviepy objects are fluent, volumex might be returning a new mock object if not configured
            # But normally we just check the call chain.
            # bg_music might have been replaced by audio_loop result

            # If loop was called, bg_music became the result of audio_loop
            # We set audio_loop.return_value = mock_audio_clip
            # So mock_audio_clip.volumex should be called.

            mock_audio_clip.volumex.assert_called_with(0.5)

if __name__ == '__main__':
    unittest.main()
