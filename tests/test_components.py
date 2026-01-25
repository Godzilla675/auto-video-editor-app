import sys
from unittest.mock import MagicMock, Mock
import importlib

# Mock heavy dependencies globally before importing src
# We need to set up specific mocks for classes we use
mock_moviepy = MagicMock()
mock_moviepy_editor = MagicMock()
sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = mock_moviepy_editor
sys.modules["moviepy.config"] = MagicMock()

# Mock fx submodules
mock_vfx = MagicMock()
mock_afx = MagicMock()
sys.modules["moviepy.video.fx.all"] = mock_vfx
sys.modules["moviepy.audio.fx.all"] = mock_afx

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

import src.transcriber
import src.analyzer
import src.generator
import src.editor

class TestComponents(unittest.TestCase):
    
    def setUp(self):
        # Reload modules to ensure they use the current mocks
        importlib.reload(src.transcriber)
        importlib.reload(src.analyzer)
        importlib.reload(src.generator)
        importlib.reload(src.editor)

        self.Transcriber = src.transcriber.Transcriber
        self.Analyzer = src.analyzer.Analyzer
        self.Generator = src.generator.Generator
        self.Editor = src.editor.Editor

        # Capture modules to access their imported mocks
        self.editor_module = src.editor

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
        self.mock_vfx = sys.modules["moviepy.video.fx.all"]
        self.mock_afx = sys.modules["moviepy.audio.fx.all"]

        # Reset mocks
        self.mock_moviepy_editor.reset_mock()
        self.mock_vfx.reset_mock()
        self.mock_afx.reset_mock()

    def test_transcriber(self):
        print("Testing Transcriber (Mocked)...")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello", "segments": []}
        self.mock_whisper.load_model.return_value = mock_model
        
        t = self.Transcriber(model_size="base")
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
            a = self.Analyzer(api_key="key")
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
            a = self.Analyzer(api_key="key")
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
                g = self.Generator(api_token="token")
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
                g = self.Generator(api_token="token")
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
        mock_clip.fx.return_value = mock_clip # fx returns itself
        mock_clip.crossfadein.return_value = mock_clip
        mock_clip.set_audio.return_value = mock_clip
        self.mock_moviepy_editor.VideoFileClip.return_value = mock_clip

        # Mock AudioFileClip
        mock_audio = MagicMock()
        mock_audio.duration = 5.0
        mock_audio.fx.return_value = mock_audio
        mock_audio.subclip.return_value = mock_audio
        self.mock_moviepy_editor.AudioFileClip.return_value = mock_audio
        self.mock_moviepy_editor.CompositeAudioClip.return_value = MagicMock()

        # Mock TextClip
        self.mock_moviepy_editor.TextClip.return_value = MagicMock()
        # Mock ImageClip
        self.mock_moviepy_editor.ImageClip.return_value = MagicMock()
        # Mock CompositeVideoClip
        mock_composite = MagicMock()
        mock_composite.set_duration.return_value = mock_composite
        mock_composite.crossfadein.return_value = mock_composite
        self.mock_moviepy_editor.CompositeVideoClip.return_value = mock_composite

        # Mock concatenate_videoclips
        mock_final = MagicMock()
        mock_final.duration = 15.0
        mock_final.audio = MagicMock()
        mock_final.set_audio.return_value = mock_final
        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_final

        # Setup fx mocks on the loaded module
        self.editor_module.afx.audio_loop.return_value = mock_audio
        self.editor_module.afx.volumex = MagicMock()
        self.editor_module.vfx.blackwhite = MagicMock()

        editor = self.Editor()
        analysis_data = {
            "segments": [{"start": 0, "end": 5}, {"start": 5, "end": 10}],
            "captions": [{"start": 0, "end": 2, "text": "Hello"}],
            "graphics": [{"timestamp": 3, "duration": 2}]
        }
        graphic_paths = {0: "graphic.png"}

        subtitle_config = {"font": "Arial", "fontsize": 50, "color": "red"}

        with patch('os.path.exists', return_value=True):
            output = editor.edit(
                "dummy.mp4",
                analysis_data,
                graphic_paths,
                music_path="music.mp3",
                music_volume=0.5,
                crossfade=1.0,
                visual_filter="black_white",
                subtitle_config=subtitle_config
            )

            self.assertEqual(output, "output.mp4")

            # Assert video loading
            self.mock_moviepy_editor.VideoFileClip.assert_called_with("dummy.mp4")

            # Assert TextClip call args for subtitle customization
            self.mock_moviepy_editor.TextClip.assert_called()
            call_args = self.mock_moviepy_editor.TextClip.call_args[1]
            self.assertEqual(call_args.get("font"), "Arial")
            self.assertEqual(call_args.get("fontsize"), 50)
            self.assertEqual(call_args.get("color"), "red")

            # Assert Music Handling
            self.mock_moviepy_editor.AudioFileClip.assert_called_with("music.mp3")

            # Assert using the module's mock
            self.editor_module.afx.audio_loop.assert_called()
            mock_audio.fx.assert_called() # volumex check implied
            self.mock_moviepy_editor.CompositeAudioClip.assert_called()
            mock_final.set_audio.assert_called()

            # Assert Filter
            # mock_clip.fx.assert_called() # This verifies fx called on clip

            # Assert Crossfade
            # The second segment has no overlays, so it is a basic clip (mock_clip)
            mock_clip.crossfadein.assert_called_with(1.0)

            # Assert Concatenate padding
            self.mock_moviepy_editor.concatenate_videoclips.assert_called()
            concat_kwargs = self.mock_moviepy_editor.concatenate_videoclips.call_args[1]
            self.assertEqual(concat_kwargs.get("padding"), -1.0)

            mock_final.write_videofile.assert_called()

if __name__ == '__main__':
    unittest.main()
