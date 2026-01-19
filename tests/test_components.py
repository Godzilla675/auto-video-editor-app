import sys
from unittest.mock import MagicMock, Mock

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
        mock_text_clip = MagicMock()
        self.mock_moviepy_editor.TextClip.return_value = mock_text_clip
        mock_text_clip.set_start.return_value = mock_text_clip
        mock_text_clip.set_duration.return_value = mock_text_clip
        mock_text_clip.set_position.return_value = mock_text_clip

        # Mock ImageClip
        mock_image_clip = MagicMock()
        self.mock_moviepy_editor.ImageClip.return_value = mock_image_clip
        mock_image_clip.set_start.return_value = mock_image_clip
        mock_image_clip.set_duration.return_value = mock_image_clip
        mock_image_clip.set_position.return_value = mock_image_clip
        mock_image_clip.resize.return_value = mock_image_clip

        # Mock CompositeVideoClip
        self.mock_moviepy_editor.CompositeVideoClip.return_value = MagicMock()
        # Mock concatenate_videoclips
        mock_final = MagicMock()
        mock_final.duration = 10.0
        # set_audio returns a copy (or self), so we mock it to return self for testing calls
        mock_final.set_audio.return_value = mock_final
        self.mock_moviepy_editor.concatenate_videoclips.return_value = mock_final

        # Mock AudioFileClip
        mock_audio = MagicMock()
        mock_audio.duration = 20.0 # Longer than video
        self.mock_moviepy_editor.AudioFileClip.return_value = mock_audio
        mock_audio.subclip.return_value = mock_audio
        mock_audio.volumex.return_value = mock_audio

        # Mock CompositeAudioClip
        self.mock_moviepy_editor.CompositeAudioClip.return_value = MagicMock()

        editor = Editor()
        analysis_data = {
            "segments": [{"start": 0, "end": 5}, {"start": 6, "end": 10}],
            "captions": [{"start": 0, "end": 2, "text": "Hello"}],
            "graphics": [{"timestamp": 3, "duration": 2}],
            "transitions": [{"timestamp": 5.5, "type": "crossfade"}]
        }
        graphic_paths = {0: "graphic.png"}

        # Use subtitle config
        subtitle_config = {
            "font": "Verdana",
            "fontsize": 50,
            "color": "yellow",
            "stroke_color": "black",
            "stroke_width": 3
        }

        with patch('os.path.exists', return_value=True):
            # Pass all new arguments
            output = editor.edit(
                "dummy.mp4",
                analysis_data,
                graphic_paths,
                music_path="music.mp3",
                music_volume=0.5,
                intro_path="intro.mp4",
                outro_path="outro.mp4",
                subtitle_config=subtitle_config
            )

            self.assertEqual(output, "output.mp4")
            # When we have intro/outro, final_video is created by concatenate_videoclips(sequence)
            # The original mock_final was for concatenate_videoclips(clips) from main content.
            # We need to check if ANY return value from concatenate_videoclips had write_videofile called.
            # Or better, since mock_final is the return value of ALL concatenate_videoclips calls
            # (because we set return_value on the mocked function), it should be the one called.

            # Debugging: print calls
            print("VideoFileClip calls:", self.mock_moviepy_editor.VideoFileClip.call_args_list)
            print("AudioFileClip calls:", self.mock_moviepy_editor.AudioFileClip.call_args_list)
            print("Concatenate calls:", self.mock_moviepy_editor.concatenate_videoclips.call_args_list)

            mock_final.write_videofile.assert_called()

            # Verify TextClip called with custom settings
            self.mock_moviepy_editor.TextClip.assert_called_with(
                "Hello",
                fontsize=50,
                font="Verdana",
                color="yellow",
                stroke_color="black",
                stroke_width=3,
                method='caption',
                size=(90.0, None)
            )

            # Verify Music
            self.mock_moviepy_editor.AudioFileClip.assert_called_with("music.mp3")

            # Verify Intro/Outro loading (VideoFileClip called multiple times)
            # We expect calls for: dummy.mp4 (main), intro.mp4, outro.mp4
            # We can check if VideoFileClip was called with these paths
            calls = self.mock_moviepy_editor.VideoFileClip.call_args_list
            args_list = [c[0][0] for c in calls]
            self.assertIn("dummy.mp4", args_list)
            self.assertIn("intro.mp4", args_list)
            self.assertIn("outro.mp4", args_list)

if __name__ == '__main__':
    unittest.main()
