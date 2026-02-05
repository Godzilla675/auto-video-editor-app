
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Mock modules to avoid importing heavy dependencies
mock_moviepy = MagicMock()
sys.modules["moviepy"] = mock_moviepy
sys.modules["moviepy.editor"] = MagicMock()
sys.modules["moviepy.video.fx.all"] = MagicMock()
sys.modules["moviepy.audio.fx.all"] = MagicMock()
sys.modules["moviepy.config"] = MagicMock()

# Mock PIL
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

# Mock requests
mock_requests = MagicMock()
class MockRequestException(Exception): pass
mock_requests.RequestException = MockRequestException
sys.modules["requests"] = mock_requests

class TestFixes(unittest.TestCase):
    def setUp(self):
        pass

    @patch("src.editor.VideoFileClip")
    @patch("src.editor.CompositeVideoClip")
    @patch("src.editor.concatenate_videoclips")
    @patch("src.editor.AudioFileClip")
    @patch("src.editor.CompositeAudioClip")
    @patch("src.editor.vfx")
    @patch("src.editor.afx")
    def test_audio_mixing_with_no_audio_video(self, mock_afx, mock_vfx, MockCompositeAudioClip, MockAudioFileClip, mock_concat, MockCompositeVideoClip, MockVideoFileClip):
        from src.editor import Editor

        # Mock initial video clip
        mock_initial_video = MagicMock()
        mock_initial_video.duration = 100.0
        mock_initial_video.w = 1920
        mock_initial_video.h = 1080
        # Mock subclip to return itself or another mock
        mock_initial_video.subclip.return_value = mock_initial_video

        MockVideoFileClip.return_value = mock_initial_video

        # Mock concatenated video clip (final)
        mock_final_video = MagicMock()
        mock_final_video.duration = 10.0
        mock_final_video.audio = None # Key: No audio on the video
        mock_final_video.write_videofile = MagicMock()
        mock_final_video.set_audio.return_value = mock_final_video

        # When concatenate_videoclips is called, return our mock final video
        mock_concat.return_value = mock_final_video

        # Mock music clip
        mock_music = MagicMock()
        mock_music.duration = 20.0
        mock_music.subclip.return_value = mock_music
        mock_music.volumex.return_value = mock_music
        MockAudioFileClip.return_value = mock_music

        # Mock afx.audio_loop to return the clip itself
        mock_afx.audio_loop.return_value = mock_music

        editor = Editor()

        # Minimal data
        analysis_data = {"segments": [{"start": 0, "end": 10}]}
        graphic_paths = {}

        with open("test_music.mp3", "w") as f:
            f.write("dummy")

        try:
            editor.edit("dummy_video.mp4", analysis_data, graphic_paths, music="test_music.mp3")
        except Exception as e:
            self.fail(f"edit raised Exception unexpectedly: {e}")
        finally:
            if os.path.exists("test_music.mp3"):
                os.remove("test_music.mp3")

        # Verify CompositeAudioClip was called
        call_args = MockCompositeAudioClip.call_args
        if call_args:
            args, _ = call_args
            clip_list = args[0]
            # Check if None is in the list
            if None in clip_list:
                print("Confirmed: CompositeAudioClip called with None in list")
            else:
                print("CompositeAudioClip called cleanly")
        else:
             print("CompositeAudioClip not called")

    def test_generator_503_invalid_json(self):
        from src.generator import Generator

        mock_post = mock_requests.post

        # Setup mock to return 503 first, then 200
        # 503 response with non-JSON content
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.text = "Service Unavailable"
        mock_response_503.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.content = b"fake_image_bytes"

        mock_post.side_effect = [mock_response_503, mock_response_200]

        # We need to ensure we can patch src.generator.Image which comes from PIL.Image
        # Since we mocked PIL.Image in sys.modules, 'from PIL import Image' gives us that mock.

        with patch("src.generator.Image") as MockImage:
             with patch("src.generator.io"):
                mock_img_instance = MagicMock()
                MockImage.open.return_value = mock_img_instance

                gen = Generator(api_token="test_token")

                try:
                    with patch("src.generator.time.sleep") as mock_sleep:
                        path = gen.generate("test prompt")
                except Exception as e:
                     print(f"Generator crashed as expected: {e}")
                     return

                # If it didn't crash, we need to know what happened
                if path:
                    print("Generator recovered and produced path")
                else:
                    print("Generator returned None")

if __name__ == "__main__":
    unittest.main()
